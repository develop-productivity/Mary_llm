# copyfrom: transformers.models.llama.modeling_llama.py

import torch.nn as nn 
import torch.nn.functional as F
import torch
import transformers
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.cache_utils import Cache
from torch import Tensor

from typing import Optional, Tuple, Callable, List, Union
import math

logger = transformers.logging.get_logger(__name__)


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)







def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    repeat the key or value tensor n_rep times along the head dimension
    The hidden states go from (batch, seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    torch.repeat_interleave(x, dim=2, repeats=n_rep)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
   
    q_embed = (q*cos) + (rotate_half(q)*sin)
    k_embed = (k*cos) + (rotate_half(k)*sin)
    
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048*2, rope_theta=10000.0):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t @ inv_freq.unsqueeze(0)
        freqs = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
        
    def forward(self, q, k):
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)

class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        # kv cache
        self.cache_k = torch.zeros((config.max_batch_size, config.max_seq_len, config.num_key_value_heads, self.head_dim))
        self.cache_v = torch.zeros((config.max_batch_size, config.max_seq_len, config.num_key_value_heads, self.head_dim))
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_seq_len * 2, config.rope_theta)
        self.use_fsdp=config.use_fsdp if hasattr(config, "use_fsdp") else False
        self.is_causal = config.is_causal if hasattr(config, "is_causal") else True


    def forward(
        self,   
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        start_pos: int = 0,
        use_kv_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, seqlen, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)  # (batch, seqlen, num_heads, head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        
        # TODO: check kv cache
        if use_kv_cache and self.eval() > 0:
            self.cache_k = self.cache_k.to(key_states)
            self.cache_v = self.cache_v.to(value_states)
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = key_states
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = value_states
            key_states = self.cache_k[:bsz, : start_pos + seqlen]
            value_states = self.cache_v[:bsz, : start_pos + seqlen]

        query_states, key_states = self.rotary_emb(query_states, key_states)

        # TODO: implemente attention
        # GQA implementation
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        key_states = key_states.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        value_states = value_states.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)

        # TODO: check the mask and adjust the mask
        # mask = attention_mask
        # if not attention_mask:
        #     attention_mask = torch.full((1, 1, self.config.max_seq_len, self.config.max_seq_len), float("-inf"))  # 初始化掩码
        #     attention_mask = torch.triu(attention_mask, diagonal=1)  # 生成上三角掩码

        if self.use_fsdp:
            output = F.scaled_dot_product_attention(
                    query_states, 
                    key_states,
                    value_states,
                    attn_mask=None, 
                    dropout_p=self.dropout if self.training else 0.0, 
                    is_causal=self.is_causal,
                )
        else:
            output = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                output = output + attention_mask
            else:
                mask = torch.full((1, 1, self.config.max_seq_len, self.config.max_seq_len), float("-inf"))  # 初始化掩码
                mask = torch.triu(mask, diagonal=1)  # 生成上三角掩码
                output = output + mask[:,:, :output.shape[-2], :output.shape[-1]]
            output = F.softmax(output.float(), dim=-1).type_as(query_states)
            output = torch.matmul(output, value_states)

        output = output.transpose(1, 2).contiguous()
        output = output.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(output)
        return output, None


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Attention(config=config, layer_idx=layer_idx)

        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_kv_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            start_pos=position_ids,
            use_kv_cache=use_kv_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        # if output_attentions:
        #     outputs += (self_attn_weights,)

        return outputs # output tuple


class MaryConfig(PretrainedConfig):
    model_type="mary_llm"
    def __init__(
            self, 
            hidden_size=768, 
            vocab_size=6400,
            intermediate_size=3072,
            num_hidden_layers=16,
            num_attention_heads=16,
            head_dim=48, 
            num_key_value_heads=8,
            hidden_act="silu",
            max_position_embeddings=2048,
            max_seq_len=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            max_batch_size=32,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
            use_fsdp=False,
            is_causal=True,
            loss_type="ForCausalLM",
            **kwargs
        ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act=hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len = max_seq_len
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.pad_token_id=pad_token_id
        self.use_fsdp=use_fsdp
        self.is_causal=is_causal
        self.loss_type=loss_type
        super().__init__(**kwargs)


class MaryPreTrainedModel(PreTrainedModel):
    config_class = MaryConfig
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            use_kv_cache: Optional[bool] = False,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
        ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else False

        # if self.gradient_checkpointing and self.training and use_cache:
        #     logger.warning_once(
        #         "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        #     )
        #     use_cache = False
        # # 
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        bsz, seqlen, _ = hidden_states.size()

        # TODO implemente causal_mask
        if attention_mask is not None:
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=hidden_states.device
            )

            causal_mask = torch.triu(causal_mask, diagonal=1)
            # TODO: check the position_ids and support the position_ids in mask modification
            # if position_ids > 0: 
            #     causal_mask = torch.hstack([
            #         torch.zeros((seqlen, position_ids), device=causal_mask.device),
            #         causal_mask
            #     ]).type_as(causal_mask)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    use_kv_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    use_kv_cache=use_kv_cache,
                )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )
        return output if return_dict else output.to_tuple()
    

class MaryCausalModel(MaryPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MaryPreTrainedModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_kv_cache: Optional[bool] = False,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else False

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_kv_cache=use_kv_cache,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        # how to ignore the padding token in loss function
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, ignore_index=torch.tensor(0, dtype=torch.long), **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
    
    @torch.inference_mode()
    def generate(
            self, 
            inputs,
            eos,
            max_gen_len=512,
            use_kv_cache: bool = True,
            temperature: float = 0.7,
            repetition_penalty=1.0,
            top_k: int = None,
            stream: bool = False,
            do_sample: bool = False,
    ):
        """
        Generate a sequence of tokens from the model
        """
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        bsz, seq_len, _ = input_ids.shape
        assert seq_len < self.model.config.max_seq_len, f"Input sequence length {seq_len} exceeds the maximum sequence length {self.model.config.max_seq_len}"

        while input_ids.size(1) < max_gen_len - 1:
            outputs = self.forward(input_ids=input_ids, use_kv_cache=use_kv_cache)
            logits = outputs[0][:, -1, :]
            logits = logits / temperature
            for token in set(input_ids.tolist()[0]):  
                logits[:, token] /= repetition_penalty
            
            if temperature == 0.0: 
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature  
                if top_k is not None:  
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf') 

                probs = F.softmax(logits, dim=-1)  
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)  

            if idx_next == eos:  
                break

            input_ids = torch.cat((input_ids, idx_next), dim=1)  
            if stream:  
                yield input_ids[:, seq_len:]  

        if not stream:  
            yield input_ids[:, seq_len:]  
            

