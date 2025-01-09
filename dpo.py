# TODO:
from transformers import (AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments, AutoConfig, AutoModelForCausalLM)
import datasets
import transformers
from PIL import Image
from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import json
from torch.utils.data import Dataset
from typing import List, Dict, Any
from dataclasses import dataclass, field

from model import MaryCausalModel, MaryConfig
from utils import rank0_print, count_parameters
os.environ["WANDB_MODE"] = "offline"

from trl import DPOTrainer, DPOConfig


trans_chinese = datasets.load_dataset("datasets/llm/dpo/argilla/distilabel-capybara-dpo-7k-binarized", split="train")


# filter chosen length < 18
# trans_chinese = trans_chinese.map(lambda r: {"messages": len(r["chosen"])}).filter(lambda r: r["messages"]<64)

tokenizer = AutoTokenizer.from_pretrained("tokenizer")



def get_hf_datasets(data_path, tokenizer, max_seq_length):
    def llm_format(examples):
        # add generation prompt
        messages = [
                {"role": "user", "content": examples["prompt"]}
            ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        # get the last assistant responses
        chosen = examples["chosen"] + "</s>" 
        rejected = examples["rejected"] + "</s>" 
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
    
    def batch_llm_format(examples):
        # TODO support batch processing
        # add generation prompt
        messages = [
            
                {"role": "user", "content": example["prompt"]}
            for example in examples
            ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        # get the last assistant responses
        chosen = examples["chosen"] + "</s>" 
        rejected = examples["rejected"] + "</s>" 
        return {
            "prompt": [prompt]*len(chosen),
            "chosen": chosen,
            "rejected": rejected   
        }
    
    trans_chinese = datasets.load_dataset(data_path, split="train")
    # filter chosen length < 18
    original_columns = trans_chinese.column_names
    # trans_chinese = trans_chinese.map(lambda r: {"messages": len(r["chosen"])}).filter(lambda r: r["messages"]<64)
    trans_chinese = trans_chinese.map(llm_format, batched=False, remove_columns=original_columns, num_proc=32)
    return trans_chinese

# TODO implement the dataset collator
# class DataCollator:
#     def __init__(self, tokenizer, max_seq_len=512):
#         self.tokenizer = tokenizer
#         self.max_seq_len = max_seq_len
    
#     def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
#         # use the max len in a batch to pad the input_ids and labels
#         # max_len = max(len(feature['input_ids']) for feature in features)
#         input_ids = []
#         labels = []
#         for feature in features:
#             prompt = feature['prompt']
#             chosen = feature['chosen']
#             rejected = feature['rejected']
#             # input_id = feature['input_ids']
#             # 0 is the padding token
            
#             input_ids.append(input_id)
#             labels.append(label)
#         return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}

def test_datasets(datasets):
  print(len(datasets))
  for i in range(10):
    print(datasets[i])


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="save/pretrain")
    tokenizer_path: Optional[str] = field(default="tokenizer")
    max_seq_length: int = field(default=2048)


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default="datasets/llm/dpo/beyond/rlhf-reward-single-round-trans_chinese")
    dataset_max_seq_length: int = field(default=2048)



@dataclass
class TrainingArguments(DPOConfig):
    output_dir: Optional[str] = field(default="save/dpo_1_beyond_trans_chinese")
    do_train: bool = field(default=True)
    max_length: int = field(default=2048)
    # gradient_checkpointing: bool = field(default=False),
    per_device_train_batch_size: int = field(default=8)
    learning_rate: float = field(default=1e-5)
    num_train_epochs: int = field(default=1)
    ddp_find_unused_parameters: bool = field(default=False)
    save_steps: int = field(default=10)
    save_total_limit: int = field(default=2)
    fp16: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=8)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    report_to: str = field(default='wandb')
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    dataset_num_proc: int = field(default=8)
    lora_enable: bool = field(default=False)
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_target: str= "llm_model\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    run_name: str = field(default="llm_pretrain")
    save_safetensors: bool=field(default=False)


if __name__ == "__main__":
    # test_datasets(trans_chinese)
    paraser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    (
        model_args,
        data_args,
        training_args,
    ) = paraser.parse_args_into_dataclasses()
    AutoConfig.register("mary_llm", MaryConfig)
    AutoModelForCausalLM.register(MaryConfig, MaryCausalModel)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    train_datasets = get_hf_datasets(data_args.data_path, tokenizer, data_args.dataset_max_seq_length)
    # test_datasets(train_datasets)
    # ref_model = AutoModelForCausalLM.from_pretrained("save/pretrain")
    # collator = DefaultDataCollator()

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        # data_collator=collator,
        train_dataset=train_datasets,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()