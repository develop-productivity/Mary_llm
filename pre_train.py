# TODO:
# train the tokenizer
from transformers import AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments, AutoProcessor
import transformers
from PIL import Image
from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import json
from torch.utils.data import Dataset,IterableDataset, DataLoader
from typing import List, Dict, Any
from dataclasses import dataclass, field
import datasets
import copy


from model import MaryCausalModel, MaryConfig
from utils import rank0_print, count_parameters
os.environ["WANDB_MODE"] = "offline"



@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="facebook/m2m100_418M")
    tokenizer_path: Optional[str] = field(default="tokenizer")
    hidden_size: int = field(default=512)
    num_hidden_layers: int = field(default=8)
    num_attention_heads: int = field(default=8)
    head_dim: int = field(default=64)
    intermediate_size: int = field(default=2048)
    hidden_act: str = field(default="silu")
    num_key_value_heads: int = field(default=4)
    max_seq_len: int = field(default=2048)


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default="datasets/llm/pretrain/mobvoi_seq_monkey_general_open_corpus.jsonl")
    data_max_seq_length: int = field(default=512)
    data_ratio: float = field(default=0.1)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="save/pretrain")
    do_train: bool = field(default=True)
    per_device_train_batch_size: int = field(default=64)
    learning_rate: float = field(default=1e-5)
    num_train_epochs: int = field(default=5)
    ddp_find_unused_parameters: bool = field(default=True)
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=5)
    fp16: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=8)
    dataloader_num_workers: int = field(default=1)
    logging_steps: int = field(default=100)
    report_to: str = field(default='wandb')
    dataloader_pin_memory: bool = field(default=True)
    lora_enable: bool = field(default=False)
    lr_scheduler_type: str = field(default="cosine")
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_target: str= "llm_model\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    freeze_vision_model: bool = field(default=True)
    freeze_llm_model: bool = field(default=True)
    run_name: str = field(default="llm_small_pretrain")
    save_safetensors: bool=field(default=False)
    loss_type: str=field(default="ForCausalLMLoss")


def get_hf_datasets(data_path, tokenizer, max_seq_len=512):

    def encode(examples):
        text = "<s>" + examples["text"] + "</s>"
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        text_len = input_ids.size(1)
        if text_len > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
        else:
            input_ids = F.pad(input_ids, (0, max_seq_len - text_len), value=0)  # dim=-1 left pad 0 and right pad max_seq_len - text_len items
        input_ids = input_ids.squeeze(0)
        labels = input_ids.clone()
        # we shift the labels and input_ids here
        # return {"input_ids": input_ids[:,:-1], "labels": labels[:,1:], "position_ids": torch.tensor(0, dtype=torch.long)}
        return {"input_ids": input_ids, "labels": labels, "position_ids": torch.tensor(0, dtype=torch.long)}

    data_ratio = 0.1
    data_list = []
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        total_lines = len(lines)
        for i, line in enumerate(lines):
            if i >= data_ratio * total_lines:
                break
            data = json.loads(line)
            data_list.append(data)
    
    def batch_encode(examples):
        text = ["<s>" + t + "</s>" for t in examples["text"]]
        input_ids = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=max_seq_len)["input_ids"]
        # input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0.0, padding_side='right')
        bs, text_len = input_ids.shape
        # # text_len = input_ids.size(1)
        if text_len > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
        else:
            input_ids = F.pad(input_ids, (0, max_seq_len - text_len), value=0)
        labels = input_ids.clone()
        # we shift the labels and input_ids here
        # return {"input_ids": input_ids[:, :-1], "labels": labels[:, 1:], "position_ids": torch.tensor([0] * bs, dtype=torch.long)}
        # do not need shift the labels and input_ids here
        return {"input_ids": input_ids, "labels": labels, "position_ids": torch.tensor([0] * bs, dtype=torch.long)}
    hf_dataset = datasets.Dataset.from_list(data_list)
    print(hf_dataset.column_names)
    hf_dataset = hf_dataset.map(batch_encode, batched=True)
    hf_dataset.set_format(type="torch", columns=["input_ids", "labels", "position_ids"])
    return hf_dataset


class LLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=512, data_ratio=0.1):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []
        data_ratio = data_ratio
        # we use 80% of the data for training
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            total_lines = len(lines)
            for i, line in enumerate(lines):
                if i >= data_ratio * total_lines:
                    break
                # data = json.loads(line)
                # self.data.append(data["text"])
                self.data.append(line)
                # count += 1
                # if count > 100:
                #     break

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        line = self.data[idx]
        text = json.loads(line)["text"]
        # text = self.data[idx]
        text = "<s>" + text + "</s>"
        input_ids = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_seq_len)["input_ids"]
        input_ids = input_ids.squeeze(0)
        labels = input_ids.clone()
        # we shift the labels and input_ids here
        # return {"input_ids": input_ids[:-1], "labels": labels[1:], "position_ids": torch.tensor(0, dtype=torch.long)}
        
        # do not need shift the labels and input_ids here
        return {"input_ids": input_ids, "labels": labels, "position_ids": torch.tensor(0, dtype=torch.long)}

#  IterableDataset datasets for memeory friendly
# class LLMDataset(IterableDataset):
#     def __init__(self, data_path, tokenizer, max_seq_len=512):
#         super().__init__()
#         self.data_path = data_path
#         self.tokenizer = tokenizer
#         self.max_seq_len = max_seq_len
        

#     def __iter__(self):
#         return super().data_generator()

#     def data_generator(self):
#         with open(self.data_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 data = json.loads(line)
#                 text = data["text"]
#                 text = "<s>" + text + "</s>"
#                 input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
#                 text_len = input_ids.size(1)
#                 if text_len > self.max_seq_len:
#                     input_ids = input_ids[:, :self.max_seq_len]
#                 else:
#                     input_ids = F.pad(input_ids, (0, self.max_seq_len - text_len), value=0)
#                 input_ids = input_ids.squeeze(0)
#                 labels = input_ids.clone()
#                 yield {"input_ids": input_ids[:-1], "labels": labels[1:], "position_ids": torch.tensor(0, dtype=torch.long)}
    


class DataCollator:
    def __init__(self, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # use the max len in a batch to pad the input_ids and labels
        # max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        for feature in features:
            input_id = feature['input_ids']
            label = feature['labels']
            # 0 is the padding token
            input_id = input_ids + [0] * (self.max_seq_len - len(input_id))
            label = labels + [0] * (self.max_seq_len - len(label))

            input_ids.append(input_id)
            labels.append(label)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}

def test_datasets(datasets):
    dataloader = DataLoader(datasets, batch_size=4)
    dataloader_iter = iter(dataloader)
    for i in range(10):
        print(next(dataloader_iter))
        print(datasets[i])

if __name__ == "__main__":
    paraser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    (
        model_args,
        data_args,
        training_args,
    ) = paraser.parse_args_into_dataclasses()

    config = MaryConfig(
        hidden_size=model_args.hidden_size,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=model_args.num_attention_heads,
        intermediate_size=model_args.intermediate_size,
        hidden_act=model_args.hidden_act,
        num_key_value_heads=model_args.num_key_value_heads,
        max_seq_len=model_args.max_seq_len
        )
    model = MaryCausalModel(config)

    tunable_param, total_param = count_parameters(model)
    rank0_print(f"Total parameters: {total_param}, Tunable parameters: {tunable_param}")

    
    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    # data_collator = DataCollator(tokenizer)
    # train_dataset = get_hf_datasets(data_args.data_path, tokenizer, max_seq_len=512)
    train_dataset = LLMDataset(data_args.data_path, tokenizer, max_seq_len=512, data_ratio=data_args.data_ratio)
    rank0_print(f"Dataset size: {len(train_dataset)}")
    # test_datasets(train_dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()




