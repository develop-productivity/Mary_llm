# TODO:
from transformers import (AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments, AutoConfig, AutoModelForCausalLM)
import transformers

from typing import Dict, Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import datasets

from model import MaryCausalModel, MaryConfig
from utils import rank0_print, count_parameters
os.environ["WANDB_MODE"] = "offline"



@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="save/pretrain")
    tokenizer_path: Optional[str] = field(default="tokenizer")
    


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default="datasets/llm/sft/deepctrl-sft-data/sft_data_zh.jsonl")
    max_seq_length: int = field(default=1024)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="save/sft")
    do_train: bool = field(default=True)
    per_device_train_batch_size: int = field(default=8)
    learning_rate: float = field(default=1e-5)
    num_train_epochs: int = field(default=5)
    ddp_find_unused_parameters: bool = field(default=False)
    save_steps: int = field(default=3000)
    save_total_limit: int = field(default=5)
    fp16: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=8)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=100)
    report_to: str = field(default='wandb')
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    lora_enable: bool = field(default=False)
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_target: str= "llm_model\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    freeze_vision_model: bool = field(default=True)
    freeze_llm_model: bool = field(default=True)
    run_name: str = field(default="llm_pretrain")
    save_safetensors: bool=field(default=False)
    loss_type: str=field(default="ForCausalLMLoss")


def get_hf_datasets(data_path, tokenizer, max_seq_len=2048):

    def encode(examples):
        instruction_text = examples['instruction']
        input_text = examples['input']
        output_text = examples['output']
        history = examples['history']
        query = instruction_text + input_text
        answer = output_text + tokenizer.eos_token

        messages = []
        if history:
            for i in history:
                messages.append({'role': 'user', 'content': i[0]})
                messages.append({'role': 'assistant', 'content': i[1]})
        messages.append({'role': 'user', 'content': query})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)   
        prompt_input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        answer_input_ids = tokenizer(answer, return_tensors="pt")["input_ids"]

        # TODO check label and input_ids
        labels = torch.cat((torch.zeros(1, prompt_input_ids), answer_input_ids), dim=1)
        input_ids = torch.cat((prompt_input_ids, answer_input_ids), dim=1)
        # labels = [0] * len(prompt_input_ids) + answer_input_ids
        # input_ids = prompt_input_ids + answer_input_ids
        text_len = input_ids.shape[-1]
        if text_len >= max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
            labels = labels[:, :max_seq_len]
        else:
            input_ids = F.pad(input_ids, (0, max_seq_len - text_len), value=0)
            labels = F.pad(labels, (0, max_seq_len - text_len), value=0)
        input_ids = input_ids.squeeze(0)
        labels = labels.squeeze(0)
        # input_ids = torch.tensor(input_ids[:-1])
        # labels = torch.tensor(labels[1:])
        
        return {"input_ids": input_ids, "labels": labels, "position_ids": torch.tensor(0, dtype=torch.long)}
    
    def batch_encode(examples):
        instruction_text = [t for t in examples['instruction']]
        input_text = [t for t in examples['input']]
        output_text = [t for t in examples['output']]
        history = [t for t in examples['history']]
        query = instruction_text + input_text
        answer = output_text + tokenizer.eos_token
        # TODO implement the batch encode
        
        # return {"input_ids": input_ids, "labels": labels, "position_ids": torch.tensor(0, dtype=torch.long)}

    data_ratio = 1
    data_list = []
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        total_lines = len(lines)
        for i, line in enumerate(lines):
            if i >= data_ratio * total_lines:
                break
            data = json.loads(line)
            data_list.append(data)
    
    hf_dataset = datasets.Dataset.from_list(data_list)
    print(hf_dataset.column_names)
    hf_dataset = hf_dataset.map(encode, batched=False)
    return hf_dataset


class SFTLengthDatasets(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer,  max_seq_length: int = 512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_length=max_seq_length
        self.data = []
        data_ratio = 0.1
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i >= data_ratio * len(lines):
                    break
                # data = json.loads(line)
                self.data.append(line)

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        line = self.data[idx]
        data = json.loads(line)
        # data has instruction, input, output, history
        instruction_text = data['instruction']
        input_text = data['input']
        output_text = data['output']
        history = data['history']
        query = instruction_text + input_text
        answer = output_text + self.tokenizer.eos_token

        messages = []
        if history:
            for i in history:
                messages.append({'role': 'user', 'content': i[0]})
                messages.append({'role': 'assistant', 'content': i[1]})
        messages.append({'role': 'user', 'content': query})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)   
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        answer_input_ids = self.tokenizer(answer, return_tensors="pt")["input_ids"]
        # TODO check the + of list and tensor
        # TODO check the + of list and tensor
        labels = torch.cat((torch.zeros(1, prompt_input_ids.shape[1]), answer_input_ids), dim=1)
        input_ids = torch.cat((prompt_input_ids, answer_input_ids), dim=1)
        text_len = input_ids.shape[-1]

        return text_len

        

class SFTDatasets(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer,  max_seq_length: int = 512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_length=max_seq_length
        self.data = []
        data_ratio = 0.1
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i >= data_ratio * len(lines):
                    break
                # data = json.loads(line)
                self.data.append(line)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        line = self.data[idx]
        data = json.loads(line)
        # data has instruction, input, output, history
        instruction_text = data['instruction']
        input_text = data['input']
        output_text = data['output']
        history = data['history']
        query = instruction_text + input_text
        answer = output_text + self.tokenizer.eos_token

        messages = []
        if history:
            for i in history:
                messages.append({'role': 'user', 'content': i[0]})
                messages.append({'role': 'assistant', 'content': i[1]})
        messages.append({'role': 'user', 'content': query})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)   
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        answer_input_ids = self.tokenizer(answer, return_tensors="pt")["input_ids"]
        # TODO check the + of list and tensor
        # TODO check the + of list and tensor
        labels = torch.cat((torch.zeros(1, prompt_input_ids.shape[1]), answer_input_ids), dim=1)
        input_ids = torch.cat((prompt_input_ids, answer_input_ids), dim=1)
        text_len = input_ids.shape[-1]
        if text_len >= self.max_seq_length:
            input_ids = input_ids[:,:self.max_seq_length]
            labels = labels[:,:self.max_seq_length]
        else:
            input_ids = F.pad(input_ids, (0, self.max_seq_length - text_len), value=0)
            labels = F.pad(labels, (0, self.max_seq_length - text_len), value=0)
        
        input_ids = input_ids.squeeze(0)
        labels = labels.squeeze(0).to(torch.long)
        # we do not need the shift of the labels and input_ids
        # input_ids = torch.tensor(input_ids[:-1])
        # labels = torch.tensor(labels[1:])
        
        return {"input_ids": input_ids, "labels": labels, "position_ids": torch.tensor(0, dtype=torch.long), "position_ids": torch.tensor(0, dtype=torch.long)}
        
def test_datasets(datasets):
    print(len(datasets))
    print(next(iter(datasets)))
    
# def test_datasets_path(data_path):
#     key_word_path = "datasets/llm/sft/deepctrl-sft-data/type_keywords_zh.json"
#     with open(key_word_path, "r", encoding="utf-8") as f:
#         type_keywords_zh = json.load(f)
#     # print(type_keywords_zh)
#     data_list = []
#     with open(data_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()
#         for i, line in enumerate(lines):
#             if i >= 10:
#                 break
#             data = json.loads(line)
#             data_list.append(data)
#     for i in range(10):
#         print("input", data_list[i]["input"])
#         print("output", data_list[i]["output"])
#         print("instruction", data_list[i]["instruction"])
#         print("history", "".join(data_list[i]["instruction"]))
    



def length_statistic(datasets):
    import matplotlib.pyplot as plt
    length_categories = ["0-100", "100-500", "500-1024", "1024-2000", ">2000"]
    length_counts = {category: 0 for category in length_categories}

    for i in range(len(datasets)):
        current_len = datasets[i]
        length = current_len
        if length <= 100:
            length_category = "0-100"
        elif length <= 500:
            length_category = "100-500"
        elif length <= 1024:
            length_category = "500-1024"
        elif length <= 2000:
            length_category = "1024-2000"
        else:
            length_category = ">2000"
        length_counts[length_category] += 1
    # print(length_counts)
    for category, count in length_counts.items():
        print(f"{category}: {count}")
    # plt.bar(length_counts.keys(), length_counts.values())
    # plt.xlabel('Length Category')
    # plt.ylabel('Number of Samples')
    # plt.title('Sample Length Distribution')
    # plt.savefig("length_sft.jpg")

        

if __name__ == "__main__":
    # test_datasets_path("datasets/llm/sft/deepctrl-sft-data/sft_data_zh.jsonl")
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

    tunable_param, total_param = count_parameters(model)
    rank0_print(f"Total parameters: {total_param}, Tunable parameters: {tunable_param}")

    train_datasets = SFTDatasets(data_args.data_path, tokenizer, data_args.max_seq_length)
    # train_datasets = SFTLengthDatasets(data_args.data_path, tokenizer, data_args.max_seq_length)
    # length_statistic(train_datasets)
    collator = DefaultDataCollator()

    # test_datasets(train_datasets)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_datasets
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()