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

from trl import DPOTrainer


trans_chinese = datasets.load_dataset("datasets/llm/dpo/beyond/rlhf-reward-single-round-trans_chinese", split="train[:1%]")


# filter chosen length < 18
trans_chinese = trans_chinese.map(lambda r: {"messages": len(r["chosen"])}).filter(lambda r: r["messages"]<18)

tokenizer = AutoTokenizer.from_pretrained("tokenizer")



# trans_chinese = trans_chinese.map(llm_format, batched=False)
# trans_chinese = trans_chinese.set_format(type="torch", columns=["chosen", "rejected"], format=llm_format)

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
    trans_chinese = datasets.load_dataset(data_path, split="train[:1%]")
    # filter chosen length < 18
    trans_chinese = trans_chinese.map(lambda r: {"messages": len(r["chosen"])}).filter(lambda r: r["messages"]<18)
    trans_chinese = trans_chinese.map(llm_format, batched=False)


def test_datasets(datasets):
  for i in range(10):
    print(datasets[i])


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="save/pretrain")
    tokenizer_path: Optional[str] = field(default="tokenizer")
    max_seq_length: int = field(default=2048)


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default="datasets/llm/sft/deepctrl-sft-data/type_keywords_zh.json")
    dataset_max_seq_length: int = field(default=2048)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="save/pretrain")
    do_train: bool = field(default=True)
    per_device_train_batch_size: int = field(default=64)
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



if __name__ == "__main__":
    test_datasets(trans_chinese)
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

    ref_model = AutoModelForCausalLM.from_pretrained("save/pretrain")
    # train_datasets = SFTDatasets(data_args.data_path, tokenizer, model_args.max_seq_length)
    collator = DefaultDataCollator()

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_datasets
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()