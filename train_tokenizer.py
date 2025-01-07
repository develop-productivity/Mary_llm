# TODO:
# train the tokenizer
from transformers import AutoTokenizer
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


from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

os.environ["WANDB_MODE"] = "offline"


def read_data(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            yield data['text']

def train_tokenizer(tokenizer_dir="./tokenizer"):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    special_tokens = ['<pad>', '<unk>', '<s>', '</s>', '<mask>']
    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens = special_tokens,
        show_progress=True,
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
        
    )
    texts = read_data('datasets/llm/tokenizer_datasets/tokenizer_train.jsonl')
    tokenizer.train_from_iterator(texts, trainer)

    tokenizer.decoder = decoders.ByteLevel()
    # tokenizer_dir = "./tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)


def save_config(tokenizer_dir):
    # 保存配置文件
    config = {
            "add_bos_token": False,
            "add_eos_token": False,
            "add_prefix_space": True,
            "added_tokens_decoder": {
                "0": {
                    "content": "'<pad>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "1": {
                    "content": "'<unk>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "2": {
                    "content": "<s>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "3": {
                    "content": "</s>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "4": {
                    "content": "<mask>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                }
            },
            "additional_special_tokens": [],
            "bos_token": "<s>",
            "clean_up_tokenization_spaces": False,
            "eos_token": "</s>",
            "legacy": True,
            "model_max_length": 100000,
            "pad_token": "<pad>",
            "pad_token_id": 0,
            "sp_model_kwargs": {},
            "spaces_between_special_tokens": False,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "<unk>",
            "use_default_system_prompt": False,
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
        }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)


def test_tokenizer(tokenizer_dir="./tokenizer"):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    input_ids = tokenizer("Hello, world!")
    print("Hello, world!", input_ids)

    print("decode result:", tokenizer.decode(input_ids['input_ids']))
    print("encode <pad>:", tokenizer.decode([0]))
    print("encode <unk>:", tokenizer.decode([1]))
    print("encode <mask>:", tokenizer.decode([4]))

if __name__ == "__main__":
    # save_config("./tokenizer")
    test_tokenizer()
