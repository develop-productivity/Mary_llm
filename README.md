
# Introduction

个人软件和硬件环境配置
```
GPU: RTX3090x4
CPU: Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz
python: 3.10
pytorch: torch2.1.2+cu121
Ubuntu == 20.04
CUDA == 12.2
```
## 模型信息

| param | hidden_size | num_hidden_layer | num_attention_head | num_kv_heads | intermediate_size | max_seq_len | vocabulary  size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 66.3m | 512 | 8 | 8 | 4 | 2048 | 2048 | 6400 |
|  |  |  |  |  |  |  |  |

## 训练细节
| stage | datasets size | times | platform |
| --- | --- | --- | --- |
| pretrain | 1.3M | 4.5h | 8x3090 |
| sft | 1.4M | 6.3h | 8x3090 |
| dpo |  |  | 8x3090 |

# QuickStart
## Environment

You should clone this project and create a python env.
```
git clone https://github.com/develop-productivity/Mary_llm.git
cd Marry_mllm
conda create -n env_name python=3.10
pip install -r requirments.txt
```
## Test
```
python gradio_vlm.py
```
## Train
### prepare the datasets
You can download the datasets follow the instructions and hyperlink.
Datasets:
* tokenizer datasets: [minimind](https://huggingface.co/datasets/jingyaogong/minimind_dataset) tokenizer数据
* pre-train datasets: [seq_monkey](http://share.mobvoi.com:5000/sharing/O91blwPkY) 10%的数据

* sft datasets: [匠数大模型](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data) 10%的数据. 

* dpo datasets: [DPO](https://huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese)

You should manguage you the file structure as follows
```
├── datasets
│   └── llm
│     ├── dpo
│     ├── pretrain
│     ├── tokenizer_datasets
│     └── dft
├── scripts
│   ├── dst_pre_train.sh
│   └── dst_sft.sh
├── sft.py
├── dpo.py
├── gradio_llm.py
├── model.py
├── pre_train.py
├── README.md
├── train_tokenizer.py
└── utils.py
```

# TODO

- [ ] Support LoRA finetine
- [ ] Suport kbit training
- [ ]  Support deepspeed config



