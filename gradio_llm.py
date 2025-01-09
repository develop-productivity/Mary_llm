# TODO
# 写一个基于gradio的交互界面，包含模型选择（SFT, DPO, pretrain）输入文本框，输出文本框

from gradio import Interface
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model import MaryConfig, MaryCausalModel

from typing import List
from utils import Dialog


# 首先定义三个模型
AutoConfig.register("mary_llm", MaryConfig)
AutoModelForCausalLM.register(MaryConfig, MaryCausalModel)

pretrain_model = AutoModelForCausalLM.from_pretrained("save/pretrain")
sft_model = AutoModelForCausalLM.from_pretrained("save/sft")
dpo_model = AutoModelForCausalLM.from_pretrained("save/dpo_1_beyond_trans_chinese")
pretrain_model.eval()
sft_model.eval()
dpo_model.eval()

tokenizer = AutoTokenizer.from_pretrained("tokenizer")


def text_completion(model, tokenizer, input_text):
    input_data = tokenizer("s" + input_text, return_tensors="pt")["input_ids"]
    generation_tokens = model.generate({"input_ids": input_data}, max_gen_len=64, temperature=0.7, repetition_penalty=10)
    res = {"generation": "".join(tokenizer.decode(t) for t in generation_tokens)}
    return res["generation"]


# TODO implemente chat_completion
def chat_completion(
        model, 
        tokenizer, 
        input_text,
        dialogs: List[Dialog],
        ):
    pass



def model_predict(model_type, input_text):
    if model_type == "SFT":
        model = sft_model
    elif model_type == "DPO":
        model = dpo_model
    elif model_type == "pretrain":
        model = pretrain_model
    else:
        return "未知模型类型"
    
    return text_completion(model, tokenizer, input_text)

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("## 基于Gradio的模型选择和文本处理界面")
    
    model_type = gr.Radio(["pretrain", "SFT", "DPO"], label="选择模型")
    input_text = gr.Textbox(label="输入文本")
    output_text = gr.Textbox(label="输出文本")
    
    submit_button = gr.Button("提交")
    submit_button.click(fn=model_predict, inputs=[model_type, input_text], outputs=output_text)


# 启动Gradio应用
demo.launch()