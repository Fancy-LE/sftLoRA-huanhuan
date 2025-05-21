from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
from modelscope import snapshot_download, AutoModel, AutoTokenizer

# huanhuan版lora微调

def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == "__main__":

    torch.cuda.empty_cache()
    # 一、模型下载

    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 选择合适的模型版本
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_path = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', local_dir='./dir', revision='master')
    # model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    # model_path = './dir'
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)

    # 二、数据集处理
    tokenizer = AutoTokenizer.from_pretrained(model_path, usmoxe_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 将JSON文件转换为CSV文件
    df = pd.read_json('huanhuan.json')
    ds = Dataset.from_pandas(df)
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    # 三、定义LoraConfig
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32,
        lora_dropout=0.05# Dropout 比例
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters() # 打印总训练参数

    model.enable_input_require_grads()
    # 四、自定义TrainingArguments参数
    args = TrainingArguments(
        output_dir="./output/deepseek1.5_lora",
        # 如果你的显存比较小，那可以把batch_size设置小一点，梯度累加增大一些。
        per_device_train_batch_size=2,  #batch_size
        gradient_accumulation_steps=10,  #steps 梯度累加
        logging_steps=10,   #多少步输出一次log
        num_train_epochs=3, #epoch
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )

    # 五、使用Trainer训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train() # 开始训练
    # 在训练参数中设置了自动保存策略此处并不需要手动保存。

    # 保存模型
    model.save_pretrained("./output/finetuned_model")
    tokenizer.save_pretrained("./output/finetuned_model")

