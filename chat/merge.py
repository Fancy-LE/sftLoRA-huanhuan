from peft import PeftModel

from transformers import AutoModelForCausalLM

# 载入基础模型
model = AutoModelForCausalLM.from_pretrained('./dir', load_in_8bit=False, device_map="auto")

# 载入微调后的模型文件
model = PeftModel.from_pretrained(model, './output/finetuned_model', device_map="auto", trust_remote_code=True)

# 合并模型
merged_model = model.merge_and_unload()

# 保存模型
merged_model.save_pretrained("./output/output_model")

