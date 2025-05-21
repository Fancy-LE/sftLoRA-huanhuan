from modelscope import snapshot_download

model_path = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', local_dir='./dir', revision='master')
