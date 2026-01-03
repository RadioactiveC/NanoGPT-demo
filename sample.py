import torch
from model import GPT, GPTconfig
from transformers import AutoTokenizer
# 环境
device = "cuda" if torch.cuda.is_available() else "cpu"
# tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", trust_remote_code=True)

# 加载模型
config = GPTconfig()
config.vocab_size = 152243 # 修改vocab_size
model = GPT(config).to(device)

# 加载权重,权重需要放置在同级目录中
model_path = "final_model.pth"
if torch.cuda.is_available():
    state_dict = torch.load(model_path)
else:
    state_dict = torch.load(model_path, map_location="cpu")

model.load_state_dict(state_dict)
model.eval()
print("Model loaded!")


# 生成函数 是否对于多个长短不一的无法生效
def generate_text(start_text, max_new_tokens=50):
    ids = tokenizer.encode(start_text, add_special_tokens=False)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)

    with torch.no_grad():
        generated_ids = model.generate(ids, max_new_tokens=max_new_tokens)

    # 解码
    return tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)


# 测试
if __name__ == "__main__":
    input_str = "我正在吃饭,"
    print(f"Input: {input_str}")
    output_str = generate_text(input_str)
    print(output_str)