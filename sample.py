import torch
import tiktoken
from model import GPT, GPTconfig

# 环境
device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

# 加载模型
config = GPTconfig()
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


# 生成函数
def generate_text(start_text, max_new_tokens=50):
    ids = enc.encode(start_text)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)

    with torch.no_grad():
        generated_ids = model.generate(ids, max_new_tokens=max_new_tokens)

    return enc.decode(generated_ids[0].tolist())


# 测试
if __name__ == "__main__":
    input_str = "我正在吃饭,"
    print(f"Input: {input_str}")
    output_str = generate_text(input_str)
    print(output_str)