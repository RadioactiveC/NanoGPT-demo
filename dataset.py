import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class MyDataset(Dataset):
    def __init__(self, split="train", block_size=512, max_lines=10000):
        # 基于huggingface的dataset加载数据 (这里用minimizing_dataset)
        dataset = load_dataset("jingyaogong/minimind_dataset", data_files="pretrain_hq.jsonl")

        # 加载 HuggingFace Tokenizer (用 Qwen/Qwen-1.5-7B-Chat 的 tokenizer)
        print("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", trust_remote_code=True)

        # Qwen的结束符通常是<|endoftext|>或者<|im_end|>，获取它的 ID
        # 如果 tokenizer 没有 pad_token，通常手动指定为 eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        # 文本清洗
        def process_text(batch):
            text_list = batch['text']
            # 清理特殊符号
            clean_text = [
                t.replace('<|im_start|>', '').replace('<|im_end|>', '')
                for t in text_list
            ]
            return {'text': clean_text}

        data = dataset[split].map(process_text, batched=True)

        # 分块处理
        raw_data = data[:max_lines]['text']  # 取部分数据
        full_encoded = []
        for text in raw_data:
            # Qwen tokenizer 能够很好地处理中文，不需要 allowed_special 这种复杂操作
            encoded_text = self.tokenizer.encode(text, add_special_tokens=False)
            full_encoded.extend(encoded_text + [self.eos_token_id])

        self.encoded_data = []
        # 分割成 block_size + 1 (因为要有输入和目标)
        for i in range(0, len(full_encoded), block_size + 1):
            chunk = full_encoded[i: i + block_size + 1]
            if len(chunk) < block_size + 1:
                chunk = chunk + [self.eos_token_id] * (block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

            # 把这个属性暴露出来给 train.py
            self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# 测试
if __name__ == "__main__":
    ds = MyDataset(max_lines=100)
    print(f"Dataset size: {len(ds)}")
    x, y = ds[0]
    print(f"Sample x shape: {x.shape}, y shape: {y.shape}")