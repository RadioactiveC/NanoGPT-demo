import torch
from torch.utils.data import Dataset
import tiktoken
from datasets import load_dataset


class MyDataset(Dataset):
    def __init__(self, split="train", block_size=512, max_lines=10000):
        # 基于huggingface的dataset加载数据 (这里用minimizing_dataset)
        dataset = load_dataset("jingyaogong/minimind_dataset", data_files="pretrain_hq.jsonl")

        # 文本清洗
        def process_text(batch):
            text_list = batch['text']

            clean_text = [
                t.replace('<|im_start|>', '').replace('<|im_end|>', '<|endoftext|>')
                for t in text_list
            ]
            return {'text': clean_text}

        data = dataset[split].map(process_text, batched=True)

        # Tokenizer设置
        self.enc = tiktoken.get_encoding("gpt2")
        self.eos_token = self.enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        # 分块处理
        raw_data = data[:max_lines]['text']  # 取部分数据
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text, allowed_special={"<|endoftext|>"})
            full_encoded.extend(encoded_text + [self.eos_token])

        self.encoded_data = []

        for i in range(0, len(full_encoded), block_size + 1):
            chunk = full_encoded[i: i + block_size + 1]
            if len(chunk) < block_size + 1:
                # Padding
                chunk = chunk + [self.eos_token] * (block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

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