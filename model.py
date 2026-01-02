import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math


@dataclass
class GPTconfig:
    block_size: int = 512  # 文本的最大长度
    vocab_size: int = 50257  # 词表大小
    n_layer: int = 12  # 层数
    n_head: int = 12  # 头数
    n_embd: int = 768  # 嵌入维度
    dropout: float = 0.1

    # 避免初始化陷阱
    @property
    def head_size(self):
        return self.n_embd // self.n_head


class SingleHeadAttention(nn.Module):
    def __init__(self, config: GPTconfig):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size)
        self.query = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)
        self.head_size = config.head_size
        self.register_buffer(
            "attention_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        weight = query @ key.transpose(-2, -1)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float('-inf')
        ) / math.sqrt(self.head_size)

        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        return weight @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTconfig):
        super().__init__()
        self.heads = nn.ModuleList([
            SingleHeadAttention(config) for _ in range(config.n_head)
        ])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, config: GPTconfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTconfig):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.l1 = nn.LayerNorm(config.n_embd)
        self.l2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.att(self.l1(x))
        x = x + self.ff(self.l2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTconfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Tie weights
        self.token_embedding_table.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch, seq_len = idx.size()
        token_embed = self.token_embedding_table(idx)
        position_embed = self.position_embedding_table(torch.arange(seq_len, device=idx.device))

        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx