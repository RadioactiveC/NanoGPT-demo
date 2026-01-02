# NanoGPT-demo: A Lightweight Chinese GPT Implementation

这是一个基于 PyTorch 从零实现的轻量级 GPT 模型（Decoder-only Transformer）。本项目参考了 Karpathy 的 nanoGPT，并针对中文语料（MiniMind 数据集）进行了适配。

**核心特点：**
- **代码极简**：核心代码模块化，拆分为 Model, Data, Train, Sample 四个独立脚本。
- **中文适配**：使用 GPT-2 Tokenizer，针对中文语料进行了特殊 Token (`<|endoftext|>`) 的处理，修复了常见的编码错误。
- **断点续训**：支持加载 Checkpoint，自动恢复模型权重、优化器状态和学习率调度器，随时中断和恢复训练。
- **开箱即用**：自动从 HuggingFace 下载预训练数据，无需手动准备。

## 📂 项目结构

```text
.
├── model.py        # GPT 模型架构定义 (SingleHead, MultiHead, FeedForward, Block)
├── dataset.py      # 数据处理脚本 (基于 tiktoken 和 datasets 库，含特殊字符清洗)
├── train.py        # 训练主脚本 (包含断点续训逻辑、Cosine 学习率调度)
├── sample.py       # 推理脚本 (加载训练好的权重生成文本)
├── requirements.txt # 依赖库列表
└── README.md       # 项目说明

```

## 🛠️ 快速开始 (Quick Start)

### 1. 安装依赖

请确保你的 Python 版本 >= 3.8，并安装项目所需的依赖库：

```bash
pip install -r requirements.txt
```
### 2. 开始训练

本项目不提供预训练权重，需要从头开始训练。运行以下命令即可自动下载 MiniMind 数据集并开始训练：

```bash
python train.py
```
**训练配置说明:**
- **设备:** 默认优先使用 GPU (cuda)，无 GPU 则使用 CPU。
- **输出:** 训练过程中的 Checkpoint 会保存在 checkpoints/ 目录下。
- **最终权重:** 训练结束后，模型会保存为 final_model.pth。
关于断点续训 (Resuming Training): 如果你断了训练，想从某个 epoch 继续，请修改 train.py 中的 resume_from 变量

### 3. 模型推理

训练完成后，确保当前目录下有 final_model.pth 文件，然后运行：

```bash
python sample.py
```
可以在 sample.py 中修改 input_str 变量来测试不同的输入提示词。


## ⚙️ 模型配置

默认配置如下 (可在 `model.py` 的 `GPTconfig` 类或 `train.py` 顶部修改)：

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| `n_layer` | 12 | Transformer 层数 |
| `n_head` | 12 | 注意力头数 |
| `n_embd` | 768 | 嵌入维度 (Hidden Size) |
| `block_size` | 512 | 上下文窗口大小 (Context Length) |
| `vocab_size` | 50257 | 基于 GPT-2 词表 (Byte-Level BPE) |
| `dropout` | 0.1 |防止过拟合的丢弃率 |

## 📚 数据集

本项目使用 [MiniMind Dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset) (`pretrain_hq.jsonl`) 进行训练。

- **数据处理**: 逻辑位于 `dataset.py`。
- **清洗规则**: 自动去除 `<|im_start|>` 等无关标签，并将 `<|im_end|>` 统一替换为 GPT 标准结束符 `<|endoftext|>`。
- **Tokenizer**: 使用 `tiktoken` 的 `gpt2` 编码器，并开启 `allowed_special` 选项以正确处理特殊 Token。
