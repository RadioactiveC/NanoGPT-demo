import os
import torch
from torch.utils.data import DataLoader, random_split
from model import GPT, GPTconfig
from dataset import MyDataset

# 配置
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "checkpoints"
resume_from = "gpt_epoch.pth"  # 想恢复的权重路径，填None则从头训练
# resume_from = None
total_epochs = 4  # 训练总轮次
start_epoch = 0  #断点续训起点

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# 准备数据
torch.manual_seed(1337)
full_dataset = MyDataset(max_lines=10000)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=12, shuffle=True)
val_loader = DataLoader(val_data, batch_size=12, shuffle=False)

# 初始化模型与优化器
config = GPTconfig()
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs * len(train_loader))

# 断点继续训，填一下续训的轮次
if resume_from and os.path.exists(resume_from):
    print(f"Loading checkpoint from {resume_from}")
    checkpoint = torch.load(resume_from, map_location=device)

    # 检查文件格式并加载
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 一个完整的 Checkpoint (包含 epoch, optimizer 等)
        print("Detected full checkpoint.")
        model.load_state_dict(checkpoint['model_state_dict'])

        # 尝试恢复优化器和调度器
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Failed to load optimizer state: {e}")

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1

    else:
        # 一个纯权重文件
        print("state_dict only")
        model.load_state_dict(checkpoint)
        # 纯权重文件没有优化器状态，只能从头开始训练，但起点是预训练好的模型
        print("Loaded weights only. Optimizer states are reset.")
        start_epoch = 0

    print(f"Resumed training from epoch {start_epoch}")


# 训练
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            total_loss += loss.item()
    return total_loss / len(loader)


# 主循环
if __name__ == "__main__":
    for epoch in range(start_epoch, total_epochs):  # 若要恢复训练，从恢复训练轮次开始计数
        print(f"--- Epoch {epoch} ---")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss = evaluate(model, val_loader)

        print(f"Epoch {epoch} Done. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存 checkpoint (包含所有状态)
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": train_loss
        }
        torch.save(checkpoint_dict, os.path.join(checkpoint_dir, f"gpt_epoch_{epoch}.pth"))

    # 训练结束后保存最终参数
    torch.save(model.state_dict(), "final_model.pth")
    print("Training Complete!")