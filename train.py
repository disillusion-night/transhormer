import torch
import torch.nn as nn
import torch.optim as optim
import json
import os as os
from torch.utils.data import DataLoader, Dataset
from model import Transformer114514


# 读取 JSON 数据
class TokenDataset(Dataset):
    def __init__(self, json_file):
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"文件 {json_file} 不存在，请检查路径。")
        
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not data:
                    raise ValueError(f"文件 {json_file} 内容为空，请提供有效的 JSON 数据。")
            except json.JSONDecodeError as e:
                raise ValueError(f"文件 {json_file} 格式错误：{e}")
        
        self.inputs = [torch.tensor(item["input"]) for item in data]
        self.targets = [torch.tensor(item["target"]) for item in data]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# 训练模型
def train_model():
    dataset = TokenDataset("train.json")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    vocab_size = 11
    model = Transformer114514(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(800):  # 训练 50 轮
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "llm_model.pth")

# 运行训练
train_model()