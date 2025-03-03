import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, Dataset
from model import Transformer114514


# 读取 JSON 数据
class TokenDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
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

    vocab_size = 8
    model = Transformer114514(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(500):  # 训练 50 轮
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