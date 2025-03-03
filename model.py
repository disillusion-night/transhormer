import torch.nn as nn

class Transformer114514(nn.Module):

    def __init__(self, vocab_size, d_model=16, num_heads=2, num_layers=2):
        super(Transformer114514, self).__init__()
        
        # 纬度转换
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer 模型
        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers,batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(d_model, vocab_size)
    # 向前传播
    def forward(self, x):
        x = self.embedding(x) 
        x = self.transformer(x, x)
        x = self.fc(x[:, -1, :])
        return x