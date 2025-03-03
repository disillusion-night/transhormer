from model import Transformer114514
import torch

def predict_next_token(model, input_tokens):
    model.eval()  # 进入推理模式

    input_tensor = torch.tensor([input_tokens]) # 转换为张量
    with torch.no_grad():
        output = model(input_tensor)  # 预测 logits
        predicted_token = torch.argmax(output, dim=1).item()  # 取最大概率的 token
    
    return predicted_token

# 加载训练好的模型
vocab_size = 8 
model = Transformer114514(vocab_size, d_model=16)  # 初始化相同结构的模型
model.load_state_dict(torch.load("llm_model.pth"))  # 加载训练好的权重


test_input = [1, 7, 0, 3]  # 李田所在这里大喊大叫
predicted = predict_next_token(model, test_input)

print(f"预测的下一个 token: {predicted}")