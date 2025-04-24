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
vocab_size = 11
model = Transformer114514(vocab_size, d_model=16)  # 初始化相同结构的模型
model.load_state_dict(torch.load("llm_model.pth"))  # 加载训练好的权重


# 允许用户输入 token 序列
user_input = input("请输入 token 序列（以空格分隔）: ")
user_tokens = list(map(int, user_input.split()))

# 使用用户输入进行推理
predicted_token = predict_next_token(model, user_tokens)

# 打印完整的句子
user_tokens.append(predicted_token)
print(f"完整的 token 序列: {user_tokens}")
import json
# 读取 token.json 文件
with open("token.json", "r", encoding="utf-8") as f:
    token_dict = json.load(f)
    sentence = ""
    for i in range(len(user_tokens)):
        sentence += token_dict[str(user_tokens[i])]
        
    print(f"句子: "+ sentence)

    
