import random
import json
from inference import predict_next_token, model  # 导入推理函数和模型

# 读取JSON文件
with open('./token.json', 'r', encoding="utf-8") as file:
    data = json.load(file)

# 获取所有的key
keys = list(data.keys())

for i in range(10):
    # 随机抽取4个key，允许重复
    random_keys = [random.choice(keys) for _ in range(4)]
    
    # 拼接生成句子
    generated_sentence = "".join(data[key] for key in random_keys)
    print(generated_sentence)

    # 调用模型预测下一个字符
    input_tokens = [int(key) for key in random_keys]  # 将随机 keys 转换为整数 token
    predicted_token = predict_next_token(model, input_tokens)
    print(f"预测的下一个 token: {predicted_token}")
    print(f"预测的下一个字符: {data[str(predicted_token)]}")

    # 输出完整的句子
    random_keys.append(str(predicted_token))  # 将预测的 token 添加到序列
    complete_sentence = "".join(data[key] for key in random_keys)
    print(f"完整的句子: {complete_sentence}")
