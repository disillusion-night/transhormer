import random
import json

# 读取JSON文件
with open('./token.json', 'r') as file:
    data = json.load(file)

# 获取所有的key
keys = list(data.keys())

for i in range(100):
    # 随机抽取4个key，允许重复
    random_keys = [random.choice(keys) for _ in range(4)]
    output = ""
    # 输出随机抽取的key
    for key in random_keys:
        output += data[key]
    print(output)