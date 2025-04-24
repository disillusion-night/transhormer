# Trashormer
项目描述：这是一个基于 Transformer 模型的简单文da本bian练习项目，使用 PyTorch 实现。

项目实现了一个最简单的生成式语言模型
这个语言模型只能理解以下几个词汇

| Token | 词汇               |
|-------|--------------------|
| 0     | 这里               |
| 1     | 李田所             |
| 2     | 沼气动力研究院     |
| 3     | 大喊大叫           |
| 4     | 久等了             |
| 5     | 下北泽野兽府邸     |
| 6     | 黑色高级轿车       |
| 7     | 在                 |
| 8     | 鸭蛋                 |
| 9     | 牡蛎                 |
| 10     | 食用                 |
##
readme.md文件由GPT生成，所以部分文本可能并不适用
## 安装

1. 克隆仓库：
    ```bash
    git clone https://github.com/Kod-e/transhomer.git
    ```
2. 进入项目目录：
    ```bash
    cd your-repo
    ```
3. 安装依赖：
    ```bash
    pip install -r requirements.txt
    ```
4. 大喊やりますねぇ
## 使用

### 训练模型

1. 运行 `train.py` 进行模型训练：
    ```bash
    python train.py
    ```

### 推理

1. 运行 `inference.py` 进行推理：
    ```bash
    python inference.py
    ```

    
### 使用例

1. 运行 `inference.py` 进行推理：
    ```bash
    python inference.py
    ```
2. 输入tokens序列（数字）
    ``` bash
    请输入 token 序列（以空格分隔）: 1 7 6 10
    完整的 token 序列: [1, 7, 6, 10, 5]
    句子: 李田所在黑色高级轿车食用下北泽野兽府邸
    ```

## 文件说明

- `train.py`：用于训练 Transformer 模型。
- `model.py`：定义 Transformer 模型结构。
- `inference.py`：用于加载训练好的模型并进行推理。
- `token.json`：包含用于训练和推理的 token 数据。
- `test.py`：用于生成随机的 token 序列进行测试。

## 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解更多信息。

## 许可证

该项目使用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。
