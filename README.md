# WaFiGPT

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Miwafi/WafiGPT?style=social)](https://github.com/Miwafi/WafiGPT)

WaFiGPT 是一个基于 PyTorch 实现的轻量级大语言模型（LLM）项目，采用 Transformer 架构，支持自定义训练与对话推理。

## 项目概述

本项目从零实现了完整的语言模型训练与推理流程，包含以下核心特性：

- **Transformer 架构**：基于自注意力机制（Self-Attention）的解码器模型
- **分组查询注意力（GQA）**：支持多头注意力与键值头共享，降低推理内存占用
- **旋转位置编码（RoPE）**：实现相对位置编码，提升长序列建模能力
- **RMS 层归一化**：替代传统 LayerNorm，提升训练稳定性
- **SwiGLU 激活函数**：采用门控线性单元，增强模型表达能力
- **8-bit 量化优化器**：集成 bitsandbytes 实现显存高效训练

## 项目结构

```
WafiGPT/
├── train.py          # 模型训练脚本
├── chat.py           # 对话推理脚本
├── data/             # 训练数据目录
├── model/            # 模型检查点存储目录
├── .gitignore        # Git 忽略配置
└── README.md         # 项目文档
```

## 环境依赖

- Python 3.11+
- PyTorch 2.0+
- CUDA Toolkit（推荐，用于 GPU 加速）

### 安装依赖

```bash
pip install -r requirements.txt
```

## 模型架构

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 1024 | 隐藏层维度 |
| `ffn_hidden_size` | 4096 | 前馈网络中间层维度 |
| `block_count` | 24 | Transformer 层数 |
| `num_heads` | 16 | 注意力头数量 |
| `num_kv_heads` | 1 | 键值头数量（GQA） |
| `rope_dim` | 64 | 旋转位置编码维度 |
| `vocab_size` | 32000 | 词表大小 |
| `max_seq_length` | 512 | 最大序列长度 |

### 特殊 Token

| Token | ID | 用途 |
|-------|-----|------|
| `<|padding|>` | 0 | 填充标记 |
| `<|unknown|>` | 1 | 未知词标记 |
| `<|system|>` | 2 | 系统提示 |
| `<|user|>` | 3 | 用户输入标记 |
| `<|assistant|>` | 5 | 助手回复标记 |
| `<|think|>` | 4 | 思考过程开始 |
| `<|/think|>` | 11 | 思考过程结束 |
| `<|end|>` | 7 | 序列结束标记 |

## 数据格式

### 文本格式（.txt）

每行一条训练样本：

```
<|user|>你好<|assistant|>你好！有什么可以帮助你的吗？<|end|>
```

### JSON Lines 格式（.jsonl）

```json
{"instruction": "解释量子计算", "input": "", "output": "量子计算是一种利用量子力学原理进行信息处理的计算范式..."}
```

或

```json
{"text": "<|user|>你好<|assistant|>你好！<|end|>"}
```

## 使用指南

### 训练模型

1. 准备训练数据，放置于 `./data/` 目录
2. 运行训练脚本：

```bash
python train.py
```

训练过程中会自动保存模型检查点至 `./model/` 目录，包含：
- `model.safetensors`：模型权重（SafeTensors 格式）
- `config.json`：模型配置
- `tokenizer.json`：分词器词表

### 对话推理

```bash
python chat.py
```

启动后输入提示词即可与模型交互，支持以下命令：
- `exit` 或 `quit`：退出程序
- `Ctrl+C`：中断当前生成

### 推理参数

在 `generate_response` 函数中可调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_length` | 512 | 最大生成长度 |
| `temperature` | 0.3 | 采样温度，控制随机性 |
| `repetition_penalty` | 1.0 | 重复惩罚系数 |
| `presence_penalty` | -1.5 | 存在惩罚系数 |

## 训练流程

训练采用分阶段（Stage-based）设计，支持多数据集连续训练：

```python
stages = [
    {"stage_name": "Pre-training", "file_path": "data/pretrain.txt", "epochs": 10},
    {"stage_name": "Fine-tuning", "file_path": "data/finetune.jsonl", "epochs": 5},
]
```

每个阶段独立进行训练/验证集划分，自动保存检查点。

## 分词器

采用基于规则的分词策略：

1. **特殊 Token**：优先匹配预定义的特殊标记
2. **英文单词**：连续字母序列作为整体
3. **数字**：单个数字独立成词
4. **空格**：保留空格信息
5. **其他字符**：按单字符切分

分词器支持动态词表扩展，训练过程中自动构建词表。

## 许可证

本项目采用 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件。

## 致谢

- [PyTorch](https://pytorch.org/)：深度学习框架
- [Hugging Face](https://huggingface.co/)：SafeTensors 格式支持
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)：8-bit 优化器实现
