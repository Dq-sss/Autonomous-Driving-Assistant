# Multimodal Vision-Language Assistant for Autonomous Driving Scenarios
# 基于自动驾驶场景的多模态视觉语言助手

本项目旨在利用多模态大模型 Qwen2.5-VL-7B-Instruct 构建一个自动驾驶场景助手。模
型通过对交通场景图像的分析理解，识别车辆、行人、信号灯等关键交通要素，并解释其
对驾驶行为的影响。当用户提出交通状况相关问题，模型可以实时给出交通状况分析以及
开车决策。本项目包含数据处理、LoRA 微调以及模型效果评估，并基于 Gradio 实现了可
视化问答交互系统。

## 任务描述

模型的主要任务是分析自车视角的交通图像，关注以下七类物体：
1. 车辆（轿车、卡车、公交车等）
2. 易受伤害的道路使用者（行人、骑行者）
3. 交通标志
4. 交通信号灯
5. 交通锥
6. 障碍物
7. 杂物

模型将描述这些物体的外观、位置、方向，并解释它们如何影响自车的驾驶，进而根据用户提出的问题给出驾驶决策。

## 目录结构

```
.
├── datasets/                  # 数据集处理脚本目录
│   ├── process_data.py        # 训练数据预处理脚本：读取 Parquet 数据，保存图片，转换为训练所需的 JSON 格式
│   └── test_data.py           # 测试数据预处理脚本：读取 Parquet 数据，转换为测试用的 JSON 格式
├── eval/                      # 模型评估相关目录
│   ├── eval.py                # 基于 LLM (glm-4.7) ，依据few_shot中的高、低质量样本示例对模型进行打分
│   └── few_shot/              # 存放用于少样本评估的示例文件 (high.json, low.json)
├── inference/                 # 推理与指标计算
│   ├── metric.py              # 评估指标计算 
│   └── predict/               # 存放不同模型的推理输出结果及参考答案
│       ├── output_adaqlora/       # AdaQLoRA 微调模型的预测结果
│       ├── output_qlora/          # QLoRA 微调模型的预测结果
│       └── Qwen2.5-VL-7B-Instruct/# 原始模型的预测结果
├── dataprocess.py             # 核心数据处理模块：定义 LazyVisionLanguageDataset 类，用于数据加载、多模态对齐和 prompt 构建
├── qwen_vl_7b.json            # 预处理完成的训练集数据文件
├── qwen_vl_7b_test.json       # 预处理完成的测试集数据文件
├── README.md                  # 项目说明文档
├── requirements.txt           # 项目 Python 依赖列表
├── train_adaqlora.py          # AdaQLoRA 微调训练
├── train_qlora.py             # QLoRA 微调训练主
└── ui.py                      # 基于 Gradio 的交互式 Web 界面
```

## 环境准备

建议使用 Python 3.9+ 和支持 CUDA 的 GPU 环境。

1. **安装依赖**

```bash
pip install -r requirements.txt
```


## 数据准备

本项目使用 Parquet 格式的数据集（CODA-LM）。

1. **下载模型权重 (Qwen2.5-VL-7B-Instruct)**

   ```bash
   # 配置 HF 镜像（确保下载速度） 
   export HF_ENDPOINT=https://hf-mirror.com 
   # 下载模型
   huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./Qwen2.5-VL-7B-Instruct --local-dir-use-symlinks False --resume-download # 断点续传，只补下损坏的文件
   ```

2. **运行数据处理脚本**

   该脚本会读取 Parquet 文件，将图片提取到指定路径下的 `converted_images/` 目录，并生成训练所需的 JSON 数据，保存为 qwen_finetune.json 。

   ```bash
   python process_data.py
   ```

   注意：请在 `process_data.py` 中修改 `parquet_path` 为你实际的数据路径。


## 模型微调 (QLoRA/AdaQLoRA Training)

使用 Qwen2.5-VL-7B-Instruct 模型进行 QLoRA/AdaQLoRA 微调。

1. **配置训练参数**

   打开 `train_qlora.py / train_adaqlora.py`，根据你的环境修改以下配置：
   - `MODEL_NAME`: 基础模型路径 (e.g., `/root/autodl-fs/Qwen2.5-VL-7B-Instruct`)
   - `DATA_PATH`: 训练数据 JSON 路径
   - `OUTPUT_DIR`: 模型保存路径

2. **开始训练**

   ```bash
   python train_qlora.py
   ```

   脚本会自动处理图像和文本的 Data Collation，并使用 PEFT 库进行高效微调。

## 推理与评估


使用 inference 文件夹下的 `metric.py` 加载微调后的模型权重进行推理。

```bash
python metric.py
```
需在脚本中指定 `BASE_MODEL_PATH`, `LORA_PATH` 和 `TEST_IMAGE_PATH`。

运行后模型预测结果保存在predict文件夹中，同时输出对应指标 Entity Recall、BERTScore、BLEURT 
和 ROUGE-L。


## 交互界面

用Gradio实现问答交互，用户上传当前从自车拍摄的图片，并提出交通决策相关问题，模型进行交通状况分析并回答驾驶决策

```bash
python ui.py
```


