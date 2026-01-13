# Multimodal Vision-Language Assistant for Autonomous Driving Scenarios
# 基于自动驾驶场景的多模态视觉语言助手

## 项目结构说明

本项目的目录结构及文件功能说明如下：

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