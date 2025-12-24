import json
import os
import torch
from PIL import Image
from dataclasses import dataclass
from typing import Dict, Sequence, Any, List
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType

# =========================
# 1. 基本配置
# =========================
MODEL_NAME = "/root/autodl-fs/Qwen2.5-VL-7B-Instruct"
DATA_PATH = "/root/autodl-fs/coda-lm-llava-format/qwen_finetune.json"
OUTPUT_DIR = "output"
MAX_LENGTH = 1536  # 适当增加长度以容纳图像Token

# =========================
# 2. 自定义 DataCollator (关键修复)
# =========================
# Qwen2-VL 需要特殊的 Collator 来处理 flattened 的 pixel_values 和 image_grid_thw
@dataclass
class DataCollatorForQwenVL:
    processor: Any

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 1. 提取 input_ids 和 labels
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        
        # 2. Padding (使用 tokenizer 的 pad_token_id)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        # 3. 生成 attention_mask
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        
        # 4. 处理图像特征 (Qwen2-VL 特有)
        # pixel_values 是 flatten 的 1D tensor，需要直接 cat，而不是 stack
        if "pixel_values" in instances[0]:
            batch["pixel_values"] = torch.cat([x["pixel_values"] for x in instances], dim=0)
            
        # 处理 image_grid_thw (图像的 grid 形状信息)
        if "image_grid_thw" in instances[0]:
            batch["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in instances], dim=0)
            
        return batch

# =========================
# 3. 数据集定义 (关键修复)
# =========================
class VisionLanguageDataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 兼容不同的 json 结构
            self.data = data["instances"] if "instances" in data else data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]
        
        # --- 步骤 A: 提取图像 ---
        images = []
        for msg in messages:
            if isinstance(msg["content"], list):
                for content in msg["content"]:
                    if content["type"] == "image":
                        image_path = content["image"]
                        # 简单的容错处理
                        if os.path.exists(image_path):
                            images.append(Image.open(image_path).convert("RGB"))
                        else:
                            print(f"Warning: Image not found {image_path}, using black image.")
                            images.append(Image.new("RGB", (224, 224), (0, 0, 0)))

        # --- 步骤 B: 使用 apply_chat_template (修复核心) ---
        # 这会自动将 {"type": "image"} 转换为模型所需的 <|vision_start|>...<|vision_end|>
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # --- 步骤 C: Processor 处理 ---
        # Qwen2.5-VL processor 会自动处理文本中的 vision token 和图像特征的对齐
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        )
        
        # 去除 batch 维度 (Dataset 返回单个样本)
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # --- 步骤 D: Labels ---
        # 简单起见，这里训练所有 token (labels = input_ids)
        # 如果需要更精细的 mask (只训练 assistant)，需要解析 input_ids
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs

# =========================
# 4. 主程序
# =========================
def train():
    print("正在加载 Processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    print("正在加载模型...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16, # 推荐使用 bf16
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 数据集与 Collator
    train_dataset = VisionLanguageDataset(DATA_PATH, processor)
    data_collator = DataCollatorForQwenVL(processor)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True, # 开启 bf16 (如果显卡支持)
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False, # 【重要】必须为 False，否则 image_grid_thw 会被删除导致报错
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    print("开始训练...")
    trainer.train()
    
    # 保存
    print(f"保存模型至 {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("✅ Qwen2.5-VL LoRA 微调完成")


train()