import json
import torch
from PIL import Image
from dataclasses import dataclass
from typing import Any, List, Dict
from torch.utils.data import Dataset

# =====================================================
# Dataset（针对自动驾驶数据格式优化）
# =====================================================
class LazyVisionLanguageDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        # 系统提示词：强化自动驾驶专家角色
        self.system_prompt = (
            "你是一名专业的自动驾驶决策与风险分析专家。\n"
            "请严格基于给定的交通场景图像和描述回答问题。\n"
            "约束：\n"
            "1. 只能基于描述或图像中明确提到的交通要素进行分析。\n"
            "2. 不得引入未出现的车辆、行人或信号设施。\n"
            "3. 所有决策必须有明确理由，且逻辑清晰。\n"
            "4. 语言专业、客观、简洁，避免泛泛而谈。"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图片
        try:
            image = Image.open(item["image"]).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to load image {item.get('image')}, error: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # 构造消息列表
        # 第一条始终是 System Prompt
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            }
        ]
        
        # 解析对话内容
        for i, turn in enumerate(item["conversations"]):
            role = "user" if turn["from"] == "human" else "assistant"
            content = []
            
            # 第一轮 User 消息中注入图片占位符
            if role == "user" and i == 0:
                content.append({"type": "image"})
                # 处理数据中的 <img> 标签，避免重复
                text_value = turn["value"].replace("<img>\n", "").replace("<img>", "")
                content.append({"type": "text", "text": text_value})
            else:
                content.append({"type": "text", "text": turn["value"]})
            
            messages.append({
                "role": role,
                "content": content
            })

        return {"messages": messages, "image": image}

# =====================================================
# DataCollator（针对 Qwen2.5-VL 动态分辨率优化）
# =====================================================
@dataclass
class DataCollatorForQwenVL:
    processor: Any
    max_length: int = 1536

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []
        for sample in batch:
            # 使用 apply_chat_template 渲染为模型输入文本
            # add_generation_prompt=False 因为训练时已经包含回答
            prompt = self.processor.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(prompt)
            images.append(sample["image"])

        # 处理多模态数据
        # Qwen2.5-VL 会自动处理图像补丁（Patches）
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        
        # 默认屏蔽所有 token 的 loss
        labels[:] = -100

        # 获取辅助定位 Token ID
        tokenizer = self.processor.tokenizer
        # Qwen 格式: <|im_start|>assistant\n
        assistant_header_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        header_len = len(assistant_header_ids)

        for i in range(input_ids.size(0)):
            cur_ids = input_ids[i].tolist()
            seq_len = len(cur_ids)
            
            idx = 0
            while idx < seq_len:
                # 寻找 assistant 响应的起始点
                if cur_ids[idx : idx + header_len] == assistant_header_ids:
                    start_pos = idx + header_len
                    
                    # 寻找对应的结束点 <|im_end|>
                    try:
                        end_pos = cur_ids.index(im_end_id, start_pos)
                        # 计算 loss 的范围包含文本内容和结束符号
                        labels[i, start_pos : end_pos + 1] = input_ids[i, start_pos : end_pos + 1]
                        idx = end_pos + 1
                    except ValueError:
                        # 若未找到结束符（序列截断情况），训练到末尾
                        labels[i, start_pos:] = input_ids[i, start_pos:]
                        break
                else:
                    idx += 1

        # 返回模型需要的字典
        return {
            "input_ids": inputs["input_ids"],
            "labels": labels,
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
        }

def get_dataset_and_collator(json_path: str, processor: Any, max_length: int = 1536):
    """
    初始化数据集和适配 Qwen2.5-VL 的 Collator
    """
    dataset = LazyVisionLanguageDataset(json_path)
    collator = DataCollatorForQwenVL(processor=processor, max_length=max_length)
    return dataset, collator