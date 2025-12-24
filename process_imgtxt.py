import json
from pathlib import Path
from PIL import Image

def convert_to_qwen_format(input_json_path, output_json_path):
    """
    将原始 JSON 文件转换成 Qwen2.5-VL-7B-Instruct 训练格式：
    - 每条 user 消息前加 <image>
    - 保证每条实例顶层 images 与 messages 内 image 对应
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_instances = []

    for inst in data.get("instances", []):
        images = inst.get("images", [])
        messages = inst.get("messages", [])

        # 生成新的 messages 列表
        new_messages = []
        for msg in messages:
            new_content = []
            if msg["role"] == "user":
                # 遍历 content，把 image 转成 <image> token + 原 image
                for item in msg["content"]:
                    if item["type"] == "image":
                        # 用 <image> 占位
                        new_content.append({"type": "text", "text": "<image>"})
                    elif item["type"] == "text":
                        new_content.append(item)
            else:
                # assistant 保持原样
                new_content = msg["content"]

            new_messages.append({
                "role": msg["role"],
                "content": new_content
            })

        new_instances.append({
            "id": inst["id"],
            "images": images,
            "messages": new_messages
        })

    # 输出为标准格式
    out_data = {
        "type": "vision_language_conversation",
        "instances": new_instances
    }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成，输出文件：{output_json_path}")


# 使用示例
input_path = "qwen_finetune_ready.json"
output_path = "qwen_finetune.json"
convert_to_qwen_format(input_path, output_path)
