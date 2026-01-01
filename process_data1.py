import pandas as pd
import json
import os
from pathlib import Path

# --- 1. 读取 Parquet 文件 ---
parquet_path = './Chinese/Mini-00000-of-00001.parquet'
df = pd.read_parquet(parquet_path)
print(f"数据总行数: {len(df)}")

# --- 2. 创建保存图片的目录（使用绝对路径） ---
# 获取当前脚本所在目录的绝对路径
base_dir = Path(__file__).parent.absolute()
image_output_dir = base_dir / "converted_images"
os.makedirs(image_output_dir, exist_ok=True)
print(f"图片将保存至: {image_output_dir}")

# --- 3. 转换每一行数据 ---
all_instances = []

for idx, row in df.iterrows():
    # 处理ID和图片
    sample_id = row['id']
    image_dict = row['image']
    
    # 关键：从字典中提取字节数据
    image_bytes = image_dict['bytes']  # 这是一个 bytes 对象
    
    # 定义图片保存路径（使用绝对路径）
    image_filename = f"{sample_id}.png"
    # 使用绝对路径
    image_path = os.path.abspath(image_output_dir / image_filename)
    
    # 将字节保存为PNG文件
    with open(image_path, 'wb') as f:
        f.write(image_bytes)
    
    # --- 处理对话，转换为 Qwen 格式 ---
    original_conv = row['conversations']  # 这是一个列表，如 [{'from': 'human', 'value': ...}]
    
    # 初始化 Qwen 格式的 messages 列表
    messages = []
    
    for turn in original_conv:
        role = "user" if turn['from'] == 'human' else "assistant"
        content_value = turn['value']
        
        # 如果内容以 '<image>' 开头，需要将其拆分为图片和文本部分
        if content_value.strip().startswith('<image>'):
            parts = content_value.split('\n', 1)  # 按第一个换行符分割
            # 第一部分是 '<image>'，我们用图片路径替换它
            # 使用绝对路径
            content_list = [{"type": "image", "image": image_path}]
            # 如果后面还有文本，则添加文本部分
            if len(parts) > 1 and parts[1].strip():
                content_list.append({"type": "text", "text": parts[1].strip()})
        else:
            # 如果没有图片，则全部是文本
            content_list = [{"type": "text", "text": content_value.strip()}]
        
        # 将这一轮对话添加到 messages
        messages.append({"role": role, "content": content_list})
    
    # --- 构造最终符合 Qwen 微调格式的样本 ---
    # 使用绝对路径
    formatted_sample = {
        "id": sample_id,
        "images": [image_path],  # 使用绝对路径
        "messages": messages
    }
    
    all_instances.append(formatted_sample)
    
    # 每处理50条打印一次进度
    if (idx + 1) % 50 == 0:
        print(f"已处理 {idx + 1}/{len(df)} 条数据...")

# --- 4. 构建符合 Qwen 微调标准的完整数据结构 ---
# Qwen 多模态微调的标准格式
qwen_finetune_data = {
    "type": "vision_language_conversation",  # 指定数据类型
    "instances": all_instances  # 所有数据实例
}

# --- 5. 保存为可直接用于微调的 JSON 文件 ---
output_json_path = base_dir / "qwen_finetune_ready.json"
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(qwen_finetune_data, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print("转换完成！")
print(f"{'='*60}")
print(f"数据集类型: {qwen_finetune_data['type']}")
print(f"数据实例总数: {len(all_instances)}")
print(f"JSON 文件已保存至: {output_json_path}")
print(f"图片已保存至目录: {image_output_dir}")
print(f"图片总数: {len(os.listdir(image_output_dir))}")
print(f"{'='*60}")

# --- 6. 验证生成的文件格式 ---
print("\n数据格式验证:")
print(f"1. 顶层包含 'type' 字段: {'type' in qwen_finetune_data}")
print(f"2. 顶层包含 'instances' 字段: {'instances' in qwen_finetune_data}")
print(f"3. 每个实例包含 'id', 'images', 'messages' 字段: ", end="")
if all_instances:
    sample = all_instances[0]
    print(f"{'id' in sample and 'images' in sample and 'messages' in sample}")
print(f"4. 图片路径是否为绝对路径: {os.path.isabs(all_instances[0]['images'][0])}")
print(f"{'='*60}")

# --- 7. 生成数据统计信息 ---
total_messages = sum(len(inst['messages']) for inst in all_instances)
total_images = sum(len(inst['images']) for inst in all_instances)

print(f"\n数据统计:")
print(f"- 总样本数: {len(all_instances)}")
print(f"- 总对话轮数: {total_messages}")
print(f"- 总图片数: {total_images}")
print(f"- 平均每样本对话轮数: {total_messages/len(all_instances):.2f}")
print(f"- 平均每样本图片数: {total_images/len(all_instances):.2f}")

# 检查是否有助理回复
has_assistant = any(
    any(msg['role'] == 'assistant' for msg in inst['messages'])
    for inst in all_instances
)
print(f"- 是否包含助理回复: {has_assistant}")
print(f"{'='*60}")

print(f"\n✅ 数据已准备好用于 Qwen 多模态模型微调！")
print(f"您可以直接使用此文件: {output_json_path}")
