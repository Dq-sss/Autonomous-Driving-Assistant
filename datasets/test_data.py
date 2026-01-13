import pandas as pd
import json
from pathlib import Path
import os
import pyarrow.parquet as pq
import gc

# =====================================================
# 0. 指定要转换的 parquet 文件列表
# =====================================================
parquet_files = [
    "/root/autodl-tmp/coda-lm-llava-format/Chinese/Test-00000-of-00004.parquet",
    "/root/autodl-tmp/coda-lm-llava-format/Chinese/Test-00001-of-00004.parquet",
 #   "/root/autodl-tmp/coda-lm-llava-format/Chinese/Test-00002-of-00004.parquet",
    "/root/autodl-tmp/coda-lm-llava-format/Chinese/Test-00003-of-00004.parquet"
]

# =====================================================
# 1. 输出目录
# =====================================================
base_dir = Path("/root/autodl-tmp/coda-lm-llava-format")
image_output_dir = base_dir / "converted_images_test"
os.makedirs(image_output_dir, exist_ok=True)

output_json_path = base_dir / "qwen_test.json"

print(f"图片保存目录: {image_output_dir}")
print(f"JSON 输出文件: {output_json_path}")

# =====================================================
# 2. 流式写 JSON 文件
# =====================================================
with open(output_json_path, "w", encoding="utf-8") as f_json:
    f_json.write('{"type": "vision_language_conversation", "instances": [\n')
    
    first_instance = True
    global_idx = 0
    
    # =====================================================
    # 3. 逐个 parquet 文件处理
    # =====================================================
    for parquet_path in parquet_files:
        parquet_path = Path(parquet_path)
        parquet_name = parquet_path.stem
        print(f"\n开始处理文件: {parquet_path}")
        
        # --- 修改点：使用 PyArrow 创建文件对象，而不是一次性加载 ---
        parquet_file = pq.ParquetFile(parquet_path)
        
        # --- 修改点：按 Batch（批次）迭代读取，batch_size=64 防止内存爆炸 ---
        # 这里的 batch_size 可以根据显存/内存调整，图片大就调小一点
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=64)):
            
            # 将当前这一个小批次转换为 Pandas DataFrame
            df_batch = batch.to_pandas()
            
            for idx, row in df_batch.iterrows():
                try:
                    sample_id = row["id"]
                    
                    # ---------- 图片保存 ----------
                    # 注意：如果数据格式是 bytes，直接写；如果是 struct，需解析
                    image_data = row["image"]
                    if isinstance(image_data, dict) and "bytes" in image_data:
                        image_bytes = image_data["bytes"]
                    else:
                        # 兼容某些格式直接就是 bytes 的情况
                        image_bytes = image_data
                    
                    image_filename = f"{parquet_name}_{sample_id}.png"
                    image_path = image_output_dir / image_filename
                    
                    # 只有当图片不存在时才写入（可选，加快断点续传速度）
                    # if not image_path.exists(): 
                    with open(image_path, "wb") as f_img:
                        f_img.write(image_bytes)
                    
                    # ---------- 对话处理 ----------
                    messages = []
                    conversations = row["conversations"]
                    
                    for turn in conversations:
                        role = "user" if turn["from"] == "human" else "assistant"
                        text = turn["value"].strip()
                        
                        content = []
                        if role == "user":
                            if text.startswith("<image>"):
                                content.append({"type": "image", "image": str(image_path.resolve())}) # Qwen 格式微调有时需要放在 content 里，或者如下处理
                                # 注意：如果是 Qwen2-VL 原生格式，通常是 text 里放 <image>，然后 images 列表放路径
                                # 这里保持你原本的逻辑：
                                content.append({"type": "text", "text": "<image>"})
                                rest_text = text.replace("<image>", "").strip()
                                if rest_text:
                                    content.append({"type": "text", "text": rest_text})
                            else:
                                content.append({"type": "text", "text": text})
                        else:
                            content.append({"type": "text", "text": text})
                        
                        messages.append({"role": role, "content": content})
                    
                    # ---------- 构造 JSON 实例 ----------
                    instance = {
                        "id": f"{parquet_name}_{sample_id}",
                        "images": [str(image_path.resolve())],
                        "messages": messages
                    }
                    
                    # ---------- 流式写入 JSON ----------
                    if not first_instance:
                        f_json.write(",\n")
                    json.dump(instance, f_json, ensure_ascii=False)
                    first_instance = False
                    
                    global_idx += 1
                    if global_idx % 100 == 0:
                        print(f"  已累计处理 {global_idx} 条样本...", end="\r")
                
                except Exception as e:
                    print(f"\n[Error] 处理样本 {sample_id} 失败: {e}")
                    continue

            # --- 修改点：手动清理内存 ---
            del df_batch
            # 每处理完几个 batch 强制回收一次内存
            if batch_idx % 10 == 0:
                gc.collect()

    f_json.write("\n]}\n")

print("\n" + "="*60)
print("✅ 全部 parquet 文件处理完成")
print(f"总样本数: {global_idx}")
print(f"JSON 输出文件: {output_json_path}")
print("="*60)

# =========================
# Qwen-VL-7B 格式转换
# =========================

INPUT_JSON = "/root/autodl-tmp/coda-lm-llava-format/qwen_test.json"
OUTPUT_JSON = "/root/autodl-tmp/coda-lm-llava-format/qwen_vl_7b_test.json"


def extract_text_from_content(content_list):
    texts = []
    for c in content_list:
        if c.get("type") == "text":
            t = c.get("text", "").strip()
            if t and t != "<image>":
                texts.append(t)
    return "\n".join(texts)


def convert_to_qwen2p5_vl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    assert "instances" in raw

    out = []

    for inst in raw["instances"]:
        inst_id = inst["id"]
        image_path = inst["images"][0]

        conversations = []
        first_human = True

        for msg in inst["messages"]:
            role = msg["role"]
            text = extract_text_from_content(msg["content"])
            if not text:
                continue

            if role == "user":
                if first_human:
                    text = "<img>\n" + text
                    first_human = False
                conversations.append({
                    "from": "human",
                    "value": text
                })
            else:
                conversations.append({
                    "from": "assistant",
                    "value": text
                })

        if len(conversations) < 2:
            continue

        out.append({
            "id": inst_id,
            "image": image_path,
            "conversations": conversations
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ 转换完成：{len(out)} 条")
    print(f"📤 输出文件：{output_path}")


if __name__ == "__main__":
    convert_to_qwen2p5_vl(INPUT_JSON, OUTPUT_JSON)
