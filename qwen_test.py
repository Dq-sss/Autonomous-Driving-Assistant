from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import time

# 模型路径
local_model_path = "/root/autodl-fs/Qwen2.5-VL-7B-Instruct"
local_image_path = "/autodl-fs/data/coda-lm-llava-format/converted_images/Mini_general_0.png"

print("正在加载模型（GPU模式）...")

# 修正：device_map="auto" 或 device_map="cuda"
model = AutoModelForImageTextToText.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,      # GPU建议使用float16或bfloat16以节省显存
    device_map="auto",               # 修正：使用"auto"或"cuda"而非"gpu"
    # low_cpu_mem_usage=True,        # 可选项，通常不需要在GPU模式下
)

processor = AutoProcessor.from_pretrained(local_model_path)
print("模型加载完成！")

# 简化输入
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": local_image_path},
            {"type": "text", "text": "这是一张从自车视角捕捉的交通图像。重点关注影响自车驾驶行为的物体：车辆（轿车、卡车、公交车等）、易受伤害的道路使用者（行人、自行车骑行者、摩托车骑行者）、交通标志（禁止停车、警告、指示等）、交通信号灯（红灯、绿灯、黄灯）、交通锥、障碍物，以及杂物（碎片、垃圾桶、动物等）。请不要讨论以上七类以外的任何物体。请描述每个物体的外观、位置、方向，并解释为什么它会影响自车的行为"},
        ],
    }
]

# 准备输入
print("正在处理输入...")
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# 确保输入数据也在GPU上
inputs = inputs.to(model.device)
print(f"输入处理完成，设备: {model.device}")
print("-" * 40)

# 生成回答
print("正在生成回答...")
start_time = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,           # 稍微增加生成长度
        do_sample=False,              # 贪婪解码，速度更快
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id
    )

generation_time = time.time() - start_time

# 解码输出
text = processor.batch_decode(
    outputs[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(f"\n生成耗时: {generation_time:.2f}秒")
print("=" * 50)
print("模型回答:")
print("=" * 50)
print(text)
print("=" * 50)