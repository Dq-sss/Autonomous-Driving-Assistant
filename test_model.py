import torch
import time
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# =========================
# 1. 配置路径
# =========================
# 基础模型路径 (必须与训练时一致)
BASE_MODEL_PATH = "/root/autodl-fs/Qwen2.5-VL-7B-Instruct"
# LoRA 权重保存路径 (训练脚本中的 OUTPUT_DIR)
LORA_PATH = "output" 
# 测试图片路径
TEST_IMAGE_PATH = "/autodl-fs/data/coda-lm-llava-format/converted_images/Mini_general_0.png"

print("Step 1: 正在加载基础模型...")
# 加载基础模型
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,  # 推荐使用 bf16
    device_map="auto",
    trust_remote_code=True
)

print(f"Step 2: 正在加载 LoRA 权重从 {LORA_PATH} ...")
# 加载 LoRA 适配器
# 这会将 LoRA 层合并到基础模型中，或者动态挂载
model = PeftModel.from_pretrained(model, LORA_PATH)

# 加载 Processor (通常 LoRA 文件夹里也会保存一份，或者用基础模型的)
try:
    processor = AutoProcessor.from_pretrained(LORA_PATH, trust_remote_code=True)
except:
    print("LoRA 目录未找到 processor，尝试从基础模型加载...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print("模型加载完成！")

# =========================
# 2. 准备输入
# =========================
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": TEST_IMAGE_PATH},
            {"type": "text", "text": "这是一张从自车视角捕捉的交通图像。重点关注影响自车驾驶行为的物体：车辆（轿车、卡车、公交车等）、易受伤害的道路使用者（行人、自行车骑行者、摩托车骑行者）、交通标志（禁止停车、警告、指示等）、交通信号灯（红灯、绿灯、黄灯）、交通锥、障碍物，以及杂物（碎片、垃圾桶、动物等）。请不要讨论以上七类以外的任何物体。请描述每个物体的外观、位置、方向，并解释为什么它会影响自车的行为"},
        ],
    }
]

print("Step 3: 处理输入数据...")
# 使用 apply_chat_template 处理文本和图像
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# 加载图像并处理
from PIL import Image
import requests

# 简单的图像加载逻辑
image = Image.open(TEST_IMAGE_PATH).convert("RGB")

inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt",
)

# 移动到 GPU
inputs = inputs.to(model.device)

# =========================
# 3. 推理生成
# =========================
print("Step 4: 开始生成...")
start_time = time.time()

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,  # 增加长度以便生成完整描述
        do_sample=False,     # 贪婪解码，结果更稳定
        # 如果需要多样性，可以开启以下参数:
        # do_sample=True,
        # temperature=0.7,
        # top_p=0.9,
    )

# 获取生成的 token (去掉输入的 prompt 部分)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)[0]

generation_time = time.time() - start_time

print("=" * 50)
print(f"生成耗时: {generation_time:.2f}秒")
print("=" * 50)
print("LoRA 模型回答:")
print(output_text)
print("=" * 50)