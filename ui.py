import torch
import time
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import gradio as gr

# =========================
# 1. 模型加载
# =========================
print("正在初始化模型，请稍候...")

BASE_MODEL_PATH = "/root/autodl-fs/Qwen2.5-VL-7B-Instruct"
LORA_PATH = "output"

print("加载基础模型...")
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("加载 LoRA...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

print("加载 Processor...")
try:
    processor = AutoProcessor.from_pretrained(LORA_PATH, trust_remote_code=True)
except:
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print("✅ 模型加载完成")

# =========================
# 2. 默认 Prompt
# =========================
DEFAULT_PROMPT = (
    "这是一张从自车视角捕捉的交通图像。重点关注影响自车驾驶行为的物体："
    "车辆、行人、自行车、交通标志、信号灯、交通锥、障碍物。"
    "请描述它们的位置、状态及对驾驶的影响。"
)

# =========================
# 3. 推理函数
# =========================
def generate_response(image, text):
    if image is None:
        return "请上传图片"
    if not text:
        return "请输入提示词"

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text}
        ]
    }]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

    output_ids = output_ids[:, inputs.input_ids.shape[1]:]
    result = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    return result + f"\n\n⏱ 推理耗时 {time.time() - start:.2f}s"

# =========================
# 4. Gradio 逻辑
# =========================
def chat(image, text, history):
    history = history or []
    history.append([text, generate_response(image, text)])
    return history, ""

def clear():
    return None, DEFAULT_PROMPT, []

# =========================
# 5. Gradio UI
# =========================
with gr.Blocks(title="Qwen2.5-VL Traffic Assistant") as demo:
    gr.Markdown("# 🚗 Qwen2.5-VL 交通场景理解")

    with gr.Row():
        image = gr.Image(type="pil", label="输入图像", height=400)
        chatbot = gr.Chatbot(height=400)

    text = gr.Textbox(
        lines=5,
        value=DEFAULT_PROMPT,
        label="提示词"
    )

    with gr.Row():
        submit = gr.Button("分析")
        clear_btn = gr.Button("清空")

    submit.click(chat, [image, text, chatbot], [chatbot, text])
    clear_btn.click(clear, [], [image, text, chatbot])

# =========================
# 6. 启动（关键修改在这里）
# =========================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=6006,
        share=False,
        show_api=False,   # 🔥 关键：关闭 API schema
        inbrowser=False
    )
