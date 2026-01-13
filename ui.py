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
LORA_PATH = "/root/autodl-tmp/coda-lm-llava-format/output_adaqlora"

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
    print("OK!")
except:
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print("✅ 模型加载完成")

# =========================
# 2. 默认 Prompt
# =========================
DEFAULT_PROMPT = ""

# =========================
# 3. 推理函数（只负责模型）
# =========================
def generate_response(image, text):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text}
        ]
    }]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
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
# 4. Gradio Chat 逻辑（核心修改）
# =========================
def chat(image, text, history):
    history = history or []

    # ---------- 前置校验 ----------
    if image is None:
        history.append(["❌ 输入错误", "请先上传一张图片"])
        yield history, ""
        return

    if not text or text.strip() == "":
        history.append(["❌ 输入错误", "请输入问题文本"])
        yield history, ""
        return

    # ---------- ① 立刻显示用户问题 ----------
    history.append([text, "🤖 正在生成中，请稍候..."])
    yield history, ""

    # ---------- ② 执行模型推理 ----------
    answer = generate_response(image, text)

    # ---------- ③ 更新最后一条回答 ----------
    history[-1][1] = answer
    yield history, ""

def clear():
    return None, "", []

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
        label="提问",
        placeholder="请输入你的问题（必须填写）"
    )

    with gr.Row():
        submit = gr.Button("发送")
        clear_btn = gr.Button("清空")

    submit.click(
        chat,
        inputs=[image, text, chatbot],
        outputs=[chatbot, text]
    )

    clear_btn.click(clear, [], [image, text, chatbot])

# =========================
# 6. 启动
# =========================
if __name__ == "__main__":
    demo.queue()  # 🔥 允许 yield / 流式更新
    demo.launch(
        server_name="0.0.0.0",
        server_port=6006,
        share=False,
        show_api=False,
        inbrowser=False
    )
