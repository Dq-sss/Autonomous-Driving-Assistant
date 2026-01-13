import os
import time
import json
import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import AdaLoraConfig, get_peft_model, TaskType
from dataprocess import get_dataset_and_collator

# =====================================================
# 1. 全局配置（RTX 4090）
# =====================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_NAME = "/root/autodl-fs/Qwen2.5-VL-7B-Instruct"
DATA_PATH = "/root/autodl-fs/coda-lm-llava-format/qwen_vl_7b.json"
OUTPUT_DIR = "/root/autodl-tmp/coda-lm-llava-format/output_adaqlora"
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")

MAX_LENGTH = 1536
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
NUM_WORKERS = 8
NUM_EPOCHS = 3

os.makedirs(METRIC_DIR, exist_ok=True)

# =====================================================
# 4. 训练监控 Callback
# =====================================================
class TrainMetricsCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.step_start_time = None

        with open(self.log_file, "w") as f:
            f.write("")

    def on_step_begin(self, args, state, control, **kwargs):
        # ✅ 只在 logging step 做同步和计时
        if state.global_step % args.logging_steps != 0:
            return

        torch.cuda.synchronize()
        self.step_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return

        # ❌ 不再 synchronize（避免性能损失）
        step_time = None
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time

        record = {
            "step": state.global_step,
            "loss": float(logs["loss"]),
            "lr": logs.get("learning_rate", None),
            "step_time_sec": round(step_time, 4) if step_time else None,
            "gpu_mem_alloc_MB": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "gpu_mem_max_MB": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        }

        print(
            f"[Step {record['step']}] "
            f"loss={record['loss']:.4f} | "
            f"time={record['step_time_sec']}s | "
            f"mem={record['gpu_mem_alloc_MB']}MB"
        )

        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

# =====================================================
# 5. 训练
# =====================================================
def train():
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 🔥 Vision token 限制
    processor.image_processor.min_pixels = 224 * 224
    processor.image_processor.max_pixels = 336 * 336

    # 获取数据处理结果
    train_dataset, data_collator = get_dataset_and_collator(DATA_PATH, processor, MAX_LENGTH)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )

    for p in model.parameters():
        p.requires_grad = False

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # 样本数
    num_samples = len(train_dataset)

    # 有效 batch size
    effective_batch_size = BATCH_SIZE * ACCUMULATION_STEPS

    # 每 epoch 的 step
    steps_per_epoch = num_samples // effective_batch_size

    # 总 step
    total_step = steps_per_epoch * NUM_EPOCHS

    print("Total training steps:", total_step)


    model = get_peft_model(
        model,
        AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            init_r=32,
            target_r=24,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=800,
            deltaT=20,
            total_step=total_step,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj","v_proj","down_proj"],
            orth_reg_weight=0.1
        )
    )
    model.print_trainable_parameters()


    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        logging_steps=10,
        save_strategy="steps",         # 修改：按步数保存
        save_steps=200,                 # 修改：每 500 步保存一次 (可根据总步数调整)
        save_total_limit=3,            # 修改：最多保留最近 2 个 Checkpoint
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=NUM_WORKERS,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[
            TrainMetricsCallback(
                log_file=os.path.join(METRIC_DIR, "train_log.jsonl")
            )
        ]
    )

    print("🔥 Start training...")
    start = time.time()
    
    # Check if there are constraints to resume from checkpoint
    checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        # 查找是否存在 checkpoint-xxx 文件夹
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if len(checkpoints) > 0:
            checkpoint = True
            print(f"🔄 检测到 Checkpoint，准备从 {OUTPUT_DIR} 恢复训练...")

    trainer.train(resume_from_checkpoint=checkpoint)
    print(f"⏱ Total time: {(time.time()-start)/60:.2f} min")


    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

# =====================================================
if __name__ == "__main__":
    train()
