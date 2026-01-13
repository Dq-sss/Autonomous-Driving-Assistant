import json
import torch
import os
import jieba
import re
import numpy as np

# [Critical Fix] 解决 DataLoader 多进程死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader

# 引入 ROUGE 和 BERTScore
from pycocoevalcap.rouge.rouge import Rouge
from bert_score import BERTScorer

# =========================
# 1. 全局配置
# =========================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BASE_MODEL_PATH = "/root/autodl-fs/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = "/root/autodl-tmp/coda-lm-llava-format/output_qlora"
TEST_JSON = "../qwen_vl_7b_test.json"
BERT_MODEL_PATH = "/root/autodl-tmp/hf_models/bert-base-chinese"
BLEURT_MODEL_PATH = "/root/autodl-tmp/hf_models/bleurt-tiny" 

adapter_name = os.path.basename(os.path.normpath(ADAPTER_PATH))
PREDICT_DIR = os.path.join("predict", adapter_name)
REFERENCE_JSON = os.path.join(PREDICT_DIR, "reference.json")
ANSWER_JSON = os.path.join(PREDICT_DIR, "answer.json")
os.makedirs(PREDICT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2. 自定义指标：Entity Recall
# =========================
def calculate_entity_recall(res_dict, gts_dict):
    driving_keywords = [
        "减速", "刹车", "制动", "加速", "左转", "右转", "停止", "避让", 
        "行人", "非机动车", "红绿灯", "交通灯", "实线", "虚线", "保持车距", 
        "障碍物", "施工", "汇入", "变道", "优先通行"
    ]
    scores = []
    for idx in gts_dict:
        gt_text = gts_dict[idx][0]
        pred_text = res_dict[idx][0]
        relevant_entities = [w for w in driving_keywords if w in gt_text]
        if not relevant_entities: continue
        found = [w for w in relevant_entities if w in pred_text]
        scores.append(len(found) / len(relevant_entities))
    return np.mean(scores) if scores else 0.0

# =========================
# 3. 数据集逻辑
# =========================
class VQAEvalDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tasks = []
        self._prepare_tasks()

    def _prepare_tasks(self):
        global_idx = 0
        for item in self.data:
            image_path = item["image"]
            convs = item["conversations"]
            for i in range(0, len(convs)-1, 2):
                if convs[i]["from"] == "human":
                    messages = [
                        {"role": "system", "content": "你是一名专业的自动驾驶决策专家，请简洁、客观地回答交通场景问题。"},
                        {"role": "user", "content": [{"type":"image", "image": image_path}, {"type":"text", "text": convs[i]["value"]}]}
                    ]
                    self.tasks.append({"id": global_idx, "messages": messages, "image_path": image_path, "gt": convs[i+1]["value"]})
                    global_idx += 1

    def __len__(self): return len(self.tasks)
    def __getitem__(self, idx):
        task = self.tasks[idx]
        try:
            image = Image.open(task["image_path"]).convert("RGB")
            image = image.resize(((image.width//28)*28, (image.height//28)*28))
        except:
            image = Image.new("RGB", (224,224), (0,0,0))
        return {"id": task["id"], "messages": task["messages"], "image": image, "gt": task["gt"]}

def custom_collate_fn(batch):
    return [item["id"] for item in batch], [item["messages"] for item in batch], [item["image"] for item in batch], [item["gt"] for item in batch]

# =========================
# 4. 主评测函数
# =========================
def evaluate(batch_size=16, max_new_tokens=512):
    print(f"Loading processor and model from {BASE_MODEL_PATH}...")
    local_processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    local_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2", trust_remote_code=True
    )
    if os.path.exists(ADAPTER_PATH):
        print(f"Loading adapter from {ADAPTER_PATH}...")
        local_model = PeftModel.from_pretrained(local_model, ADAPTER_PATH)
    local_model.eval()

    dataset = VQAEvalDataset(TEST_JSON)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)

    gts, res = {}, {}

    print("Step 1: Inference...")
    for b_ids, b_msgs, b_imgs, b_gts in tqdm(dataloader):
        texts = [local_processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in b_msgs]
        inputs = local_processor(text=texts, images=b_imgs, padding=True, return_tensors="pt").to(DEVICE)
        
        with torch.inference_mode():
            generated_ids = local_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
        
        for i, (gen_ids, input_len) in enumerate(zip(generated_ids, [len(ids) for ids in inputs.input_ids])):
            out_text = local_processor.decode(gen_ids[input_len:], skip_special_tokens=True).strip()
            gts[b_ids[i]] = [b_gts[i]]
            res[b_ids[i]] = [out_text]

    # --- [恢复保存逻辑] 推理结束，先保存预测结果 ---
    print(f"Saving gts to {REFERENCE_JSON}...")
    with open(REFERENCE_JSON, "w", encoding="utf-8") as f:
        json.dump(gts, f, indent=2, ensure_ascii=False)

    print(f"Saving res to {ANSWER_JSON}...")
    with open(ANSWER_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # 释放显存
    del local_model, local_processor
    torch.cuda.empty_cache()

    print("Step 2: Calculating Metrics...")
    
    # 1. BERTScore
    print("Calculating BERTScore...")
    sorted_ids = sorted(gts.keys())
    cands = [res[i][0] for i in sorted_ids]
    refs = [gts[i][0] for i in sorted_ids]
    
    scorer = BERTScorer(model_type=BERT_MODEL_PATH, num_layers=9, lang="zh", device=DEVICE)
    _, _, f1 = scorer.score(cands, refs)
    bert_f1 = f1.mean().item()
    del scorer
    torch.cuda.empty_cache()

    # 2. BLEURT
    print("Calculating BLEURT...")
    bleurt_val = 0.0
    try:
        bl_tokenizer = AutoTokenizer.from_pretrained(BLEURT_MODEL_PATH)
        bl_model = AutoModelForSequenceClassification.from_pretrained(BLEURT_MODEL_PATH).to(DEVICE)
        bl_model.eval()
        all_bl_scores = []
        for i in range(0, len(cands), batch_size):
            b_cands = cands[i : i + batch_size]
            b_refs = refs[i : i + batch_size]
            bl_inputs = bl_tokenizer(b_cands, b_refs, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                outputs = bl_model(**bl_inputs)
                all_bl_scores.extend(outputs.logits.flatten().cpu().tolist())
        bleurt_val = np.mean(all_bl_scores)
        del bl_model, bl_tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"BLEURT failed: {e}")

    # 3. ROUGE-L
    def tokenize_zh(t_list):
        return [" ".join(jieba.cut(re.sub(r"\s+", "", t_list[0])))]
    gts_tok = {k: tokenize_zh(v) for k, v in gts.items()}
    res_tok = {k: tokenize_zh(v) for k, v in res.items()}
    rouge_score = Rouge().compute_score(gts_tok, res_tok)[0]

    # 4. Entity Recall
    entity_recall = calculate_entity_recall(res, gts)

    print("\n" + "="*35)
    print(f"{'Metric':<20} | {'Score':<10}")
    print("-" * 35)
    print(f"{'ROUGE-L':<20} | {rouge_score:.4f}")
    print(f"{'BERTScore (F1)':<20} | {bert_f1:.4f}")
    print(f"{'Entity Recall':<20} | {entity_recall:.4f}")
    print(f"{'BLEURT':<20} | {bleurt_val:.4f}")
    print("="*35)

if __name__ == "__main__":
    evaluate(batch_size=16)