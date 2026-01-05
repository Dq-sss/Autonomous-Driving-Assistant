import json
import random
import time
from openai import OpenAI

# =========================
# 1. 初始化客户端（SiliconFlow · OpenAI Compatible）
# =========================
API_KEY = "sk-hyayuucpskezqfipnurdcyonpamhobpgclicouclmbujjgab"
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.siliconflow.cn/v1",
    timeout=60.0
)

# =========================
# 2. 指令模板池
# =========================
INSTRUCTION_TEMPLATES = [
    "当前场景中是否存在潜在驾驶风险？请指出并解释原因。",
    "在该场景下，自车是否需要减速、制动或变道？请给出决策依据。",
    "请指出当前场景中最关键的一个交通参与者，并说明其位置与影响。",
    "如果前方车辆突然紧急刹车，自车应采取什么应急策略？",
    "根据当前道路结构和车辆分布，自车最安全的行驶策略是什么？",
    "当前场景中是否存在可能影响通行效率的因素？请分析。",
    "是否存在需要提前预判的交通行为？请说明理由。"
]

# =========================
# 3. 教师模型 System Prompt
# =========================
TEACHER_SYSTEM_PROMPT = """你是一名专业的自动驾驶决策与风险分析专家。
请严格基于给定的交通场景文字描述回答问题。
约束：
1. 只能基于描述中明确提到的交通要素进行分析。
2. 不得引入描述中未出现的车辆、行人或信号设施。
3. 所有决策必须有明确理由，且逻辑清晰。
4. 语言专业、客观、简洁，避免泛泛而谈。
"""

# =========================
# 4. 教师模型生成答案 - 批量版本 (按场景批量)
# =========================
def get_teacher_response_batch(scene_description, question_list):
    """
    一次性为一个场景的多个问题生成答案
    :param scene_description: 场景描述
    :param question_list: 问题列表，如 [q1, q2, q3]
    :return: 答案列表，与问题顺序对应，如 [ans1, ans2, ans3]
    """
    try:
        print(f"    🤖 正在批量生成 {len(question_list)} 个问题的分析...")
        start_time = time.time()
        
        responses = []
        for q in question_list:
            user_question = f"基于上述交通场景，{q}"
            response = client.chat.completions.create(
                model="Qwen/Qwen3-8B",
                messages=[
                    {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"交通场景描述如下：\n"
                            f"{scene_description}\n\n"
                            f"问题：{user_question}"
                        )
                    }
                ],
                temperature=0.5,
                max_tokens=300,
                timeout=30.0
            )
            answer = response.choices[0].message.content.strip()
            responses.append(answer)
        
        elapsed_time = time.time() - start_time
        avg_time_per_question = elapsed_time / len(question_list)
        print(f"    ✅ 批量生成成功 (共耗时{elapsed_time:.1f}秒， 平均每个问题{avg_time_per_question:.1f}秒)")
        for i, ans in enumerate(responses):
            preview = ans[:50] + "..." if len(ans) > 50 else ans
            print(f"      问题{i+1}摘要: {preview}")
        return responses
        
    except Exception as e:
        print(f"    ❌ 批量调用失败: {e}")
        return [None] * len(question_list)

# =========================
# 5. 数据增强主流程（已集成批量处理）
# =========================
def process_and_augment_data(
    input_json_path,
    output_json_path,
    num_questions_per_image=2
):
    print("📂 正在读取原始数据文件...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total_instances = len(data["instances"])
    print(f"📊 数据加载完成，共发现 {total_instances} 个交通场景实例")
    print("="*60)
    
    processed_count = 0
    enhanced_qa_pairs = 0
    total_start_time = time.time()
    
    for idx, item in enumerate(data["instances"]):
        processed_count += 1
        messages = item["messages"]
        
        original_description = None
        for msg in messages:
            if msg["role"] == "assistant":
                original_description = msg["content"][0]["text"]
                break
        
        if not original_description:
            print(f"[{idx+1}/{total_instances}] ⚠️  跳过 {item['id']}: 未找到场景描述")
            continue
        
        available_questions = min(num_questions_per_image, len(INSTRUCTION_TEMPLATES))
        selected_questions = random.sample(INSTRUCTION_TEMPLATES, k=available_questions)
        
        scene_preview = original_description[:80] + "..." if len(original_description) > 80 else original_description
        print(f"[{idx+1}/{total_instances}] 🚗 处理 {item['id']}")
        print(f"   描述预览: {scene_preview}")
        print(f"   将为该场景生成 {len(selected_questions)} 个增强问题")
        
        all_questions_for_this_scene = []
        for q in selected_questions:
            user_question = f"基于上述交通场景，{q}"
            all_questions_for_this_scene.append(user_question)
        
        answers = get_teacher_response_batch(original_description, selected_questions)
        
        for user_question, answer in zip(all_questions_for_this_scene, answers):
            if answer is None:
                print(f"   ⚠️  问题生成失败，跳过: {user_question[:30]}...")
                continue
                
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_question}]
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            })
            enhanced_qa_pairs += 1
        
        print(f"   📝 本场景处理完成，成功增加 {len([a for a in answers if a is not None])} 个QA对")
        time.sleep(0.1)
        print("-"*50)
    
    total_elapsed_time = time.time() - total_start_time
    print(f"\n💾 正在保存增强后的数据到 {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("="*60)
    print(f"🎉 数据增强完成！")
    print(f"   • 总耗时: {total_elapsed_time / 3600:.2f} 小时")
    print(f"   • 共处理场景: {processed_count}/{total_instances}")
    print(f"   • 新增QA对话对: {enhanced_qa_pairs}")
    print(f"   • 平均每个场景耗时: {total_elapsed_time / processed_count if processed_count else 0:.2f} 秒")
    print(f"   • 输出文件: {output_json_path}")
    print("="*60)


# =========================
# 6. Qwen-VL-7B 格式转换
# =========================
INPUT_JSON = "qwen_finetune.json"
OUTPUT_JSON = "qwen_vl_7b.json"


def extract_text_from_content(content_list):
    texts = []
    for c in content_list:
        if c.get("type") == "text":
            texts.append(c.get("text", "").strip())
    return "\n".join([t for t in texts if t])


def convert_to_qwen_vl7b(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    assert "instances" in raw_data, "❌ 输入 JSON 不包含 instances 字段"

    output_data = []

    for inst in raw_data["instances"]:
        inst_id = inst.get("id", "")
        images = inst.get("images", [])
        messages = inst.get("messages", [])

        if not images:
            continue

        image_path = images[0]

        conversations = []
        first_human = True

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", [])

            text = extract_text_from_content(content)
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

            elif role == "assistant":
                conversations.append({
                    "from": "assistant",
                    "value": text
                })

        if len(conversations) < 2:
            continue

        output_data.append({
            "id": inst_id,
            "image": image_path,
            "conversations": conversations
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("✅ 转换完成")
    print(f"📥 输入文件: {input_path}")
    print(f"📤 输出文件: {output_path}")
    print(f"📊 样本数量: {len(output_data)}")


# =========================
# 7. Main
# =========================
if __name__ == "__main__":
    print("🚀 开始交通场景数据增强流程")
    print(f"使用的教师模型: Qwen/Qwen3-8B")
    print(f"优化策略: 场景内问题批量处理 | 生成长度限制:200 | 调用间隔缩短:0.5秒")
    print(f"指令模板池: {len(INSTRUCTION_TEMPLATES)} 个预设问题")
    print("="*60)

    process_and_augment_data(
        input_json_path="qwen_finetune.json",
        output_json_path="qwen_finetune_aug_batch_optimized.json",
        num_questions_per_image=2
    )

    convert_to_qwen_vl7b(
        INPUT_JSON,
        OUTPUT_JSON
    )

