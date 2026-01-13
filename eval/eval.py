from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from tqdm import tqdm
import time
import os
import json
import argparse
import re

class GPTBatcher:
    """
    Batch processing for GPT evaluation
    """

    def __init__(self, 
                 api_key="d808ca603c19493aa31ce03251c763c5.Z75lBbIBAKgAAnWo", 
                 model_name="glm-4.7", 
                 api_base_url="https://open.bigmodel.cn/api/paas/v4",
                 system_prompt="",
                 temperature=0, 
                 num_workers=32,
                 timeout_duration=60,
                 retry_attempts=2
                 ):
        
        self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        self.model_name = model_name
        self.system_prompt = "你是一位公正的评判者，负责评估自动驾驶AI助手生成的预测文本质量。你需要将预测文本与参考文本进行比较，重点关注对影响本车驾驶行为的物体描述，以及这些物体产生影响的原因解释。你的评估标准应包括：准确性（检查预测文本是否正确识别了参考文本中提到的物体）、抑制幻觉（确保参考文本中未提到的物体不会错误地出现在预测文本中）、相关性（评估物体对本车驾驶行为产生影响的原因在参考文本和预测文本中是否一致）。请尽可能保持客观。不要让预测文本的长度影响你的评估。请充分发挥你的文本理解能力，自由匹配相似度高的物体，适当忽略物体的相对位置和颜色属性差异。在提供简短的解释后，你必须严格按照以下格式对回答进行1到10分的打分：\"[[评分]]\"，例如：\"评分：[[10]]\"。"
        self.temperature = temperature
        self.num_workers = num_workers
        self.timeout_duration = timeout_duration
        self.retry_attempts = retry_attempts
        self.miss_index = []
        if api_base_url:
            self.client.base_url = api_base_url

    def create_messages(self, message):
        ret = []
        # system prompt
        ret.append({
            "role": "system",
            "content": self.system_prompt
        })

        # Load few-shot examples from file
        few_shot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "few_shot")
        examples = []
        
        # Load high quality examples
        high_path = os.path.join(few_shot_dir, "high.json")
        if os.path.exists(high_path):
            try:
                with open(high_path, "r", encoding="utf-8") as f:
                    high_data = json.load(f)
                    if isinstance(high_data, list):
                        examples.extend(high_data)
                    else:
                        examples.append(high_data)
            except Exception as e:
                print(f"Warning: Could not load {high_path}: {e}")
        else:
            print(f"Warning: {high_path} not found.")

        # Load low quality examples
        low_path = os.path.join(few_shot_dir, "low.json")
        if os.path.exists(low_path):
            try:
                with open(low_path, "r", encoding="utf-8") as f:
                    low_data = json.load(f)
                    if isinstance(low_data, list):
                        examples.extend(low_data)
                    else:
                        examples.append(low_data)
            except Exception as e:
                print(f"Warning: Could not load {low_path}: {e}")
        else:
            print(f"Warning: {low_path} not found.")

        template = "[The Start of Reference Text]\n{}\n[The End of Reference Text]\n\n[The Start of Prediction Text]\n{}\n[The End of Prediction Text]"

        for ex in examples:
            ret.append({
                "role": "user", 
                "content": template.format(ex["reference"], ex["prediction"])
            })
            ret.append({
                "role": "assistant", 
                "content": ex["response"]
            })

        ret.append({
            "role": "user", 
            "content": template.format(message["reference"], message["prediction"])
        })
        return ret

    def get_attitude(self, ask_text):
        index, message_content = ask_text
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_content,
                temperature=self.temperature,
            )
            return (index, completion.choices[0].message.content)
        except Exception as e:
            print(f"Error occurred for index {index}: {e}")
            self.miss_index.append(index)
            return (index, None)

    def process_attitude(self, message_list):
        new_list = []
        num_workers = self.num_workers
        timeout_duration = self.timeout_duration
        retry_attempts = self.retry_attempts
    
        executor = ThreadPoolExecutor(max_workers=num_workers)
        message_chunks = list(self.chunk_list(message_list, num_workers))
        try:
            for chunk in tqdm(message_chunks, desc="Processing messages"):
                future_to_message = {executor.submit(self.get_attitude, message): message for message in chunk}
                for _ in range(retry_attempts):
                    done, not_done = wait(future_to_message.keys(), timeout=timeout_duration)
                    for future in not_done:
                        future.cancel()
                    new_list.extend(future.result() for future in done if future.done())
                    if len(not_done) == 0:
                        break
                    # Retry for not done
                    future_to_message = {executor.submit(self.get_attitude, future_to_message[future]): future for future in not_done}
        except Exception as e:
            print(f"Error occurred in batch processing: {e}")
        finally:
            executor.shutdown(wait=False)
            return new_list

    def complete_attitude_list(self, attitude_list, max_length):
        completed_list = []
        current_index = 0
        # attitude_list is list of tuples (index, value)
        # sort just in case
        attitude_list.sort(key=lambda x: x[0])
        
        for item in attitude_list:
            index, value = item
            # Fill in missing indices
            while current_index < index:
                print(f"Warning: Missing result for index {current_index}, filling None")
                completed_list.append((current_index, None))
                self.miss_index.append(current_index)
                current_index += 1
            # Add the current element from the list
            completed_list.append(item)
            current_index = index + 1
            
        while current_index < max_length:
            print(f"Warning: Missing result for index {current_index}, filling None")
            self.miss_index.append(current_index)
            completed_list.append((current_index, None))
            current_index += 1
            
        return completed_list

    def chunk_list(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def handle_message_list(self, all_messages):
        # all_messages is a list of lists of dicts (messages for chat completion)
        indexed_list = [(index, data) for index, data in enumerate(all_messages)]
        max_length = len(indexed_list)
        
        attitude_list = self.process_attitude(indexed_list)
        attitude_list.sort(key=lambda x: x[0])
        
        attitude_list = self.complete_attitude_list(attitude_list, max_length)
        result_texts = [x[1] for x in attitude_list]
        return result_texts
    
    def get_miss_index(self):
        return self.miss_index

def parse_score(text):
    if text is None:
        return None
    # Try different patterns
    # Format: [[rating]]
    try:
        match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', text)
        if match:
            return float(match.group(1))
    except:
        pass
    
    # Fallback/Loose matching
    try:
        match = re.search(r'Rating:\s*\[\[(\d+(?:\.\d+)?)\]\]', text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except:
        pass
        
    return None

if __name__ == "__main__":
    # 配置路径
    ADAPTER_PATH = "/root/autodl-fs/coda-lm-llava-format/inference"
    adapter_name = os.path.basename(os.path.normpath(ADAPTER_PATH))
    PREDICT_DIR = os.path.join(ADAPTER_PATH, "predict/output_adaqlora")
    print(PREDICT_DIR)
    REFERENCE_JSON = os.path.join(PREDICT_DIR, "reference.json")
    ANSWER_JSON = os.path.join(PREDICT_DIR, "answer.json")

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_file", type=str, default=REFERENCE_JSON)
    parser.add_argument("--prediction_file", type=str, default=ANSWER_JSON)
    parser.add_argument("--save_path", type=str, default="eval_res")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--model_name", type=str, default="glm-4.7")
    parser.add_argument("--api_key", type=str, default="d808ca603c19493aa31ce03251c763c5.Z75lBbIBAKgAAnWo")
    parser.add_argument("--api_base_url", type=str, default=None)
    parser.add_argument("--retry_attempts", type=int, default=3)
    args = parser.parse_args()
    
    # Default API Key/Base URL handling if not provided
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    api_base_url = args.api_base_url or os.environ.get("OPENAI_BASE_URL")
    
    if not api_key:
        print("Error: API Key is required. Please provide --api_key or set OPENAI_API_KEY env var.")
        # Proceeding anyway usually crashes, but let class handle it
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print(f"Loading references from {args.reference_file}...")
    with open(args.reference_file, "r", encoding="utf-8") as f:
        references = json.load(f)
        
    print(f"Loading predictions from {args.prediction_file}...")
    with open(args.prediction_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
        
    # references and predictions are dicts: {"0": ["text..."], "1": ["text..."]}
    
    # Find common keys
    ref_keys = set(references.keys())
    pred_keys = set(predictions.keys())
    common_keys = sorted(list(ref_keys.intersection(pred_keys)), key=lambda x: int(x) if x.isdigit() else x)
    
    print(f"Found {len(common_keys)} items to evaluate.")
    
    batcher = GPTBatcher(
        api_key=api_key, 
        model_name=args.model_name, 
        num_workers=args.num_workers,
        retry_attempts=args.retry_attempts,
        api_base_url=api_base_url
    )
    
    prepared_messages = []
    ids_list = []
    
    for k in common_keys:
        ref_text = references[k][0] if isinstance(references[k], list) else references[k]
        pred_text = predictions[k][0] if isinstance(predictions[k], list) else predictions[k]
        
        message_data = {
            "reference": ref_text,
            "prediction": pred_text
        }
        
        chat_msgs = batcher.create_messages(message_data)
        prepared_messages.append(chat_msgs)
        ids_list.append(k)
        
    print("Starting batch evaluation...")
    results = batcher.handle_message_list(prepared_messages)
    
    all_scores = []
    final_output = {}
    
    for idx, (res_text, item_id) in enumerate(zip(results, ids_list)):
        score = parse_score(res_text)
        
        if score is not None:
            all_scores.append(score)
        else:
            print(f"Warning: Could not extract score for item {item_id}")
            
        # Save individual result to text file in save_path (as requested by original code structure)
        # item_id might be "0", "1"...
        with open(os.path.join(args.save_path, f"{item_id}.txt"), "w", encoding='utf-8') as f:
            if res_text:
                f.write(res_text)
            else:
                f.write("ERROR: No response")
                
        final_output[item_id] = {
            "response": res_text,
            "score": score
        }

    # Calculate average
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        print(f"Stage1_score: {avg_score:.4f}")
        
        with open(os.path.join(args.save_path, "all_score.txt"), "w", encoding='utf-8') as f:
            f.write(f"Stage1_score: {avg_score:.4f}\n")
            f.write(f"Count: {len(all_scores)}\n")
            
        # Also save a comprehensive JSON result
        with open(os.path.join(args.save_path, "evaluation_results.json"), "w", encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
            
    else:
        print("No valid scores extracted.")
