import json
import os.path
import time
import re
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

print_lock = threading.Lock()
file_counter_lock = threading.Lock()
error_counter_lock = threading.Lock()

processed_count = 0
error_count = 0
total_files = 0

def build_stage1_prompt(function_code: str) -> str:
    return f"""
        You are a static program analysis expert. Extract semantic information about the functional behavior and correctness-related conditions of the following C/C++ function.
        Rules:
        1. Do NOT classify or label the function as vulnerable or safe. Do NOT use security-related terms such as "Vulnerable", "Unsafe", "Bug", "Exploit", or any vulnerability names.
        2. Do NOT speculate about external context or environment beyond what can be inferred from the function itself, its comments, and the surrounding code.
        3. Focus on operational semantics, constraints, and resource-handling behaviors.
        4. Perform the analysis in three steps:
           (a) Identify constraints that are explicitly enforced in the code (e.g., conditional checks, guards).
           (b) Infer expected constraints based on the function’s role, its comments, and control-flow intent.
           (c) Compare (a) and (b), and identify expected constraints that are not explicitly enforced in the code.
        5. Prioritize constraints related to device state, operating mode, protocol state, permissions, and execution preconditions.
        6. When function names or comments imply required operating conditions (e.g., permission checks, receive/send capability, validation, or state transitions), infer the corresponding preconditions and verify whether they are explicitly enforced.
        7. When describing missing constraints, express them as concrete guard conditions or preconditions. Do not evaluate their correctness.
        Output JSON format:
        {
          "function_role": "...",
          "parameter_flows": ["..."],
          "resource_lifetime": ["..."],
          "api_protocols": ["..."],
          "error_handling_logic": ["..."],
          "state_transitions": ["..."],
          "path_constraints": ["..."],
          "implicit_assumptions": ["..."],
          "absent_constraints": ["..."]
        }
        Function code:
        ```c
        {function_code}
        """.strip()

api_key = 'sk-'
api_base = 'https://api.deepseek.com/'
client = OpenAI(api_key=api_key, base_url=api_base)

def call_stage1_llm(function_code: str, model: str = "deepseek-chat"):
    prompt = build_stage1_prompt(function_code)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a static program analysis expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0, 
        max_tokens=1200, 
        top_p=1.0,
        stream=False
    )

    raw_output = response.choices[0].message.content
    return raw_output

def parse_llm_json(output: str) -> dict:
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        start = output.find("{")
        end = output.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(output[start:end + 1])
            except:
                return None
        else:
            return None

def semantic_json_to_text(semantic_json: dict, output_path: str) -> str:
    text = (
        f"Function role: {semantic_json.get('function_role', '')}\n"
        f"Parameter flows include: {'; '.join(semantic_json.get('parameter_flows', []))}.\n"
        f"Resource lifetime management includes: {'; '.join(semantic_json.get('resource_lifetime', []))}.\n"
        f"API protocols include: {'; '.join(semantic_json.get('api_protocols', []))}.\n"
        f"Error handling logic includes: {'; '.join(semantic_json.get('error_handling_logic', []))}.\n"
        f"State transitions include: {'; '.join(semantic_json.get('state_transitions', []))}.\n"
        f"Path constraints include: {'; '.join(semantic_json.get('path_constraints', []))}.\n"
        f"Implicit assumptions include: {'; '.join(semantic_json.get('implicit_assumptions', []))}."
        f"Absent constraints include: {'; '.join(semantic_json.get('absent_constraints', []))}."
    ).replace('..', '.').replace('.;', ';')
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def save_semantic_json(semantic_json: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(semantic_json, f, indent=2, ensure_ascii=False)

def load_error_file(bad_file_path):
    with open(bad_file_path, "r", encoding="utf-8") as f:
        error_files = json.load(f)
    return error_files

def load_use_file(use_file_path):
    use_file_ids = []
    with open(use_file_path, "r", encoding="utf-8") as f:
        file = json.load(f)
    use_file_ids.extend(file['train_data_ids'])
    use_file_ids.extend(file['valid_data_ids'])
    use_file_ids.extend(file['test_data_ids'])
    return use_file_ids

def load_success_file(success_file_path):
    success_file_ids = []
    for file in os.listdir(success_file_path):
        if file.endswith(".json"):
            success_file_ids.append(file.replace(".json", '.c'))
    return success_file_ids

def check_file(llm_out_path, llm_text_path):
    os.makedirs(llm_text_path, exist_ok=True)

    json_files = [f for f in os.listdir(llm_out_path) if f.endswith('.json')]

    missing_files_count = 0
    processed_files_count = 0

    for json_file in json_files:
        file_id = json_file.replace('.json', '')
        text_file = file_id + '.txt'

        text_file_path = os.path.join(llm_text_path, text_file)
        if not os.path.exists(text_file_path):
            missing_files_count += 1
            with print_lock:
                print(f"发现缺失的文本文件: {text_file}")

            json_file_path = os.path.join(llm_out_path, json_file)
            with open(json_file_path, 'r', encoding='utf-8') as f:
                semantic_json = json.load(f)

            semantic_json_to_text(semantic_json, text_file_path)
            processed_files_count += 1
            with print_lock:
                print(f"已为 {json_file} 生成对应的文本文件 {text_file}")
    with print_lock:
        print(f"检查完成！共发现 {missing_files_count} 个缺失的文本文件，已处理 {processed_files_count} 个文件。")

def process_single_file(file_info):
    global processed_count, error_count

    file, idx, raw_code_path, llm_out_path, llm_text_path = file_info
    file_id = file.replace('.c', '')

    try:
        start_time = time.time()

        with print_lock:
            print(f"线程 {threading.current_thread().name} 正在处理第 {idx + 1} 个文件: {file}")

        with open(os.path.join(raw_code_path, file), "r", encoding="utf-8") as f:
            function_code = f.read()
            cleaned = function_code.strip()
            cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
            lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
            cleaned = '\n'.join(lines)

        llm_output = call_stage1_llm(cleaned)
        semantic_json = parse_llm_json(llm_output)

        end_time = time.time()
        elapsed_time = end_time - start_time

        if semantic_json:
            save_semantic_json(semantic_json, os.path.join(llm_out_path, file_id + '.json'))
            semantic_json_to_text(semantic_json, os.path.join(llm_text_path, file_id + '.txt'))

            with print_lock:
                print(f"✓ 线程 {threading.current_thread().name} 完成: {file} (耗时: {elapsed_time:.2f}秒)")
        else:
            with error_counter_lock:
                error_count += 1
            with print_lock:
                print(f"✗ 线程 {threading.current_thread().name} 解析失败: {file}")

        with file_counter_lock:
            processed_count += 1
            remaining = total_files - processed_count
            with print_lock:
                print(
                    f"进度: {processed_count}/{total_files} (成功率: {((processed_count - error_count) / processed_count * 100):.1f}%) 剩余: {remaining}")
        return True
    except Exception as e:
        with error_counter_lock:
            error_count += 1
        with print_lock:
            print(f"✗ 线程 {threading.current_thread().name} 处理文件 {file} 时发生错误: {str(e)}")
        return False

def process_files_multithreaded(files_to_process, raw_code_path, llm_out_path, llm_text_path, max_workers=5):
    global total_files
    total_files = len(files_to_process)

    file_tasks = [(file, idx, raw_code_path, llm_out_path, llm_text_path)
                  for idx, file in enumerate(files_to_process)]

    print(f"开始多线程处理 {total_files} 个文件，使用 {max_workers} 个线程...")

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Worker") as executor:
        future_to_file = {executor.submit(process_single_file, task): task
                          for task in file_tasks}

        completed = 0
        for future in as_completed(future_to_file):
            completed += 1
            file_info = future_to_file[future]
            try:
                future.result()
            except Exception as e:
                with print_lock:
                    print(f"任务异常: {file_info[0]}, 错误: {e}")

    print(f"\n处理完成！")
    print(f"总计: {total_files} 个文件")
    print(f"成功: {processed_count - error_count} 个文件")
    print(f"失败: {error_count} 个文件")
    print(f"成功率: {((processed_count - error_count) / total_files * 100):.1f}%")

if __name__ == "__main__":
    base_path = 'D:\\app_data\\ample_data\\'
    llm_out_path = os.path.join(base_path, 'llm_v2', 'reveal_llm_origin_out')
    llm_text_path = os.path.join(base_path, 'llm_v2', 'reveal_llm_origin_out_text')
    os.makedirs(llm_out_path, exist_ok=True)
    os.makedirs(llm_text_path, exist_ok=True)
    raw_code_path = os.path.join(base_path, 'reveal_raw_code')

    success_file_ids = load_success_file(llm_out_path)

    files_to_process = [f for f in os.listdir(raw_code_path)
                        if f.endswith(".c") and f not in success_file_ids]
    random.shuffle(files_to_process)

    print("总文件数:%d, 已成功数:%d, 剩余数:%d" %
          (len(os.listdir(raw_code_path)), len(success_file_ids), len(files_to_process)))

    if files_to_process:
        max_workers = 5
        process_files_multithreaded(files_to_process, raw_code_path, llm_out_path, llm_text_path, max_workers)
    else:
        print("没有需要处理的文件。")
    check_file(llm_out_path, llm_text_path)