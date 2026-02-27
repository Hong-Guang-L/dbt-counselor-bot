import json
import os

datasets_dir = r"c:\Users\24048\PycharmProjects\AI_Model\datasets"

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print("=" * 60)
print("转换 mixed_partially_shuffled 为训练格式...")
print("=" * 60)

mixed_file = os.path.join(datasets_dir, "mixed_partially_shuffled.json")
mixed_data = load_json(mixed_file)

print(f"原始数据: {len(mixed_data)} 条")

system_prompt = """你是一位温暖、耐心的智能助手，专门为老年人提供服务。请用通俗易懂、亲切友好的语言回答问题，像和老朋友聊天一样自然。"""

sharegpt_format = []

for item in mixed_data:
    instruction = item.get("instruction", "")
    output = item.get("output", "")
    
    if not instruction or not output:
        continue
    
    conv = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output}
    ]
    
    sharegpt_format.append({"conversations": conv})

print(f"转换完成: {len(sharegpt_format)} 条")

output_file = os.path.join(datasets_dir, "老年人日常对话_训练格式.json")
save_json(sharegpt_format, output_file)
print(f"已保存: 老年人日常对话_训练格式.json")

print("\n" + "=" * 60)
print("转换 PsyDTCorpus 为训练格式...")
print("=" * 60)

psyt_file = os.path.join(datasets_dir, "PsyDTCorpus_train_mulit_turn_packing.json")
psyt_data = load_json(psyt_file)

print(f"原始数据: {len(psyt_data)} 条")

psyt_sharegpt = []

for item in psyt_data:
    messages = item.get("messages", [])
    
    if not messages:
        continue
    
    has_user = False
    has_assistant = False
    for msg in messages:
        if msg.get("role") == "user":
            has_user = True
        if msg.get("role") == "assistant":
            has_assistant = True
    
    if not (has_user and has_assistant):
        continue
    
    psyt_sharegpt.append({"conversations": messages})

print(f"转换完成: {len(psyt_sharegpt)} 条")

psyt_output = os.path.join(datasets_dir, "PsyDTCorpus_训练格式.json")
save_json(psyt_sharegpt, psyt_output)
print(f"已保存: PsyDTCorpus_训练格式.json")

print("\n" + "=" * 60)
print("合并所有训练数据...")
print("=" * 60)

all_data = []

cbt_dbt_file = os.path.join(datasets_dir, "训练数据_sharegpt格式.json")
if os.path.exists(cbt_dbt_file):
    cbt_dbt_data = load_json(cbt_dbt_file)
    all_data.extend(cbt_dbt_data)
    print(f"CBT/DBT数据: {len(cbt_dbt_data)} 条")

all_data.extend(sharegpt_format)
print(f"老年人日常对话: {len(sharegpt_format)} 条")

all_data.extend(psyt_sharegpt)
print(f"PsyDTCorpus: {len(psyt_sharegpt)} 条")

print(f"\n总计: {len(all_data)} 条")

combined_file = os.path.join(datasets_dir, "训练数据_全部合并.json")
save_json(all_data, combined_file)
print(f"已保存: 训练数据_全部合并.json")

print("\n" + "=" * 60)
print("数据集处理完成！")
print("=" * 60)
print("\n生成的文件:")
print("  - 老年人日常对话_训练格式.json")
print("  - PsyDTCorpus_训练格式.json")
print("  - 训练数据_全部合并.json")
