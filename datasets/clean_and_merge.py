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
print("处理 PsyDTCorpus 数据 - 提取通用心理咨询技巧")
print("=" * 60)

psyt_file = os.path.join(datasets_dir, "PsyDTCorpus_train_mulit_turn_packing.json")
psyt_data = load_json(psyt_file)

print(f"原始数据: {len(psyt_data)} 条")

exclude_keywords = [
    "男朋友", "女朋友", "恋爱", "约会", "分手", "出轨", "婚外情",
    "怀孕", "流产", "堕胎", "性", "做爱", "避孕", "小三", "劈腿",
    "做爱", "性爱", "性行为"
]

cleaned_psyt = []

for item in psyt_data:
    messages = item.get("messages", [])
    
    if not messages:
        continue
    
    user_content = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_content += msg.get("content", "")
    
    has_exclude = any(kw in user_content for kw in exclude_keywords)
    
    if has_exclude:
        continue
    
    new_system = """你是一位专业的心理咨询师，运用理情行为疗法(REBT)为来访者提供帮助。

核心技巧：
1. 识别非理性信念：帮助来访者发现不合理的想法
2. 与非理性信念辩论：质疑不合理的思维
3. 建立理性信念：引导来访者形成合理的认知
4. 行为改变：鼓励来访者采取积极行动

请用温暖、耐心的语气与来访者交流。"""
    
    new_messages = [{"role": "system", "content": new_system}]
    new_messages.extend([m for m in messages if m.get("role") in ["user", "assistant"]])
    
    cleaned_psyt.append({"conversations": new_messages})

print(f"清洗后: {len(cleaned_psyt)} 条")

psyt_output = os.path.join(datasets_dir, "PsyDTCorpus_通用技巧版.json")
save_json(cleaned_psyt, psyt_output)
print(f"已保存: PsyDTCorpus_通用技巧版.json")

print("\n" + "=" * 60)
print("合并所有训练数据...")
print("=" * 60)

all_data = []

cbt_dbt_file = os.path.join(datasets_dir, "训练数据_sharegpt格式.json")
if os.path.exists(cbt_dbt_file):
    cbt_dbt_data = load_json(cbt_dbt_file)
    all_data.extend(cbt_dbt_data)
    print(f"CBT/DBT空巢老人: {len(cbt_dbt_data)} 条")

mixed_file = os.path.join(datasets_dir, "老年人日常对话_训练格式.json")
if os.path.exists(mixed_file):
    mixed_data = load_json(mixed_file)
    all_data.extend(mixed_data)
    print(f"老年人日常对话: {len(mixed_data)} 条")

all_data.extend(cleaned_psyt)
print(f"PsyDTCorpus通用技巧版: {len(cleaned_psyt)} 条")

print(f"\n总计: {len(all_data)} 条")

combined_file = os.path.join(datasets_dir, "训练数据_最终合并版.json")
save_json(all_data, combined_file)
print(f"已保存: 训练数据_最终合并版.json")

print("\n" + "=" * 60)
print("数据清洗和合并完成！")
print("=" * 60)
