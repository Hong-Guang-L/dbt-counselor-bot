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
print("转换为LLaMA-Factory训练格式...")
print("=" * 60)

input_file = os.path.join(datasets_dir, "CBT_DBT空巢老人数据集_合并版.json")
data = load_json(input_file)

alpaca_format = []
sharegpt_format = []

system_prompt = """你是一位专业的心理咨询师，专门为空巢老人提供情感支持。请根据用户的问题，运用认知行为疗法(CBT)或辩证行为疗法(DBT)的技巧来帮助他们。

CBT核心技巧：认知重构、证据检验、行为激活、问题解决、概率估算、成本收益分析
DBT核心技巧：正念观察、痛苦耐受、情绪调节、人际效能

请用温暖、耐心、尊重的语气，使用尊称（如：大爷、阿姨）与来访者交流。"""

for item in data["conversations"]:
    conv = item["conversation"]
    
    if len(conv) < 2:
        continue
    
    user_msg = conv[0]["content"]
    
    assistant_msgs = []
    for i in range(1, len(conv)):
        if conv[i]["role"] == "assistant":
            assistant_msgs.append(conv[i]["content"])
    
    if not assistant_msgs:
        continue
    
    instruction = user_msg
    output = "\n".join(assistant_msgs)
    
    therapy_type = item.get("therapy_type", "CBT")
    scenario = item.get("scenario", "")
    techniques = item.get("techniques", [])
    
    alpaca_item = {
        "instruction": instruction,
        "input": "",
        "output": output,
        "system": system_prompt,
        "history": []
    }
    alpaca_format.append(alpaca_item)
    
    sharegpt_conv = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]
    
    for i in range(1, len(conv)):
        sharegpt_conv.append({
            "role": conv[i]["role"],
            "content": conv[i]["content"]
        })
    
    sharegpt_item = {
        "conversations": sharegpt_conv
    }
    sharegpt_format.append(sharegpt_item)

print(f"转换完成:")
print(f"  Alpaca格式: {len(alpaca_format)} 条")
print(f"  ShareGPT格式: {len(sharegpt_format)} 条")

alpaca_output = os.path.join(datasets_dir, "训练数据_alpaca格式.json")
sharegpt_output = os.path.join(datasets_dir, "训练数据_sharegpt格式.json")

save_json(alpaca_format, alpaca_output)
print(f"\n已保存: 训练数据_alpaca格式.json")

save_json(sharegpt_format, sharegpt_output)
print(f"已保存: 训练数据_sharegpt格式.json")

print("\n" + "=" * 60)
print("LLaMA-Factory训练格式转换完成！")
print("=" * 60)
print("\n使用说明:")
print("1. Alpaca格式: 适用于标准微调")
print("2. ShareGPT格式: 适用于多轮对话微调")
print("\n在LLaMA-Factory的data/dataset_info.json中添加:")
print("""
{
  "empty_nest_counseling": {
    "file_name": "训练数据_sharegpt格式.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
""")
