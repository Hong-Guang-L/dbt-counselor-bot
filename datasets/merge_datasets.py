import json
import os

datasets_dir = r"c:\Users\24048\PycharmProjects\AI_Model\datasets"

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def normalize_cbt_conversation(item):
    return {
        "id": item.get("id", 0),
        "therapy_type": "CBT",
        "scenario": item.get("scenario", ""),
        "techniques": item.get("techniques", []),
        "conversation": item.get("conversation", [])
    }

def normalize_dbt_conversation(item):
    return {
        "id": item.get("id", 0),
        "therapy_type": "DBT",
        "module": item.get("module", ""),
        "scenario": item.get("scenario", ""),
        "techniques": item.get("skills", []),
        "conversation": item.get("conversation", [])
    }

print("=" * 60)
print("开始合并数据集...")
print("=" * 60)

cbt_files = [
    "CBT空巢老人数据集_扩展版.json",
    "CBT空巢老人数据集_扩展版_第2批.json",
    "CBT空巢老人数据集_扩展版_第3批.json",
    "CBT空巢老人数据集_扩展版_第4批.json",
    "CBT空巢老人数据集_扩展版_第5批.json"
]

dbt_files = [
    "DBT空巢老人数据集_扩展版.json",
    "DBT空巢老人数据集_扩展版_第2批.json",
    "DBT空巢老人数据集_扩展版_第3批.json",
    "DBT空巢老人数据集_扩展版_第4批.json",
    "DBT空巢老人数据集_扩展版_第5批.json"
]

cbt_all = []
for filename in cbt_files:
    filepath = os.path.join(datasets_dir, filename)
    if os.path.exists(filepath):
        data = load_json(filepath)
        conversations = data.get("conversations", [])
        for item in conversations:
            normalized = normalize_cbt_conversation(item)
            cbt_all.append(normalized)
        print(f"已加载: {filename} ({len(conversations)} 条)")
    else:
        print(f"文件不存在: {filename}")

dbt_all = []
for filename in dbt_files:
    filepath = os.path.join(datasets_dir, filename)
    if os.path.exists(filepath):
        data = load_json(filepath)
        conversations = data.get("conversations", [])
        for item in conversations:
            normalized = normalize_dbt_conversation(item)
            dbt_all.append(normalized)
        print(f"已加载: {filename} ({len(conversations)} 条)")
    else:
        print(f"文件不存在: {filename}")

print("\n" + "=" * 60)
print("合并统计:")
print(f"  CBT 数据集: {len(cbt_all)} 条")
print(f"  DBT 数据集: {len(dbt_all)} 条")
print(f"  总计: {len(cbt_all) + len(dbt_all)} 条")
print("=" * 60)

cbt_output = {
    "dataset_name": "CBT空巢老人心理咨询数据集（合并版）",
    "therapy_type": "认知行为疗法(CBT)",
    "total_samples": len(cbt_all),
    "format": "统一格式",
    "conversations": cbt_all
}

dbt_output = {
    "dataset_name": "DBT空巢老人心理咨询数据集（合并版）",
    "therapy_type": "辩证行为疗法(DBT)",
    "total_samples": len(dbt_all),
    "format": "统一格式",
    "conversations": dbt_all
}

cbt_output_path = os.path.join(datasets_dir, "CBT空巢老人数据集_合并版.json")
dbt_output_path = os.path.join(datasets_dir, "DBT空巢老人数据集_合并版.json")

save_json(cbt_output, cbt_output_path)
print(f"\n已保存: CBT空巢老人数据集_合并版.json")

save_json(dbt_output, dbt_output_path)
print(f"已保存: DBT空巢老人数据集_合并版.json")

all_data = cbt_all + dbt_all
for i, item in enumerate(all_data):
    item["id"] = i + 1

combined_output = {
    "dataset_name": "CBT+DBT空巢老人心理咨询数据集（合并版）",
    "therapy_types": ["认知行为疗法(CBT)", "辩证行为疗法(DBT)"],
    "total_samples": len(all_data),
    "cbt_samples": len(cbt_all),
    "dbt_samples": len(dbt_all),
    "format": "统一格式",
    "conversations": all_data
}

combined_output_path = os.path.join(datasets_dir, "CBT_DBT空巢老人数据集_合并版.json")
save_json(combined_output, combined_output_path)
print(f"已保存: CBT_DBT空巢老人数据集_合并版.json")

print("\n" + "=" * 60)
print("数据集合并完成！")
print("=" * 60)
