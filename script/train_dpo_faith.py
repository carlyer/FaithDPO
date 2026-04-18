#!/usr/bin/env python3
"""
FaithDPO 数据格式转换脚本

将原始数据 (FaithDPO-Data/data.jsonl) 转换为 DPO 训练格式

V1: 每个样本 1 对 (chosen, rejected)
V2: 每个样本 3 对 (1 个 final_chosen × 3 个 reject) → 1095 条
"""

import json
import os
from pathlib import Path


def convert_to_dpo_format_v1(input_path, output_path):
    """V1: 每个样本 1 对 DPO 数据"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    dpo_data = []
    skipped = 0

    for line in lines:
        sample = json.loads(line.strip())

        context = sample.get("context", "")
        query = sample.get("query", "")
        prompt = context + "\n\nQ: " + query + "\nA:"

        chosen_list = [
            r["response"].strip()
            for r in sample.get("responses", [])
            if r.get("category") == "chosen" and r.get("response", "").strip()
        ]
        reject_list = [
            r["response"].strip()
            for r in sample.get("responses", [])
            if r.get("category") == "reject" and r.get("response", "").strip()
        ]

        if not chosen_list or not reject_list:
            skipped += 1
            continue

        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen_list[0],
            "rejected": reject_list[0],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"V1 转换完成: {len(dpo_data)} 条 DPO 数据 -> {output_path}")
    if skipped > 0:
        print(f"跳过: {skipped} 条")


def convert_to_dpo_format_v2(input_path, output_path):
    """V2: 每个样本 3 对 DPO 数据

    - chosen: is_final_selection=True 的响应（每样本 1 个）
    - rejected: category="reject" 的 3 个响应（每样本 3 个）

    总计: 365 样本 × 3 对 = 1095 条 DPO 数据
    """

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    dpo_data = []
    skipped = 0

    for line in lines:
        sample = json.loads(line.strip())

        context = sample.get("context", "")
        query = sample.get("query", "")
        prompt = context + "\n\nQ: " + query + "\nA:"

        # 找 final_chosen（is_final_selection=True 的唯一响应）
        final_chosen_list = [
            r["response"].strip()
            for r in sample.get("responses", [])
            if r.get("is_final_selection") is True and r.get("response", "").strip()
        ]

        # 收集所有 reject
        reject_list = [
            r["response"].strip()
            for r in sample.get("responses", [])
            if r.get("category") == "reject" and r.get("response", "").strip()
        ]

        if not final_chosen_list:
            skipped += 1
            print(f"跳过 sample_id={sample.get('sample_id')}: 缺少 final_chosen")
            continue

        if len(reject_list) < 3:
            skipped += 1
            print(f"跳过 sample_id={sample.get('sample_id')}: reject 不足 3 个")
            continue

        # 每个 final_chosen 与 3 个 reject 组成 3 对 DPO 数据
        final_chosen = final_chosen_list[0]
        for reject in reject_list[:3]:
            dpo_data.append({
                "prompt": prompt,
                "chosen": final_chosen,
                "rejected": reject,
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"V2 转换完成: {len(dpo_data)} 条 DPO 数据 -> {output_path}")
    if skipped > 0:
        print(f"跳过: {skipped} 条")


def main():
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "FaithDPO-Data" / "data.jsonl"

    # V1: 365 条（每个样本 1 对）
    v1_output = base_dir / "FaithDPO-Data" / "dpo_data_v1.jsonl"
    convert_to_dpo_format_v1(str(input_path), str(v1_output))

    # V2: 1095 条（每个样本 3 对）
    v2_output = base_dir / "FaithDPO-Data" / "dpo_data_v2.jsonl"
    convert_to_dpo_format_v2(str(input_path), str(v2_output))


if __name__ == "__main__":
    main()
