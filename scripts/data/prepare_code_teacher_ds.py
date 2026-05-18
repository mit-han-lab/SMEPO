#!/usr/bin/env python3

import argparse
import os
import sys

import datasets

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from masking.semantic_masking import mask_code_teacher_trace


INSTRUCTION_FOLLOWING = (
    "You are an expert Python programmer.\n"
    "Write correct, concise code.\n"
    "Return ONLY Python code in a single ```python``` block.\n"
    "Do NOT include explanations, markdown outside the code block, or extra text.\n"
)
CODE_DATA_SOURCE = "smepo-code"


def build_prompt_from_question(question: str) -> list[dict]:
    return [
        {"role": "system", "content": INSTRUCTION_FOLLOWING},
        {"role": "user", "content": question},
    ]


def load_rows(path: str):
    if path.endswith(".parquet"):
        ds = datasets.load_dataset("parquet", data_files=path)["train"]
        for row in ds:
            yield row
        return
    raise ValueError(f"Unsupported input format: {path}")


def normalize_row(obj: dict, index: int, mask_teacher_ds: bool) -> dict:
    extra_info = obj.get("extra_info")
    if not isinstance(extra_info, dict):
        extra_info = {}

    teacher_ds = str(extra_info.get("teacher_ds", ""))
    if mask_teacher_ds:
        teacher_ds = mask_code_teacher_trace(teacher_ds)

    prompt = obj.get("prompt", [])
    if not prompt:
        prompt = build_prompt_from_question(str(obj.get("question", "")))

    normalized = {
        "data_source": obj.get("data_source", CODE_DATA_SOURCE),
        "prompt": prompt,
        "ability": obj.get("ability", "code"),
        "reward_model": obj.get("reward_model", {}),
        "extra_info": {
            "index": index,
            "teacher_ds": teacher_ds,
        },
    }
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", required=True)
    parser.add_argument("--out-parquet", required=True)
    parser.add_argument("--mask-teacher-ds", action="store_true")
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    rows = []
    for idx, obj in enumerate(load_rows(args.in_file)):
        if args.limit > 0 and idx >= args.limit:
            break
        rows.append(normalize_row(obj, len(rows), args.mask_teacher_ds))

    os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(args.out_parquet)


if __name__ == "__main__":
    main()
