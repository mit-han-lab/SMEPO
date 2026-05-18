#!/usr/bin/env python3

import argparse
import os
import sys

import datasets

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from masking.semantic_masking import mask_math_teacher_trace


DATA_SOURCE = "smepo-math"

INSTRUCTION_FOLLOWING = (
    "You are an expert math assistant. \n"
    "Solve the problem step by step IN ENGLISH. \n"
    "Finally output ONLY ONE line with the final answer in the form \\boxed{...}. \n"
    "Do not include any other boxed expressions after it. \n"
)


def make_row(question: str, answer: str, teacher_ds: str, index: int) -> dict:
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": INSTRUCTION_FOLLOWING},
            {"role": "user", "content": question},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": index,
            "question": question,
            "answer": answer,
            "teacher_ds": teacher_ds,
        },
    }


def extract_answer(obj: dict) -> str:
    reward_model = obj.get("reward_model", {})
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is not None:
            return str(ground_truth)
    return str(obj.get("answer", ""))


def load_rows(path: str):
    if path.endswith(".parquet"):
        ds = datasets.load_dataset("parquet", data_files=path)["train"]
        for row in ds:
            yield row
        return
    raise ValueError(f"Unsupported input format: {path}")


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
        teacher_ds = str(obj.get("teacher_ds", ""))
        if args.mask_teacher_ds:
            teacher_ds = mask_math_teacher_trace(teacher_ds)
        rows.append(
            make_row(
                question=str(obj.get("question", "")),
                answer=extract_answer(obj),
                teacher_ds=teacher_ds,
                index=len(rows),
            )
        )

    os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(args.out_parquet)


if __name__ == "__main__":
    main()
