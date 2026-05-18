#!/usr/bin/env python3

import argparse
import os

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--filename", default=None)
    parser.add_argument("--out-parquet", required=True)
    args = parser.parse_args()

    if args.filename:
        downloaded = hf_hub_download(repo_id=args.repo, filename=args.filename, repo_type="dataset")
        dataset = load_dataset("parquet", data_files=downloaded)["train"]
    else:
        dataset = load_dataset(args.repo, args.subset, split=args.split)

    os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
    dataset.to_parquet(args.out_parquet)


if __name__ == "__main__":
    main()
