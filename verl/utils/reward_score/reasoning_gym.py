"""Reward scoring for Reasoning Gym datasets.

This module provides grading functionality for reasoning_gym datasets.
It extracts answers from model completions and scores them using the
dataset-specific scoring functions from reasoning_gym.
"""

import random
from typing import Any, Optional, Union
import json

from reasoning_gym.utils import extract_answer
from reasoning_gym.factory import create_dataset, DATASETS

def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    tag_name: str = "answer",
) -> float:
    assert isinstance(ground_truth, str), "Ground truth must be a string."
    extracted_answer = extract_answer(solution_str, tag_name=tag_name)
    metadata = json.loads(extra_info["metadata"])
    source_dataset = metadata["source_dataset"]
    dataset_cls, config_cls = DATASETS[source_dataset]
    dataset = dataset_cls(config=config_cls())
    entry = {
        "answer": ground_truth if ground_truth else None,
        "metadata": metadata
    }
    return dataset.score_answer(extracted_answer, entry)
