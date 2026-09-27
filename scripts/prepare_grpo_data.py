#!/usr/bin/env python3
"""Build the sample table the reinforcement-learning trainer reads.

    python scripts/prepare_grpo_data.py \
        --input data/prompts/MindCube_train_drmv3d.jsonl \
        --output data/grpo/train.parquet \
        --image-root data --max-prompt-length 8192

Prompts longer than the budget are dropped rather than truncated; the count is
reported.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drmv3d.constants import BASE_MODEL
from drmv3d.grpo.data import build_rl_samples
from drmv3d.utils.io import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", "-i", required=True, help="prompt JSONL file")
    parser.add_argument("--output", "-o", required=True, help="parquet file to write")
    parser.add_argument("--image-root", default="data", help="directory holding the images")
    parser.add_argument("--max-prompt-length", type=int, default=8192,
                        help="token budget per prompt; 0 keeps every row")
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help="model whose processor measures prompt length")
    args = parser.parse_args()

    items = load_jsonl(args.input)
    budget = args.max_prompt_length or None

    processor = None
    if budget:
        from transformers import AutoProcessor
        print(f"🔍 Loading {args.base_model} processor to measure prompt length")
        processor = AutoProcessor.from_pretrained(args.base_model)

    print(f"🔍 Building rows for {len(items)} items")
    rows, dropped = build_rl_samples(items, args.image_root, budget, processor)
    if dropped:
        print(f"⚠️  Dropped {dropped} items whose prompt exceeds {budget} tokens")

    import datasets
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(args.output)
    print(f"✅ Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
