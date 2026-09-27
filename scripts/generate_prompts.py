#!/usr/bin/env python3
"""Stage 2: turn scaffolded items into prompts and training targets.

Reads a scaffolded file and writes the same items with ``input_prompt`` and
``grounded_output`` added.

    python scripts/generate_prompts.py \
        --input data/scaffold/MindCube_train.jsonl \
        --output data/prompts/MindCube_train_drmv3d.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drmv3d.prompts import add_prompt
from drmv3d.utils.io import load_jsonl, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", "-i", required=True, help="scaffolded JSONL file")
    parser.add_argument("--output", "-o", required=True, help="prompt JSONL file to write")
    args = parser.parse_args()

    items = load_jsonl(args.input)
    print(f"🔍 Building prompts for {len(items)} items from {args.input}")
    prompts = [add_prompt(item) for item in items]
    save_jsonl(prompts, args.output)
    print(f"✅ Wrote {len(prompts)} items to {args.output}")


if __name__ == "__main__":
    main()
