#!/usr/bin/env python3
"""Stage 3: convert prompts into the conversation format the trainer reads.

Reads a prompt file and writes the JSON list of conversations that
``scripts/train_sft.sh`` consumes.

    python scripts/convert_to_sft.py \
        --input data/prompts/MindCube_train_drmv3d.jsonl \
        --output data/sft/MindCube_train_drmv3d_qwen_sft.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drmv3d.training.sft_data import to_conversations
from drmv3d.utils.io import load_jsonl, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", "-i", required=True, help="prompt JSONL file")
    parser.add_argument("--output", "-o", required=True, help="training JSON file to write")
    args = parser.parse_args()

    items = load_jsonl(args.input)
    print(f"🔍 Converting {len(items)} items from {args.input}")
    conversations = to_conversations(items)
    save_json(conversations, args.output)
    print(f"✅ Wrote {len(conversations)} conversations to {args.output}")


if __name__ == "__main__":
    main()
