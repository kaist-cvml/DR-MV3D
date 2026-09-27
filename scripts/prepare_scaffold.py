#!/usr/bin/env python3
"""Stage 1: derive cognitive maps, egocentric maps and reasoning chains.

Reads the raw benchmark file and writes the same items with the scaffold fields
added.

    python scripts/prepare_scaffold.py \
        --input data/raw/MindCube_train.jsonl \
        --output data/scaffold/MindCube_train.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drmv3d.scaffold.pipeline import add_scaffold_to_all
from drmv3d.utils.io import load_jsonl, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", "-i", required=True, help="raw benchmark JSONL file")
    parser.add_argument("--output", "-o", required=True, help="scaffolded JSONL file to write")
    parser.add_argument("--quiet", "-q", action="store_true", help="suppress non-fatal warnings")
    args = parser.parse_args()

    items = load_jsonl(args.input)
    print(f"🔍 Scaffolding {len(items)} items from {args.input}")
    scaffolded = add_scaffold_to_all(items, quiet=args.quiet)
    save_jsonl(scaffolded, args.output)
    print(f"✅ Wrote {len(scaffolded)} items to {args.output}")


if __name__ == "__main__":
    main()
