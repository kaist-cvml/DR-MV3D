#!/usr/bin/env python3
"""Score response files: accuracy of the chosen option, overall and per setting.

    python scripts/run_evaluation.py results/tinybench_responses.jsonl

Several files are read as one set, which is how a sharded inference run is scored.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drmv3d.eval.accuracy import format_scores, score_responses
from drmv3d.utils.io import load_jsonl, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("responses", nargs="+", help="response JSONL file(s)")
    parser.add_argument("--output", "-o", help="write the scores as JSON here")
    args = parser.parse_args()

    items = [item for path in args.responses for item in load_jsonl(path)]
    print(f"🔍 Scoring {len(items)} responses from {len(args.responses)} file(s)")
    scores = score_responses(items)
    print(format_scores(scores))

    if args.output:
        save_json(scores, args.output)
        print(f"✅ Wrote {args.output}")
    else:
        print()
        print(json.dumps({"accuracy": round(scores["accuracy"] * 100, 2)}))


if __name__ == "__main__":
    main()
