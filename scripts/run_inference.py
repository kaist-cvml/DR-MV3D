#!/usr/bin/env python3
"""Generate responses for a prompt file with a trained checkpoint.

    python scripts/run_inference.py \
        --model-path jihochoi/DR-MV3D-R-GRPO \
        --input data/prompts/MindCube_tinybench_drmv3d.jsonl \
        --output results/tinybench_responses.jsonl \
        --image-root data --batch-size 4

To split the file across GPUs, run one process per GPU with a different
``--shard`` and the same ``--num-shards``, then concatenate the outputs.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drmv3d.constants import INFERENCE_MAX_NEW_TOKENS
from drmv3d.inference.qwen import DTYPES, run_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", required=True, help="checkpoint directory or HF repo id")
    parser.add_argument("--input", "-i", required=True, help="prompt JSONL file")
    parser.add_argument("--output", "-o", required=True, help="response JSONL file to write")
    parser.add_argument("--image-root", default="data", help="directory holding the benchmark images")
    parser.add_argument("--batch-size", type=int, default=4, help="items per generation call")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--attn-implementation", default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--max-new-tokens", type=int, default=INFERENCE_MAX_NEW_TOKENS)
    parser.add_argument("--shard", type=int, default=0, help="index of this shard")
    parser.add_argument("--num-shards", type=int, default=1, help="how many shards in total")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many items")
    args = parser.parse_args()

    print(f"🔍 {args.model_path} → {args.output} (shard {args.shard}/{args.num_shards})")
    run_file(
        model_path=args.model_path,
        input_file=args.input,
        output_file=args.output,
        image_root=args.image_root,
        batch_size=args.batch_size,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        shard=args.shard,
        num_shards=args.num_shards,
        limit=args.limit,
    )
    print(f"✅ Wrote {args.output}")


if __name__ == "__main__":
    main()
