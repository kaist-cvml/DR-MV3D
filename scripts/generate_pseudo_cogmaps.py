#!/usr/bin/env python3
"""Replace each item's annotated cognitive map with one estimated from its images.

Sits between the scaffold and prompt stages of the main pipeline:

    python scripts/prepare_scaffold.py -i data/raw/MindCube_train.jsonl \
        -o data/scaffold/MindCube_train.jsonl

    python scripts/generate_pseudo_cogmaps.py \
        -i data/scaffold/MindCube_train.jsonl \
        -o data/scaffold/MindCube_train_pseudo.jsonl \
        --vggt-checkpoint modules/VGGT-1B/model.pt \
        --sam3-root modules/sam3 --image-root data

    python scripts/generate_prompts.py -i data/scaffold/MindCube_train_pseudo.jsonl \
        -o data/prompts/MindCube_train_drmv3d_pseudo.jsonl

The egocentric maps and the reasoning chain are rebuilt from the estimated map, so
run this before the prompt stage, not after. Needs a GPU and the two external
models; see docs/pseudo_cogmap.md.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drmv3d.pseudo.pipeline import add_pseudo_cogmap
from drmv3d.scaffold.egomap import build_multi_view_egomap
from drmv3d.scaffold.reasoning import build_reasoning_chain
from drmv3d.utils.io import load_jsonl, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", "-i", required=True, help="scaffolded JSONL file")
    parser.add_argument("--output", "-o", required=True, help="JSONL file to write")
    parser.add_argument("--vggt-checkpoint", required=True, help="path to the VGGT weights")
    parser.add_argument("--sam3-root", required=True, help="path to the SAM 3 source tree")
    parser.add_argument("--sam3-checkpoint", default="",
                        help="SAM 3 weights; empty fetches them from the model hub")
    parser.add_argument("--image-root", default="data", help="directory holding the images")
    parser.add_argument("--max-views", type=int, default=4, help="views to reconstruct from")
    parser.add_argument("--confidence-threshold", type=float, default=0.50,
                        help="minimum segmentation confidence")
    parser.add_argument("--shard", type=int, default=0, help="index of this shard")
    parser.add_argument("--num-shards", type=int, default=1, help="how many shards in total")
    parser.add_argument("--keep-debug", action="store_true",
                        help="also record the estimator's diagnostics on each item")
    parser.add_argument("--skip-failures", action="store_true",
                        help="drop items the estimator cannot build a map for, "
                             "instead of stopping")
    args = parser.parse_args()

    items = load_jsonl(args.input)[args.shard::args.num_shards]
    print(f"🔍 Estimating cognitive maps for {len(items)} items "
          f"(shard {args.shard}/{args.num_shards})")

    done, failed = [], 0
    for index, item in enumerate(items, start=1):
        try:
            add_pseudo_cogmap(
                item,
                vggt_checkpoint=args.vggt_checkpoint,
                sam3_root=args.sam3_root,
                image_root=args.image_root,
                sam3_checkpoint=args.sam3_checkpoint,
                max_views=args.max_views,
                confidence_threshold=args.confidence_threshold,
                keep_debug=args.keep_debug,
            )
            # The downstream stages read the map, so rebuild them from the new one.
            egomap, meta = build_multi_view_egomap(item)
            item["grounded_egomap"] = egomap
            item["reasoning_chain"] = build_reasoning_chain(item, meta)
            done.append(item)
        except Exception as error:
            if not args.skip_failures:
                raise
            failed += 1
            print(f"⚠️  {item.get('id', '?')}: {type(error).__name__}: {error}")
        if index % 25 == 0 or index == len(items):
            print(f"   {index}/{len(items)} items, {failed} failed", flush=True)

    save_jsonl(done, args.output)
    print(f"✅ Wrote {len(done)} items to {args.output}" + (f" ({failed} failed)" if failed else ""))


if __name__ == "__main__":
    main()
