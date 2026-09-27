#!/usr/bin/env python3
"""Merge a sharded reinforcement-learning checkpoint into a loadable model.

The trainer shards the policy across GPUs and saves one file per rank, which
nothing else can load. This reassembles the shards into a single directory that
``transformers`` opens like any other checkpoint.

    python scripts/convert_checkpoint.py \
        -i experiments/grpo/results/<run>/global_step_500/actor \
        -o experiments/grpo/results/<run>/global_step_500_hf
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drmv3d.constants import BASE_MODEL

#: Rank files are named by the world size they were written with.
_SHARD_NAME = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt")


def find_world_size(directory: str) -> int:
    """Return how many ranks wrote this checkpoint.

    Args:
        directory: The trainer's ``actor`` directory.

    Returns:
        The world size recorded in the shard file names.

    Raises:
        FileNotFoundError: If the directory holds no shard files.
    """
    for name in sorted(os.listdir(directory)):
        match = _SHARD_NAME.match(name)
        if match and match.group(2) == "0":
            return int(match.group(1))
    raise FileNotFoundError(f"no model_world_size_*_rank_0.pt in {directory}")


def _merge(tensors: list, placement) -> "object":
    """Reassemble one parameter from its per-rank pieces.

    A replicated parameter is identical on every rank, so any one will do. A
    sharded one was split along a single dimension, so the pieces concatenate
    along it.
    """
    if placement.is_replicate():
        return tensors[0]
    if placement.is_shard():
        import torch

        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise ValueError(f"cannot merge a parameter placed as {placement}")


def _to_transformers_key(key: str) -> str:
    """Rewrite one parameter name from the trainer's layout to the model's."""
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    if key.startswith("model.visual."):
        return key[len("model."):]
    return key


def convert(input_dir: str, output_dir: str, base_model: str = BASE_MODEL) -> str:
    """Merge a sharded checkpoint and write it as a loadable model directory.

    Args:
        input_dir: The trainer's ``actor`` directory, holding one file per rank.
        output_dir: Directory to write. Created if needed.
        base_model: Model whose config and processor describe the architecture.
            Reinforcement learning changes only the weights.

    Returns:
        The directory written.
    """
    import torch
    from safetensors.torch import save_file
    from torch.distributed._tensor import DTensor
    from transformers import AutoConfig, AutoProcessor

    world_size = find_world_size(input_dir)
    print(f"🔍 {world_size} shards in {input_dir}")

    def shard_path(rank: int) -> str:
        return os.path.join(input_dir, f"model_world_size_{world_size}_rank_{rank}.pt")

    shards: list = [None] * world_size
    shards[0] = torch.load(shard_path(0), map_location="cpu", weights_only=False)

    # How each parameter was split is recorded on the tensors themselves.
    pivot = shards[0][sorted(shards[0])[0]]
    placements = {}
    if isinstance(pivot, DTensor):
        for key, value in shards[0].items():
            placements[key] = value.placements

    def load(rank: int) -> tuple[int, dict]:
        return rank, torch.load(shard_path(rank), map_location="cpu", weights_only=False)

    with ThreadPoolExecutor(max_workers=min(8, world_size)) as pool:
        for rank, shard in pool.map(load, range(1, world_size)):
            shards[rank] = shard

    print("🔍 Merging parameters")
    merged = {}
    for key in sorted(shards[0]):
        pieces = []
        for shard in shards:
            tensor = shard.pop(key, None)
            if tensor is None:
                continue
            local = tensor._local_tensor if isinstance(tensor, DTensor) else tensor
            pieces.append(local.bfloat16())
        if key in placements:
            merged[_to_transformers_key(key)] = _merge(pieces, placements[key][0])
        elif len(pieces) == 1:
            merged[_to_transformers_key(key)] = pieces[0]
        else:
            merged[_to_transformers_key(key)] = torch.cat(pieces, dim=0)
    del shards

    os.makedirs(output_dir, exist_ok=True)
    save_file(merged, os.path.join(output_dir, "model.safetensors"), metadata={"format": "pt"})
    AutoConfig.from_pretrained(base_model).save_pretrained(output_dir)
    AutoProcessor.from_pretrained(base_model).save_pretrained(output_dir)
    print(f"✅ Wrote {len(merged)} parameters to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", "-i", required=True, help="the trainer's actor directory")
    parser.add_argument("--output", "-o", required=True, help="directory to write")
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help="model whose config and processor to copy")
    args = parser.parse_args()
    convert(args.input, args.output, args.base_model)


if __name__ == "__main__":
    main()
