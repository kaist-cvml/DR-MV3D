# Provenance: `qwen_vl_finetune`

The supervised fine-tuning stage of DR-MV3D uses the Qwen2.5-VL fine-tuning
trainer. Rather than depend on a clone the reader would have to find and pin, the
files that stage actually imports are vendored here, unchanged except for the
six edits recorded below.

## Upstream

| | |
|---|---|
| Vendored from | <https://github.com/QinengWang-Aiden/Qwen2.5-VL-MindCube> |
| Commit | `1de85895e4114ec8fa9cbf199693fdbd9b36490b` ("Update data_qwen.py", 2025-06-23) |
| Subdirectory | `qwen-vl-finetune/` |
| That repository is a fork of | <https://github.com/QwenLM/Qwen2.5-VL> (Alibaba Cloud, Qwen team) |
| License | Apache License 2.0, copied verbatim to `LICENSE` |

Upstream ships no filled-in copyright line, so attribution here is by repository.
`qwenvl/train/train_qwen.py` carries its own header crediting FastChat and
Stanford Alpaca, which is preserved.

## What is vendored

Only the files the training entry point imports, plus the DeepSpeed config it
loads. Nothing else from upstream is included.

| File | Lines | Why it is here |
|---|---|---|
| `qwenvl/train/train_qwen.py` | 178 | The entry point |
| `qwenvl/train/trainer.py` | 400 | `Trainer` subclass and the flattened-attention patch |
| `qwenvl/train/argument.py` | 38 | Argument dataclasses |
| `qwenvl/data/data_qwen.py` | 652 | Dataset and collator |
| `qwenvl/data/data_qwen_packed.py` | 624 | Imported unconditionally by the entry point |
| `qwenvl/data/__init__.py` | 63 | Dataset registry |
| `qwenvl/data/rope2d.py` | 424 | 2D rotary position indices |
| `scripts/zero3.json` | 27 | DeepSpeed ZeRO-3 config |

Line counts are of the upstream files. There is deliberately no `qwenvl/__init__.py`
or `qwenvl/train/__init__.py`: upstream relies on implicit namespace packages, and
the entry point puts this directory on `sys.path` itself.

## Changes

Six changes, each marked inline with a `DR-MV3D` comment except the import removal
in `rope2d.py`, which leaves nothing to mark; that file carries a notice in its
header instead. Apache-2.0 requires that modified files say so, and these are all
of them.

1. **`qwenvl/data/data_qwen.py` — the image-token expansion.** Upstream leaves the
   multiplication at what is now line 112 commented out, so one `<image>` expands
   to a single `<|image_pad|>` token while `qwenvl/data/rope2d.py` advances the
   position index by the image's full patch grid. The two then disagree and the
   loader raises `ValueError: 151655 is not in list`. The sibling
   `qwenvl/data/data_qwen_packed.py` keeps the multiplication; this restores it.

2. **`qwenvl/train/train_qwen.py` — `low_cpu_mem_usage=True` at model load.**
   Without it every rank materialises a full copy of the model before ZeRO-3
   shards it, which exhausts host memory at four ranks and above.

3. **`qwenvl/data/data_qwen.py` — `rank0_print` asks `torch.distributed` for the
   rank.** Upstream tests a module-level `local_rank` that nothing in that file
   assigns, so it is always `None` and the function prints nothing.

4. **`qwenvl/train/trainer.py` — `flash_attn` is imported where it is used.**
   Upstream imports it at module scope, so the trainer cannot be imported at all
   without flash-attn, even though only the flattened-attention path calls it.
   Training with `--data_flatten True`, which is what `configs/sft.sh` uses, still
   requires flash-attn.

5. **The video decoders are no longer imported at module scope.** DR-MV3D is
   image-only, so `decord` and `torchcodec` are never reached. In
   `qwenvl/data/data_qwen.py` and `qwenvl/data/data_qwen_packed.py` the imports
   moved into the two functions that use them; in `qwenvl/data/rope2d.py` nothing
   used the import, so it was removed. Either way an image-only install does not
   need those packages.

6. **`qwenvl/data/__init__.py` — datasets are read from `DRMV3D_SFT_REGISTRY`.**
   Upstream expects each dataset's paths to be written into this file, which means
   editing vendored third-party source in place and baking absolute paths into it.
   The added `_load_external_registry` reads a JSON manifest named by that
   environment variable and resolves relative paths against `DRMV3D_DATA_ROOT`, or
   against the manifest's own directory. With the variable unset the file behaves
   exactly as upstream.

Nothing else differs.

## Two notes for anyone editing this

The loader reads each record's image paths from `image`, singular, at
`qwenvl/data/data_qwen.py:394`. Under any other key `"image" in sources[0]` is
false, the text-only branch runs and `pixel_values` is never set, without raising
or warning. `drmv3d/training/sft_data.py` writes `image` for that reason.

`qwenvl/data/data_qwen_packed.py` is imported by the entry point whether or not
`--data_packing` is set, so it cannot be dropped without editing the entry point.
No configuration in this repository uses it.
