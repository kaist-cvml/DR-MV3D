"""Greedy batched generation with a Qwen2.5-VL checkpoint.

Decoding is greedy, so a checkpoint and a prompt file determine the responses.
Batching is what makes a full evaluation practical, and it needs left padding:
the model continues from the end of each sequence, so the padding has to sit at
the front.
"""

import json
import os
from typing import Any

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ..constants import INFERENCE_MAX_NEW_TOKENS, INFERENCE_MAX_PIXELS
from ..utils.io import ensure_dir

#: Torch dtypes selectable from the command line.
DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _read_config(model_path: str) -> Any:
    """Read a checkpoint's config, accepting one written by a newer transformers.

    From version 4.52 onwards this model's config nests its language-model fields
    under ``text_config`` and spells the weight type ``dtype``. The pinned version
    reads them from the top level as ``torch_dtype`` and fails on the nested form.
    The top-level fields are still present in the newer layout, so dropping the
    nested copy and restoring the two names it moved is enough.

    Args:
        model_path: Local directory or Hugging Face repository id.

    Returns:
        A config object ``from_pretrained`` accepts.
    """
    local = os.path.join(model_path, "config.json")
    if os.path.exists(local):
        path = local
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_path, "config.json")

    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)

    nested = raw.pop("text_config", None)
    if isinstance(nested, dict):
        raw.setdefault("tie_word_embeddings", nested.get("tie_word_embeddings", True))
    if "torch_dtype" not in raw and "dtype" in raw:
        raw["torch_dtype"] = raw.pop("dtype")

    model_type = raw.pop("model_type")
    return AutoConfig.for_model(model_type, **raw)


def load_model(
    model_path: str,
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    device: str = "cuda",
    max_pixels: int = INFERENCE_MAX_PIXELS,
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """Load a checkpoint and its processor.

    Args:
        model_path: Local directory or Hugging Face repository id.
        dtype: One of the keys of :data:`DTYPES`.
        attn_implementation: ``sdpa`` needs no extra package; ``flash_attention_2``
            is faster where it is installed.
        device: Device to place the model on.
        max_pixels: Upper bound on image tokens per image.

    Returns:
        The model in eval mode, and the processor set up for batched generation.
    """
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}, expected one of {sorted(DTYPES)}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=_read_config(model_path),
        torch_dtype=DTYPES[dtype],
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    )
    model = model.eval().to(device)

    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
    # The model continues from the end of each sequence, so pad on the left.
    processor.tokenizer.padding_side = "left"
    return model, processor


def resolve_image_paths(images: list[str], image_root: str) -> list[str]:
    """Resolve an item's image paths against the image root.

    Args:
        images: Paths as the data file records them, relative to the image root.
        image_root: Directory the benchmark images were extracted into.

    Returns:
        Paths that exist on disk, in the item's own order.

    Raises:
        FileNotFoundError: If any image is missing, since dropping one silently
            would shift every later image onto the wrong placeholder.
    """
    resolved = []
    for path in images:
        full = path if os.path.isabs(path) else os.path.join(image_root, path)
        if not os.path.exists(full):
            raise FileNotFoundError(f"image not found: {full}")
        resolved.append(full)
    return resolved


def _build_chat_text(processor: AutoProcessor, prompt: str, image_count: int) -> str:
    """Render one item into the model's chat format.

    The image placeholders come first, in the item's order, so that the nth
    placeholder binds to the nth image passed to the processor.
    """
    content: list[dict[str, Any]] = [{"type": "image"} for _ in range(image_count)]
    content.append({"type": "text", "text": prompt})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


def generate_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    prompts: list[str],
    image_paths: list[list[str]],
    max_new_tokens: int = INFERENCE_MAX_NEW_TOKENS,
) -> list[str]:
    """Generate one response per prompt.

    Args:
        model: Loaded checkpoint.
        processor: Its processor, padding on the left.
        prompts: One prompt per item.
        image_paths: One list of resolved image paths per item.
        max_new_tokens: Generation budget.

    Returns:
        One response per prompt, in order.
    """
    texts = [
        _build_chat_text(processor, prompt, len(paths))
        for prompt, paths in zip(prompts, image_paths, strict=False)
    ]
    images = [Image.open(path).convert("RGB") for paths in image_paths for path in paths]

    inputs = processor(text=texts, images=images or None, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    prompt_length = inputs["input_ids"].shape[1]
    return processor.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def run_file(
    model_path: str,
    input_file: str,
    output_file: str,
    image_root: str = "data",
    batch_size: int = 4,
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    max_new_tokens: int = INFERENCE_MAX_NEW_TOKENS,
    shard: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
) -> str:
    """Generate responses for a prompt file and write them as JSON Lines.

    Each output record is the input record with the model's response added under
    ``answer``. Records are appended as each batch finishes, so an interrupted run
    leaves usable partial output.

    Args:
        model_path: Local directory or Hugging Face repository id.
        input_file: Prompt JSONL file.
        output_file: Where to write responses.
        image_root: Directory the benchmark images were extracted into.
        batch_size: Items per generation call.
        dtype: Model dtype.
        attn_implementation: Attention kernel.
        max_new_tokens: Generation budget.
        shard: Index of this shard, for splitting a file across processes.
        num_shards: How many shards the file is split into.
        limit: Stop after this many items of the shard, for a quick check.

    Returns:
        The path written.
    """
    with open(input_file, encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    items = items[shard::num_shards]
    if limit is not None:
        items = items[:limit]

    model, processor = load_model(
        model_path, dtype=dtype, attn_implementation=attn_implementation
    )

    ensure_dir(os.path.dirname(output_file))
    done = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            responses = generate_batch(
                model,
                processor,
                [item["input_prompt"] for item in batch],
                [resolve_image_paths(item["images"], image_root) for item in batch],
                max_new_tokens=max_new_tokens,
            )
            for item, response in zip(batch, responses, strict=False):
                out.write(json.dumps({**item, "answer": response}, ensure_ascii=False) + "\n")
            out.flush()
            done += len(batch)
            print(f"   {done}/{len(items)} items", flush=True)

    return output_file
