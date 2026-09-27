"""Building the sample table the reinforcement-learning trainer reads.

The trainer wants one row per prompt: the prompt itself, the images, the correct
answer, and whatever the reward function needs. The last part is what matters
here. Every reference signal the dense reward scores against is written into each
row's metadata, so the reward function stays a pure function of a response and
its row.
"""

import os
from typing import Any

from .parsing import parse_json_relaxed, primary_egomap
from .reward import (
    KEY_REFERENCE_ANSWER_VIEW,
    KEY_REFERENCE_COGMAP,
    KEY_REFERENCE_SELECTED_VIEW,
)

#: Dataset name the trainer records on every row.
DATA_SOURCE = "mindcube"

#: Free-text capability label the trainer records on every row.
ABILITY = "spatial_reasoning"


def reference_view_trajectory(item: dict) -> tuple[str, str]:
    """Return the reference view trajectory of an item.

    The trajectory is read out of the item's own egocentric maps, where the
    primary entry names the viewpoint the question reasons from and the viewpoint
    that shows the answer object.

    Args:
        item: Scaffolded item.

    Returns:
        The selected viewpoint and the viewpoint showing the answer object, each
        an empty string if the item does not name it.
    """
    primary = primary_egomap(parse_json_relaxed(item.get("grounded_egomap", "")))
    if primary is None:
        return "", ""
    return str(primary.get("view", "") or ""), str(primary.get("answer_visible_in_view", "") or "")


def to_rl_sample(item: dict, index: int, image_root: str) -> dict[str, Any]:
    """Convert one prompt item into a trainer row.

    Args:
        item: Prompt item, which must carry ``input_prompt``, ``images``,
            ``gt_answer``, ``grounded_cogmap`` and ``grounded_egomap``.
        index: Row number, which the trainer uses to identify a sample.
        image_root: Directory the benchmark images were extracted into. Image
            paths are made absolute because the trainer's workers do not share
            this process's working directory.

    Returns:
        The row.
    """
    from ..training.sft_data import IMAGE_PLACEHOLDER

    images = item["images"]
    placeholders = "\n".join(IMAGE_PLACEHOLDER for _ in images)
    selected_view, answer_view = reference_view_trajectory(item)
    answer = item["gt_answer"]

    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": f"{placeholders}\n{item['input_prompt']}"}],
        "images": [{"image": os.path.abspath(os.path.join(image_root, path))} for path in images],
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": index,
            "id": item.get("id", ""),
            "answer": answer,
            "num_images": len(images),
            KEY_REFERENCE_COGMAP: item.get("grounded_cogmap", ""),
            KEY_REFERENCE_SELECTED_VIEW: selected_view,
            KEY_REFERENCE_ANSWER_VIEW: answer_view,
        },
    }


def prompt_token_length(processor: Any, prompt: str, image_paths: list[str]) -> int:
    """Return how many tokens a prompt occupies once its images are encoded.

    Args:
        processor: The base model's processor.
        prompt: Prompt text, including one image placeholder per image.
        image_paths: Absolute paths of the item's images.

    Returns:
        The token count.
    """
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in image_paths]
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    encoded = processor(text=text, images=images, return_tensors="pt", padding=False)
    return int(encoded["input_ids"].shape[1])


def build_rl_samples(
    items: list[dict],
    image_root: str,
    max_prompt_length: int | None = None,
    processor: Any = None,
) -> tuple[list[dict], int]:
    """Convert a prompt file into trainer rows, dropping over-long prompts.

    The trainer allocates a fixed prompt buffer, so a prompt that does not fit
    would be truncated mid-scene. Dropping those rows is the honest option.

    Args:
        items: Prompt items, in file order.
        image_root: Directory the benchmark images were extracted into.
        max_prompt_length: Token budget, or ``None`` to keep every row.
        processor: The base model's processor, required when a budget is given.

    Returns:
        The rows kept, and how many were dropped as too long.

    Raises:
        ValueError: If a budget is given without a processor.
    """
    if max_prompt_length is not None and processor is None:
        raise ValueError("max_prompt_length needs a processor to measure prompts with")

    rows: list[dict] = []
    dropped = 0
    for index, item in enumerate(items):
        row = to_rl_sample(item, index, image_root)
        if max_prompt_length is not None:
            length = prompt_token_length(
                processor,
                row["prompt"][0]["content"],
                [image["image"] for image in row["images"]],
            )
            if length > max_prompt_length:
                dropped += 1
                continue
        rows.append(row)
    return rows, dropped
