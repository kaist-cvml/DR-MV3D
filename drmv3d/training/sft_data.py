"""Converting prompts into the conversation format the Qwen trainer reads.

The trainer expects a JSON list of conversations. Each holds the item's image
paths and a two-turn exchange: the human turn is one ``<image>`` placeholder per
image followed by the prompt, and the assistant turn is the target.

The image paths go under ``image``, singular, even though every item has several.
That is the key the trainer looks for; with anything else it takes its text-only
path, sets no pixel values, and trains without ever seeing the images.
"""

#: Placeholder the trainer replaces with a real image.
IMAGE_PLACEHOLDER = "<image>"

#: The key the trainer reads an item's image paths from. Singular by its choice,
#: not ours.
IMAGE_KEY = "image"


def to_conversation(item: dict) -> dict:
    """Convert one prompt item into a training conversation.

    Args:
        item: Prompt item with ``images``, ``input_prompt`` and
            ``grounded_output``.

    Returns:
        The conversation record.

    Raises:
        KeyError: If a required field is missing.
        TypeError: If ``images`` is not a list, which would silently produce one
            placeholder per character.
    """
    images = item["images"]
    if not isinstance(images, list):
        raise TypeError(f"images must be a list, got {type(images).__name__}")

    placeholders = "\n".join(IMAGE_PLACEHOLDER for _ in images)
    return {
        IMAGE_KEY: images,
        "conversations": [
            {"from": "human", "value": f"{placeholders}\n{item['input_prompt']}"},
            {"from": "gpt", "value": item["grounded_output"]},
        ],
    }


def to_conversations(items: list[dict]) -> list[dict]:
    """Convert a whole file's prompt items.

    Args:
        items: Prompt items, in file order.

    Returns:
        One conversation per item, in the same order.
    """
    return [to_conversation(item) for item in items]
