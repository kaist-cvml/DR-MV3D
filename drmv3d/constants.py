"""The few values that more than one module has to agree on.

Everything else is written out where it is used. Field names of data items are
literals, because the golden tests catch a typo immediately and a literal reads
better; the output tag names are literals for the same reason. What is left here is
what a reader would otherwise have to hunt for: the decoding budget the reported
numbers were measured with, the seed that makes data generation reproducible, and
the question settings the released splits contain.

The task names themselves live in ``configs/sft_registry.json`` and
``configs/sft.sh``, since the training launcher reads them from there.
"""

from typing import Final

#: Model the released checkpoints were fine-tuned from. Its processor and config
#: describe the architecture, which fine-tuning does not change.
BASE_MODEL: Final[str] = "Qwen/Qwen2.5-VL-3B-Instruct"

#: Upper bound on image tokens at inference time, the value the reported numbers
#: were measured with. Training uses a smaller budget, set in configs/sft.sh. The
#: lower bound is left at the processor's own default, so only this is overridden.
INFERENCE_MAX_PIXELS: Final[int] = 480 * 480

#: Long enough for the full four-block output; the longest training target is
#: about 1,600 tokens.
INFERENCE_MAX_NEW_TOKENS: Final[int] = 4096

#: Question settings in the released splits. An item id names exactly one, and the
#: order is the order they are looked for in.
SETTINGS: Final[tuple[str, ...]] = ("around", "among", "rotation")

#: Seed for the object order in the generated cognitive-map instruction. Fixed so
#: that regenerating the data reproduces the released prompts.
SCAFFOLD_SEED: Final[int] = 42
