"""Running the three scaffold stages over a whole file.

The stages are strictly ordered. The cognitive map places the scene on a grid,
the egocentric maps re-express it from each viewpoint, and the reasoning chain
walks from the scene layout to the answer. Each stage reads what the previous one
wrote.
"""

import random
from collections.abc import Iterable
from typing import Any

from ..constants import SCAFFOLD_SEED
from .cogmap import add_cogmap
from .egomap import build_multi_view_egomap
from .reasoning import build_reasoning_chain


def add_scaffold(item: dict, rng: random.Random, quiet: bool = False) -> dict:
    """Add the cognitive map, the egocentric maps and the reasoning chain.

    Args:
        item: Raw benchmark item, modified in place.
        rng: Random source for the object order in the cognitive-map
            instruction. See :func:`drmv3d.scaffold.cogmap.add_cogmap`.
        quiet: Suppress non-fatal warnings.

    Returns:
        The same item, with ``grounded_cogmap``, ``cogmap_instruction``,
        ``grounded_egomap`` and ``reasoning_chain`` added.
    """
    add_cogmap(item, rng, quiet=quiet)
    egomap, egomap_meta = build_multi_view_egomap(item)
    item["grounded_egomap"] = egomap
    item["reasoning_chain"] = build_reasoning_chain(item, egomap_meta)
    return item


def add_scaffold_to_all(items: Iterable[dict[str, Any]], quiet: bool = False) -> list[dict]:
    """Scaffold a whole file's items in order.

    The random source is created here and drawn from in item order, so scaffolding
    a whole file is reproducible but a single item in isolation is not. That is
    deliberate: the released prompts were produced this way.

    Args:
        items: Raw benchmark items, in file order.
        quiet: Suppress non-fatal warnings.

    Returns:
        The scaffolded items, in the same order.
    """
    rng = random.Random(SCAFFOLD_SEED)
    return [add_scaffold(item, rng, quiet=quiet) for item in items]
