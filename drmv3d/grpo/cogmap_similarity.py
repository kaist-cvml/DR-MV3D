"""Comparing a predicted allocentric map with a reference map of the same scene.

Two maps of one room are compared on what they claim about the scene, not on
their coordinates. The absolute cells a model picks are arbitrary, so the
comparison is over *relations*: for every ordered pair of named things, which way
the second lies from the first. Two maps that agree on every relation describe the
same arrangement, wherever they put its origin.

The score has two parts:

directional
    The fraction of the reference map's pairwise relations the prediction gets
    right. This carries most of the weight, because the arrangement is what the
    question is about.
facing
    The fraction of the reference map's viewpoints whose direction of view the
    prediction gets right. A viewpoint's facing decides what "my left" means, so
    getting it wrong makes the egocentric stage wrong even from a correct map.

Both denominators count everything the reference map states, so leaving an object
or a viewpoint out costs score. Scoring only what the prediction happens to
mention would make an almost-empty map the easiest way to a perfect score.
"""

import math
from dataclasses import dataclass
from typing import Any

#: Relative weights of the two parts. The arrangement dominates.
DIRECTIONAL_WEIGHT = 0.7
FACING_WEIGHT = 0.3

#: Grid facings a viewpoint or object may have.
GRID_FACINGS: frozenset[str] = frozenset({"up", "down", "left", "right"})

#: Spellings that mean one of the grid facings, so that a model writing "north"
#: is not marked wrong for a naming choice.
FACING_ALIASES: dict[str, str] = {
    "top": "up",
    "bottom": "down",
    "north": "up",
    "south": "down",
    "east": "right",
    "west": "left",
}

#: Label for two things the map places in the same cell, which states no
#: direction. Both maps use it, so it compares like any other relation.
SAME_CELL = "same-cell"


@dataclass(frozen=True)
class CogmapSimilarity:
    """How closely a predicted map matches a reference map.

    Attributes:
        valid: Whether both maps were well formed enough to compare. Every other
            field is zero when this is ``False``.
        coverage: Fraction of the reference map's names the prediction also names.
            Reported for diagnosis; the scores below already penalise what is
            missing.
        directional: Fraction of pairwise relations the prediction gets right.
        facing: Fraction of viewpoint facings the prediction gets right.
        overall: The weighted combination, in ``[0, 1]``.
    """

    valid: bool
    coverage: float
    directional: float
    facing: float
    overall: float


_INVALID = CogmapSimilarity(valid=False, coverage=0.0, directional=0.0, facing=0.0, overall=0.0)


def normalise_facing(value: Any) -> str | None:
    """Return a facing as one of the grid directions.

    Args:
        value: A facing as a map records it, possibly absent or spelled
            differently.

    Returns:
        A member of :data:`GRID_FACINGS`, or ``None`` if the value names no
        recognised direction.
    """
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    lowered = FACING_ALIASES.get(lowered, lowered)
    return lowered if lowered in GRID_FACINGS else None


def _cell(value: Any) -> tuple[int, int] | None:
    """Return a grid cell as a pair of numbers, or ``None`` if malformed."""
    if not isinstance(value, list | tuple) or len(value) < 2:
        return None
    try:
        return (int(round(float(value[0]))), int(round(float(value[1]))))
    except (TypeError, ValueError):
        return None


def _read_entries(cogmap: Any) -> tuple[dict[str, tuple], dict[str, str | None]] | None:
    """Read a map into cells by name, and the facings of its viewpoints.

    Objects and viewpoints share one namespace, so a relation can run between an
    object and a viewpoint as readily as between two objects.

    Args:
        cogmap: A parsed cognitive map.

    Returns:
        Cells by name and viewpoint facings by name, or ``None`` if the map is not
        a well-formed cognitive map.
    """
    if not isinstance(cogmap, dict):
        return None
    objects, views = cogmap.get("objects"), cogmap.get("views")
    if not isinstance(objects, list) or not isinstance(views, list):
        return None

    cells: dict[str, tuple] = {}
    for entry in objects + views:
        if not isinstance(entry, dict):
            return None
        name = entry.get("name")
        cell = _cell(entry.get("position"))
        if not isinstance(name, str) or not name or cell is None:
            return None
        cells[name] = cell

    view_facings: dict[str, str | None] = {}
    for entry in views:
        name = entry.get("name")
        if isinstance(name, str) and name:
            view_facings[name] = normalise_facing(entry.get("facing"))

    return (cells, view_facings) if cells else None


def relative_direction(origin: tuple[int, int], target: tuple[int, int]) -> str:
    """Return which way ``target`` lies from ``origin`` on the grid.

    The dominant axis wins, so the result is one of the four grid directions. A
    tie resolves vertically, which only has to be consistent, not principled,
    because both maps are read the same way.

    Args:
        origin: Cell the relation is measured from, as ``[x, y]``.
        target: Cell the relation points to.

    Returns:
        ``up``, ``down``, ``left``, ``right``, or :data:`SAME_CELL`.
    """
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    if math.hypot(dx, dy) == 0:
        return SAME_CELL
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def compare_cogmaps(predicted: Any, reference: Any) -> CogmapSimilarity:
    """Score a predicted cognitive map against a reference map of the same scene.

    Args:
        predicted: The map the model produced.
        reference: The map to score against.

    Returns:
        The breakdown. ``valid`` is ``False`` if either map is malformed, in which
        case every score is zero.
    """
    left = _read_entries(predicted)
    right = _read_entries(reference)
    if left is None or right is None:
        return _INVALID

    predicted_cells, predicted_facings = left
    reference_cells, reference_facings = right

    shared = set(predicted_cells) & set(reference_cells)
    coverage = len(shared) / len(reference_cells)

    # Every ordered pair the reference map states, whether or not the prediction
    # names both of its ends.
    total = matched = 0
    reference_names = sorted(reference_cells)
    for origin in reference_names:
        for target in reference_names:
            if origin == target:
                continue
            total += 1
            if origin in shared and target in shared:
                wanted = relative_direction(reference_cells[origin], reference_cells[target])
                got = relative_direction(predicted_cells[origin], predicted_cells[target])
                matched += int(got == wanted)
    directional = matched / total if total else 0.0

    # Every viewpoint the reference map gives a facing for.
    stated = [name for name, facing in reference_facings.items() if facing]
    facing = (
        sum(predicted_facings.get(name) == reference_facings[name] for name in stated) / len(stated)
        if stated
        else 0.0
    )

    return CogmapSimilarity(
        valid=True,
        coverage=coverage,
        directional=directional,
        facing=facing,
        overall=DIRECTIONAL_WEIGHT * directional + FACING_WEIGHT * facing,
    )
