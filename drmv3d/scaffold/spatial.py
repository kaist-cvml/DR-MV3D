"""Grid geometry shared by the cognitive-map, egocentric-map and reasoning stages.

The scene is laid out on a discrete bird's-eye grid whose origin ``[0, 0]`` is the
top-left cell, so ``+x`` points east and ``+y`` points south. A viewpoint is a grid
cell plus one of four facing directions.

This module is a leaf: it imports nothing from the rest of the package.
"""

from typing import Any

#: Unit step on the grid for each facing direction, as ``(dx, dy)``.
DIRECTION_VECTORS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

#: Facing after a 90-degree clockwise turn.
FACING_CLOCKWISE: dict[str, str] = {"up": "right", "right": "down", "down": "left", "left": "up"}

#: Facing after a 90-degree counter-clockwise turn.
FACING_COUNTERCLOCKWISE: dict[str, str] = {
    "up": "left",
    "left": "down",
    "down": "right",
    "right": "up",
}

#: Facing after a 180-degree turn.
FACING_OPPOSITE: dict[str, str] = {"up": "down", "down": "up", "left": "right", "right": "left"}

#: Egocentric frame for each facing: which grid direction is "forward" and which
#: is "right", plus the axis labels that appear verbatim in reasoning chains.
FACING_AXES: dict[str, dict[str, Any]] = {
    "up": {
        "forward": (0, -1),
        "right": (1, 0),
        "forward_label": "-y (north)",
        "right_label": "+x (east)",
    },
    "down": {
        "forward": (0, 1),
        "right": (-1, 0),
        "forward_label": "+y (south)",
        "right_label": "-x (west)",
    },
    "left": {
        "forward": (-1, 0),
        "right": (0, -1),
        "forward_label": "-x (west)",
        "right_label": "-y (north)",
    },
    "right": {
        "forward": (1, 0),
        "right": (0, 1),
        "forward_label": "+x (east)",
        "right_label": "+y (south)",
    },
}

_UNKNOWN_AXES: dict[str, Any] = {
    "forward": (0, -1),
    "right": (1, 0),
    "forward_label": "?",
    "right_label": "?",
}

#: Egocentric directions that satisfy each spatial query word. A question asking
#: what is "to my right" accepts ``right`` as well as the two diagonals on that
#: side.
TARGET_DIRECTION_SETS: dict[str, set[str]] = {
    "right": {"right", "forward-right", "back-right"},
    "left": {"left", "forward-left", "back-left"},
    "front": {"forward", "forward-left", "forward-right"},
    "behind": {"back", "back-left", "back-right"},
}

#: Neighbouring egocentric directions on the eight-way compass, used to find the
#: closest direction consistent with a ground-truth answer.
DIRECTION_ADJACENCY: dict[str, list[str]] = {
    "forward": ["forward-right", "forward-left"],
    "forward-right": ["forward", "right"],
    "right": ["forward-right", "back-right"],
    "back-right": ["right", "back"],
    "back": ["back-right", "back-left"],
    "back-left": ["back", "left"],
    "left": ["back-left", "forward-left"],
    "forward-left": ["left", "forward"],
}


def get_facing_axes(facing: str) -> dict[str, Any]:
    """Return the egocentric frame for a facing direction.

    Args:
        facing: One of ``up``, ``down``, ``left``, ``right``.

    Returns:
        Mapping with ``forward`` and ``right`` unit vectors and their human
        readable labels. Unknown directions fall back to the ``up`` frame with
        ``"?"`` labels.
    """
    return FACING_AXES.get(facing, _UNKNOWN_AXES)


def compute_ego_direction(
    obj_position: list[int],
    camera_position: list[int],
    camera_facing: str,
) -> str:
    """Describe where an object lies relative to a viewpoint.

    The displacement from the camera to the object is projected onto the
    camera's forward and right axes; each non-zero component contributes one
    word, giving values such as ``forward``, ``left`` or ``back-right``.

    Args:
        obj_position: Object cell as ``[x, y]``.
        camera_position: Viewpoint cell as ``[x, y]``.
        camera_facing: Viewpoint facing, one of ``up``, ``down``, ``left``, ``right``.

    Returns:
        An egocentric direction, ``"at_camera"`` if the object shares the
        viewpoint cell, or ``"unknown"`` if the facing is not recognised.
    """
    dx = obj_position[0] - camera_position[0]
    dy = obj_position[1] - camera_position[1]
    if dx == 0 and dy == 0:
        return "at_camera"

    axes = FACING_AXES.get(camera_facing)
    if axes is None:
        return "unknown"

    forward_x, forward_y = axes["forward"]
    right_x, right_y = axes["right"]
    forward_delta = dx * forward_x + dy * forward_y
    right_delta = dx * right_x + dy * right_y

    forward_word = "forward" if forward_delta > 0 else ("back" if forward_delta < 0 else "")
    right_word = "right" if right_delta > 0 else ("left" if right_delta < 0 else "")

    if forward_word and right_word:
        return f"{forward_word}-{right_word}"
    # The axes are orthonormal, so at least one projection is non-zero unless the
    # object shares the viewpoint's cell, which returned above.
    return forward_word or right_word


def rotate_facing(facing: str, action: str) -> str:
    """Apply a turn action to a facing direction.

    Args:
        facing: Current facing.
        action: ``TURN_LEFT_90``, ``TURN_RIGHT_90`` or ``TURN_180``.

    Returns:
        The new facing, or ``facing`` unchanged for any other action.
    """
    if action == "TURN_LEFT_90":
        return FACING_COUNTERCLOCKWISE.get(facing, facing)
    if action == "TURN_RIGHT_90":
        return FACING_CLOCKWISE.get(facing, facing)
    if action == "TURN_180":
        return FACING_OPPOSITE.get(facing, facing)
    return facing


def move_camera(position: list[int], facing: str, action: str) -> list[int]:
    """Apply a one-cell translation expressed in camera-relative terms.

    Args:
        position: Current cell as ``[x, y]``.
        facing: Current facing, which defines what "forward" means.
        action: ``MOVE_FORWARD``, ``MOVE_BACK``, ``MOVE_LEFT`` or ``MOVE_RIGHT``.

    Returns:
        The new cell, or a copy of ``position`` for any other action.
    """
    x, y = position
    forward_x, forward_y = DIRECTION_VECTORS.get(facing, (0, -1))
    right_x, right_y = -forward_y, forward_x

    if action == "MOVE_FORWARD":
        return [x + forward_x, y + forward_y]
    if action == "MOVE_BACK":
        return [x - forward_x, y - forward_y]
    if action == "MOVE_LEFT":
        return [x - right_x, y - right_y]
    if action == "MOVE_RIGHT":
        return [x + right_x, y + right_y]
    return [x, y]
