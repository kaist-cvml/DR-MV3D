"""Allocentric cognitive maps: the whole scene on one bird's-eye grid.

Each benchmark item is a set of photographs of one room plus a spatial question.
The images alone carry no common frame of reference, so the first stage of the
pipeline lays the scene out on a shared 10x10 grid: where each object sits, and
where each camera stood and which way it looked.

The benchmark's three settings differ in how the cameras relate to the scene, so
each gets its own builder:

``around``
    Cameras stand outside a row of objects and look inward from a few sides.
``among``
    Cameras stand inside a ring of five objects and look outward.
``rotation``
    One camera stays put and turns on the spot.

The grid coordinates are fixed per setting rather than measured, because the
benchmark annotates only the qualitative arrangement. That is what makes these
maps a supervision target: they are exactly recoverable from the annotation.
"""

import json
import os
import random
import re
from typing import Any

from ..constants import SETTINGS

#: The task description prepended to every prompt. It tells the model what a
#: cognitive map is and which objects to place. ``{birdview}`` and ``{obj}`` are
#: filled in per item; the surrounding braces of ``{{obj}}`` are literal and
#: appear in the finished prompt.
COGMAP_INSTRUCTION = '''[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints. You will then create a detailed cognitive map representing the scene using a **10x10 grid coordinate system**. 
[Rules]
1. Focus ONLY on these categories of objects in the scene: {{obj}}
2. Create a cognitive map with the following structure{birdview}:
   - A 10x10 grid where [0, 0] is at the top-left corner and [9, 9] is at the bottom-right corner
   - up = towards the top of the grid (decreasing y)
   - right = towards the right of the grid (increasing x)
   - down = towards the bottom of the grid (increasing y)
   - left = towards the left of the grid (decreasing x)
   - Include positions of all objects from the specified categories
   - Estimate the center location (coordinates [x, y]) of each instance within provided categories
   - If a category contains multiple instances, include all of them
   - Object positions must maintain accurate relative spatial relationships
   - Combine and merge information from the images since they are pointing to the same scene, calibrating the object locations with grid coordinates accordingly
3. Carefully integrate information from all views to create a single coherent spatial representation.
<orientation_info>'''

#: Filled into ``{birdview}`` above. Every setting released here is annotated
#: from above, so this is constant.
_BIRDVIEW_PHRASE = " in the bird's view"

#: How the benchmark's object-orientation labels map onto grid facings. The
#: labels are relative to the photographer, so "face" means facing the camera,
#: which on a grid drawn from above points down the page.
_ORIENTATION_TO_FACING: dict[str | None, str | None] = {
    "face": "down",
    "back": "up",
    "right": "right",
    "left": "left",
    "None": None,
    "null": None,
    None: None,
}

#: Grid facing for a camera placed on each side of the scene.
_SIDE_TO_FACING: dict[str, str] = {
    "front": "up",
    "left": "right",
    "right": "left",
    "back": "down",
}

#: Two ``around`` image groups came from a six-view capture but are annotated as
#: five-view groups. Matching them by id restores the six-view side assignment.
#: In the released splits both groups only ever use views 1 to 3, where the two
#: layouts agree, so this changes no map today; it is kept so that a question
#: using a later view of those groups is placed correctly.
_AROUND_GROUPS_NEEDING_SIX_VIEW_LAYOUT: tuple[str, ...] = (
    "d254259ad5be1ec58a5591aa18011ee766c05bf76fd1ea4cd17fe994e6e7707c",
    "e3f7e530666356f04f6ea50d7eb3173c130d7d5b0da8bcb332de4cf20961c60c",
)

#: Which side of the scene each ``around`` view was shot from, keyed by the
#: group's view count and by which source dataset the group came from.
_AROUND_SIX_VIEW_SIDES: dict[int, str] = {
    1: "front", 2: "left", 3: "right", 4: "front", 5: "left", 6: "right",
}
_AROUND_VIEW_SIDES: dict[tuple[int, str], dict[int, str]] = {
    (3, "self"): {1: "front", 2: "left", 3: "right"},
    (3, "dl3dv10k"): {1: "front", 2: "left", 3: "right", 4: "back"},
    (4, "self"): {1: "front", 2: "left", 3: "right", 4: "back"},
    (4, "dl3dv10k"): {1: "front", 2: "left", 3: "right", 4: "back"},
    (5, "self"): {1: "front", 2: "left", 3: "right", 4: "left", 5: "right"},
    (5, "dl3dv10k"): {1: "front", 2: "left", 3: "right", 4: "left", 5: "right"},
    (6, "self"): {1: "front", 2: "left", 3: "right", 4: "left", 5: "right", 6: "back"},
    (6, "dl3dv10k"): _AROUND_SIX_VIEW_SIDES,
}

#: Camera cell for each side of an ``around`` scene, indexed by how many objects
#: the row holds: entry 0 is a two-object row, entry 2 a four-object row. More
#: objects means a wider row, so the side cameras step further out.
_AROUND_CAMERA_CELLS: dict[str, list[list[int]]] = {
    "front": [[5, 6], [5, 6], [5, 6]],
    "left": [[3, 5], [3, 5], [2, 5]],
    "right": [[6, 5], [7, 5], [7, 5]],
    "back": [[5, 4], [5, 4], [5, 4]],
}

#: Object cells for an ``around`` row, by row length: the row is centred on the
#: grid and laid out left to right.
_AROUND_OBJECT_CELLS: dict[int, list[list[int]]] = {
    2: [[4, 5], [5, 5]],
    3: [[4, 5], [5, 5], [6, 5]],
    4: [[3, 5], [4, 5], [5, 5], [6, 5]],
}

#: The five ``among`` objects ring the centre of the grid: one at the middle,
#: then one on each side.
_AMONG_OBJECT_CELLS: list[list[int]] = [[5, 5], [5, 8], [2, 5], [5, 2], [8, 5]]

#: Camera cell for each side of an ``among`` scene. The cameras stand just
#: inside the ring and look outward.
_AMONG_CAMERA_CELLS: dict[str, list[int]] = {
    "front": [5, 6],
    "left": [4, 5],
    "right": [6, 5],
    "back": [5, 4],
}

#: Sides assumed for a four-image ``among`` group whose file names do not say
#: which side they were shot from.
_AMONG_FALLBACK_SIDES: tuple[str, ...] = ("front", "left", "back", "right")

#: Object cell and camera facing for each ``rotation`` configuration. The camera
#: never leaves the centre cell, so a turn brings a different object into view.
_ROTATION_LAYOUTS: dict[str, tuple[list[list[int]], list[str]]] = {
    "two_view_clockwise": ([[5, 3], [7, 5]], ["up", "right"]),
    "two_view_counterclockwise": ([[5, 3], [3, 5]], ["up", "left"]),
    "two_view_opposite": ([[5, 3], [5, 7]], ["up", "down"]),
    "three_view": ([[3, 5], [5, 3], [7, 5]], ["left", "up", "right"]),
    "four_view": ([[3, 5], [5, 3], [7, 5], [5, 7]], ["left", "up", "right", "down"]),
}

_ROTATION_CAMERA_CELL: list[int] = [5, 5]

#: Matches the view number in an ``around`` image file name.
_AROUND_FRAME_NUMBER = re.compile(r"(\d+)_frame(?:_[^.]+)?\.(?:png|jpg|jpeg)")

#: One ``around`` group's third view is named "33" instead of "3". Correcting it
#: here avoids rewriting the published image files.
_AROUND_FRAME_NUMBER_FIXES: dict[int, int] = {33: 3}


def detect_setting(item_id: str) -> str | None:
    """Return the benchmark setting an item belongs to.

    Args:
        item_id: Identifier such as ``"among_group693_q1_5_2"``.

    Returns:
        One of :data:`SETTINGS`, or ``None`` if the id names none of them.
    """
    for setting in SETTINGS:
        if setting in item_id:
            return setting
    return None


def format_cogmap(cogmap: dict[str, list[dict]]) -> str:
    """Serialise a cognitive map with one object or viewpoint per line.

    The layout is part of the released prompts, so it is written out by hand
    rather than left to :func:`json.dumps`: keys stay in insertion order and
    each entry is a single compact line inside a multi-line envelope.

    Args:
        cogmap: Mapping with ``objects`` and ``views`` lists.

    Returns:
        A JSON string.

    Raises:
        ValueError: If the result does not parse back to the input, which would
            mean an entry held something JSON cannot represent exactly.
    """
    parts = ["{\n"]
    for key in ("objects", "views"):
        parts.append(f'  "{key}": [\n')
        entries = cogmap[key]
        for index, entry in enumerate(entries):
            separator = "," if index < len(entries) - 1 else ""
            parts.append("    " + json.dumps(entry, ensure_ascii=False) + separator + "\n")
        parts.append("  ],\n" if key == "objects" else "  ]\n")
    parts.append("}")
    result = "".join(parts)

    reparsed = json.loads(result)
    if reparsed["objects"] != cogmap["objects"] or reparsed["views"] != cogmap["views"]:
        raise ValueError("serialised cognitive map does not round-trip")
    return result


def _object_entry(
    name: str,
    position: list[int],
    facing: str | None = None,
) -> dict[str, Any] | None:
    """Build one object entry, or ``None`` for an unnamed object.

    Args:
        name: Object name from the annotation. An empty name means the scene has
            fewer objects than the layout has slots.
        position: Grid cell as ``[x, y]``.
        facing: Grid facing, or ``None`` for an object with no annotated
            orientation, in which case the key is left out entirely.

    Returns:
        The entry, or ``None`` if ``name`` is empty.
    """
    if name == "":
        return None
    if facing is None:
        return {"name": name, "position": position}
    return {"name": name, "position": position, "facing": facing}


def _view_entries(names: list[str], cells: list[list[int]], facings: list[str]) -> list[dict]:
    """Zip viewpoint names, cells and facings into cognitive-map entries."""
    return [
        {"name": name, "position": cell, "facing": facing}
        for name, cell, facing in zip(names, cells, facings, strict=False)
    ]


def _view_base_name(question: str) -> str:
    """Return the word the question uses for a viewpoint.

    Questions are phrased with either "image" or "view", and the map has to use
    the same word so the two can be read together.

    Args:
        question: Full question text.

    Returns:
        ``"Image"`` or ``"View"``.
    """
    return "Image" if "image" in question.lower() else "View"


def _around_view_numbers(images: list[str]) -> list[int]:
    """Return the group-wide view number of each image of an ``around`` item.

    An ``around`` question shows a subset of its group's views, so the file
    names are what say which sides of the scene are on screen.

    Args:
        images: Image paths of one item.

    Returns:
        One view number per image, in image order.

    Raises:
        ValueError: If a file name carries no view number.
    """
    numbers = []
    for image in images:
        match = _AROUND_FRAME_NUMBER.search(os.path.basename(image))
        if not match:
            raise ValueError(f"no view number in image file name: {image}")
        number = int(match.group(1))
        numbers.append(_AROUND_FRAME_NUMBER_FIXES.get(number, number))
    return numbers


def build_around_cogmap(item: dict) -> tuple[str, list[str], list[str]]:
    """Build the cognitive map of an ``around`` item.

    The objects form a row across the middle of the grid and the cameras stand
    around it. Which sides are shown depends on the group's view count, its
    source dataset, and which of the group's views this question uses.

    Args:
        item: Raw benchmark item.

    Returns:
        The serialised map, the object names, and the names of those objects
        whose facing the annotation gives.

    Raises:
        AssertionError: If the annotation is internally inconsistent.
        ValueError: If the group's view count has no known camera layout.
    """
    item_id = item.get("id", "")
    meta_info = item.get("meta_info", [])
    question = item.get("question", "")
    images = item.get("images", [])

    # The id's leading token names the source dataset: "aroundnew" for the
    # authors' own captures, anything else for the DL3DV-10K subset.
    source = "self" if item_id.split("_")[0].replace("around", "") == "new" else "dl3dv10k"

    group_view_count = meta_info[0][0]
    object_count = meta_info[1][0]
    objects = meta_info[1][1]
    orientations = meta_info[1][2]

    assert object_count == len(objects), (
        f"object count {object_count} does not match {len(objects)} names, id: {item_id}"
    )
    assert object_count == len(orientations), (
        f"object count {object_count} does not match {len(orientations)} orientations, id: {item_id}"
    )
    assert isinstance(group_view_count, int), (
        f"group view count {group_view_count!r} is not an integer, id: {item_id}"
    )
    assert len(images) <= group_view_count, (
        f"item shows {len(images)} images but its group has {group_view_count}, id: {item_id}"
    )
    assert 3 <= group_view_count <= 6, (
        f"group view count {group_view_count} is outside 3..6, id: {item_id}"
    )

    if any(group in item_id for group in _AROUND_GROUPS_NEEDING_SIX_VIEW_LAYOUT):
        view_sides = _AROUND_SIX_VIEW_SIDES
    else:
        view_sides = _AROUND_VIEW_SIDES.get((group_view_count, source), {})
    if not view_sides:
        raise ValueError(
            f"no camera layout for {group_view_count} views from {source}, id: {item_id}"
        )

    view_numbers = _around_view_numbers(images)
    assert len(view_numbers) == len(set(view_numbers)), (
        f"repeated view number in {view_numbers}, id: {item_id}"
    )
    for number in view_numbers:
        assert 1 <= number <= len(view_sides), (
            f"view number {number} is outside 1..{len(view_sides)}, id: {item_id}"
        )

    assert object_count in _AROUND_OBJECT_CELLS, (
        f"no object layout for {object_count} objects, id: {item_id}"
    )
    object_entries = [
        _object_entry(name, cell, _ORIENTATION_TO_FACING[orientation])
        for name, cell, orientation in zip(
            objects, _AROUND_OBJECT_CELLS[object_count], orientations, strict=False
        )
    ]
    object_entries = [entry for entry in object_entries if entry is not None]

    # A wider object row pushes the side cameras further out.
    width_index = object_count - 2
    base = _view_base_name(question)
    sides = [view_sides[number] for number in view_numbers]
    view_entries = _view_entries(
        [f"{base} {index + 1}" for index in range(len(sides))],
        [_AROUND_CAMERA_CELLS[side][width_index] for side in sides],
        [_SIDE_TO_FACING[side] for side in sides],
    )

    cogmap = {"objects": object_entries, "views": view_entries}
    oriented = [entry["name"] for entry in object_entries if "facing" in entry]
    return format_cogmap(cogmap), objects, oriented


def build_among_cogmap(item: dict, quiet: bool = False) -> tuple[str, list[str], list[str]]:
    """Build the cognitive map of an ``among`` item.

    Five objects ring the centre of the grid and the cameras stand inside the
    ring looking outward. Which side each camera is on is normally read from the
    image file name.

    Args:
        item: Raw benchmark item.
        quiet: Suppress the warning printed when file names do not name a side.

    Returns:
        The serialised map, the object names, and the names of those objects
        whose facing the annotation gives.

    Raises:
        AssertionError: If the annotation is internally inconsistent.
        ValueError: If the sides cannot be recovered from the file names.
    """
    item_id = item.get("id", "")
    meta_info = item.get("meta_info", [])
    question = item.get("question", "")
    images = item.get("images", [])

    objects = meta_info[0]
    orientations = meta_info[1]

    assert len(objects) == 5, f"among needs 5 objects, got {len(objects)}, id: {item_id}"
    assert len(orientations) == 5, (
        f"among needs 5 orientations, got {len(orientations)}, id: {item_id}"
    )
    assert len(images) in (2, 4), (
        f"among needs 2 or 4 images, got {len(images)}, id: {item_id}"
    )

    named_sides = [os.path.basename(image).split("_")[0] for image in images]
    if all(side in _SIDE_TO_FACING for side in named_sides):
        sides = named_sides
    else:
        # Some groups use opaque file names. A full four-image group can still
        # be placed, because those are always shot in the same order.
        if len(images) != 4:
            raise ValueError(
                f"image names {named_sides} do not name a side and the group is "
                f"not a full four-image group, id: {item_id}"
            )
        sides = list(_AMONG_FALLBACK_SIDES)
        if not quiet:
            print(f"⚠️  {item_id}: image names {named_sides} do not name a side; "
                  f"assuming {sides}")

    object_entries = [
        _object_entry(name, cell, _ORIENTATION_TO_FACING[orientation])
        for name, cell, orientation in zip(objects, _AMONG_OBJECT_CELLS, orientations, strict=False)
    ]
    object_entries = [entry for entry in object_entries if entry is not None]

    base = _view_base_name(question)
    view_entries = _view_entries(
        [f"{base} {index + 1}" for index in range(len(sides))],
        [_AMONG_CAMERA_CELLS[side] for side in sides],
        [_SIDE_TO_FACING[side] for side in sides],
    )

    cogmap = {"objects": object_entries, "views": view_entries}
    oriented = [entry["name"] for entry in object_entries if "facing" in entry]
    return format_cogmap(cogmap), objects, oriented


def build_rotation_cogmap(item: dict) -> tuple[str, list[str], list[str]]:
    """Build the cognitive map of a ``rotation`` item.

    The camera stays in the centre cell and turns, so each view faces a
    different object. The item's type names which turn sequence was used.

    Args:
        item: Raw benchmark item.

    Returns:
        The serialised map, the object names, and an empty oriented-object list,
        since rotation items annotate no object facings.

    Raises:
        AssertionError: If the object count does not match the turn sequence.
        ValueError: If the item's type names no known turn sequence.
    """
    item_id = item.get("id", "")
    item_type = item.get("type", "")
    objects = item.get("meta_info", [])
    question = item.get("question", "")

    layout = _ROTATION_LAYOUTS.get(item_type)
    if layout is None:
        raise ValueError(f"unknown rotation type {item_type!r}, id: {item_id}")
    cells, facings = layout
    assert len(objects) == len(cells), (
        f"{item_type} needs {len(cells)} objects, got {len(objects)}, id: {item_id}"
    )

    object_entries = [_object_entry(name, cell) for name, cell in zip(objects, cells, strict=False)]
    object_entries = [entry for entry in object_entries if entry is not None]

    base = _view_base_name(question)
    view_entries = _view_entries(
        [f"{base} {index + 1}" for index in range(len(facings))],
        [list(_ROTATION_CAMERA_CELL) for _ in facings],
        facings,
    )

    cogmap = {"objects": object_entries, "views": view_entries}
    return format_cogmap(cogmap), objects, []


def build_cogmap(item: dict, quiet: bool = False) -> tuple[str, list[str], list[str]]:
    """Build the cognitive map of an item, dispatching on its setting.

    Args:
        item: Raw benchmark item.
        quiet: Suppress non-fatal warnings.

    Returns:
        The serialised map, the object names, and the names of those objects
        whose facing the annotation gives.

    Raises:
        ValueError: If the item id names no known setting.
    """
    item_id = item.get("id", "")
    setting = detect_setting(item_id)
    if setting == "around":
        return build_around_cogmap(item)
    if setting == "among":
        return build_among_cogmap(item, quiet=quiet)
    if setting == "rotation":
        return build_rotation_cogmap(item)
    raise ValueError(f"no known setting in item id: {item_id!r}")


def add_cogmap(item: dict, rng: random.Random, quiet: bool = False) -> dict:
    """Add the cognitive map and its prompt instruction to an item, in place.

    Two fields are written. ``grounded_cogmap`` is the map itself, the
    supervision target for the model's first output block.
    ``cogmap_instruction`` is the task description that goes into the prompt; it
    names the objects to place and, where the annotation gives an orientation,
    asks for a facing too.

    The object names in that instruction are shuffled so the model cannot read the
    answer off their order. Both shuffles draw from ``rng`` in a fixed order, so
    passing a freshly seeded generator and iterating a file from the start
    reproduces the released prompts exactly. A single item scaffolded on its own
    does not.

    Args:
        item: Raw benchmark item, modified in place.
        rng: Random source for the object-name order.
        quiet: Suppress non-fatal warnings.

    Returns:
        The same item.
    """
    cogmap, objects, oriented = build_cogmap(item, quiet=quiet)

    # Copy before shuffling: ``objects`` is the annotation's own list, and
    # reordering it in place would leave the item's metadata inconsistent with
    # the map just built from it.
    objects = list(objects)
    oriented = list(oriented)
    rng.shuffle(objects)
    rng.shuffle(oriented)

    if oriented:
        orientation_info = (
            f"4. For objects [{' '.join(oriented)}], determine their facing direction as "
            f"up, right, down, or left. For other objects, omit the facing direction."
        )
    else:
        orientation_info = ""

    instruction = (
        COGMAP_INSTRUCTION
        .replace("{birdview}", _BIRDVIEW_PHRASE)
        .replace("<orientation_info>", orientation_info)
        .replace("{obj}", ", ".join(objects))
    )

    item["grounded_cogmap"] = cogmap
    item["cogmap_instruction"] = instruction
    return item
