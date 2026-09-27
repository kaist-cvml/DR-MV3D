"""Replacing an item's annotated cognitive map with one estimated from its images.

Three steps. A 3D vision foundation model recovers camera poses and a per-pixel
point cloud from the item's views. A promptable segmentation model finds each
named object in each view, and the point cloud turns those masks into world
positions. Those positions are then laid out on the same 10x10 bird's-eye grid the
annotated map uses.

What comes out has the same shape as the annotated map, so the egocentric-map,
reasoning and prompt stages run on it unchanged.
"""

import json
import os
import re
from typing import Any

#: Words that appear in the object-name metadata but name a direction or a size
#: rather than an object, so they must not become segmentation prompts.
_NOT_OBJECT_NAMES: frozenset[str] = frozenset({
    "none", "left", "right", "up", "down", "front", "back",
    "face", "middle", "little", "heavy", "heavey", "total",
})

#: Where the annotated map is kept once the estimated one replaces it.
FIELD_ANNOTATED_COGMAP = "annotated_cogmap"


def object_names(item: dict) -> list[str]:
    """Return the objects to look for in the item's images.

    The benchmark already names the objects in a scene, so they are read from the
    annotation rather than detected. Each setting stores them differently, and all
    three shapes occur in the released splits:

    ``among``
        ``meta_info[0]`` is the list of names.
    ``rotation``
        ``meta_info`` is itself the list of names.
    ``around``
        ``meta_info[1][1]`` is the list of names; ``meta_info[0]`` holds the view
        count and size words, which are not objects.

    Args:
        item: Raw benchmark item.

    Returns:
        Object names, de-duplicated, in annotation order. Empty if the metadata
        holds none in any of the three shapes.
    """
    meta = item.get("meta_info")
    if not isinstance(meta, list) or not meta:
        return []

    def strings(value: object) -> list[str]:
        if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            return list(value)
        return []

    # Tried in the order that puts the most specific shape first, so that an
    # "around" item is not misread as an "among" one.
    candidates = (
        strings(meta[1][1]) if len(meta) > 1 and isinstance(meta[1], list)
        and len(meta[1]) > 1 else []
    )
    candidates = candidates or strings(meta[0]) or strings(meta)

    names: list[str] = []
    for name in candidates:
        cleaned = name.strip()
        if cleaned and cleaned.lower() not in _NOT_OBJECT_NAMES and cleaned not in names:
            names.append(cleaned)
    return names


def resolve_image_paths(item: dict, image_root: str) -> list[str]:
    """Return the item's image paths, made absolute against the image root."""
    images = item.get("images")
    if not isinstance(images, list):
        return []
    return [
        path if os.path.isabs(path) else os.path.join(image_root, path)
        for path in images
        if isinstance(path, str) and path.strip()
    ]


def _view_word(question: str) -> str:
    """Return the word the question uses for a viewpoint, ``Image`` or ``View``."""
    lowered = (question or "").lower()
    if "image" in lowered:
        return "Image"
    return "View" if "view" in lowered else "Image"


def align_view_names(
    cogmap: dict,
    question: str,
    reconstructed: list[str] | None = None,
    item_order: list[str] | None = None,
) -> dict:
    """Give the map's viewpoints the names the question will use to refer to them.

    Two things have to be corrected. The estimator names its viewpoints "Image k",
    but a question phrased with "view" has to be able to refer to them. And the
    estimator reorders the images before reconstructing, putting the front view
    first, so its k counts positions in *that* order while the question counts
    positions in the item's own order.

    Args:
        cogmap: Estimated cognitive map.
        question: Full question text.
        reconstructed: Image paths in the order the estimator used them.
        item_order: Image paths in the item's own order. Given both, viewpoint
            numbers are mapped back to this order; without them the estimator's
            numbering is kept.

    Returns:
        The map with its viewpoint names rewritten.

    Raises:
        ValueError: If a reconstructed path is not one of the item's, which would
            mean the two lists describe different items.
    """
    views = cogmap.get("views")
    if not isinstance(views, list):
        return cogmap

    renumber: dict[int, int] = {}
    if reconstructed and item_order:
        for position, path in enumerate(reconstructed, start=1):
            if path not in item_order:
                raise ValueError(f"reconstructed image is not one of the item's: {path}")
            renumber[position] = item_order.index(path) + 1

    word = _view_word(question)
    renamed = []
    for view in views:
        if not isinstance(view, dict):
            continue
        entry = dict(view)
        number = re.search(r"\d+", str(entry.get("name", "")))
        if number:
            found = int(number.group(0))
            entry["name"] = f"{word} {renumber.get(found, found)}"
        renamed.append(entry)
    # Sorted so the map lists the viewpoints in the item's own order.
    renamed.sort(key=lambda entry: int(re.search(r"\d+", entry["name"]).group(0)))
    return {**cogmap, "views": renamed}


def _jsonable(value: Any) -> Any:
    """Convert a value to something ``json.dump`` accepts, arrays included."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def add_pseudo_cogmap(
    item: dict,
    vggt_checkpoint: str,
    sam3_root: str,
    image_root: str = "data",
    sam3_checkpoint: str = "",
    max_views: int = 4,
    confidence_threshold: float = 0.50,
    keep_debug: bool = False,
) -> dict:
    """Replace an item's cognitive map with one estimated from its images.

    The annotated map, if the item has one, is kept under
    :data:`FIELD_ANNOTATED_COGMAP` so the two can be compared.

    Args:
        item: Benchmark item, modified in place.
        vggt_checkpoint: Path to the 3D foundation model's weights.
        sam3_root: Path to the segmentation model's source tree.
        image_root: Directory the benchmark images were extracted into.
        sam3_checkpoint: Segmentation weights, or empty to fetch them from the
            model hub.
        max_views: How many of the item's views to reconstruct from.
        confidence_threshold: Minimum segmentation confidence to accept a mask.
        keep_debug: Also record the estimator's diagnostics on the item, which are
            large but useful when a map comes out wrong.

    Returns:
        The same item, with ``grounded_cogmap`` replaced.

    Raises:
        ValueError: If the item names no objects or no images, since neither the
            geometry nor the segmentation step can run without them.
    """
    from .cogmap import PseudoCogmapConfig, build_pseudo_cogmap
    from .vggt import VGGTTopdownConfig, vggt_run_predictions

    names = object_names(item)
    if not names:
        raise ValueError(f"{item.get('id', '?')}: no object names in meta_info")
    image_paths = resolve_image_paths(item, image_root)
    if not image_paths:
        raise ValueError(f"{item.get('id', '?')}: no images")

    predictions = vggt_run_predictions(
        image_paths,
        cfg=VGGTTopdownConfig(
            ckpt_path=vggt_checkpoint,
            # Only the unused debug renderer writes here, but the field is required.
            output_dir=os.path.join(os.getcwd(), "vggt_renders"),
            max_input_views=max(1, max_views),
        ),
    )
    selected = predictions.get("_selected_image_paths")
    view_count = len(selected) if isinstance(selected, list) else min(len(image_paths), max_views)

    _, meta = build_pseudo_cogmap(
        predictions,
        cfg=PseudoCogmapConfig(
            device="cuda",
            object_names=tuple(names),
            max_views=max(1, view_count),
            det_backend="sam3_image",
            sam3_root=sam3_root,
            sam3_checkpoint=sam3_checkpoint,
            sam3_conf_threshold=confidence_threshold,
            sample_id=str(item.get("id", "")),
        ),
    )

    cogmap = meta.get("cogmap") if isinstance(meta, dict) else None
    if not isinstance(cogmap, dict):
        raise ValueError(f"{item.get('id', '?')}: the estimator returned no map")
    if not cogmap.get("objects"):
        # The estimator reports "no object found in any view" as an empty map
        # rather than an error. Letting it through would put a target in the
        # training data that marks every option wrong and then states the answer.
        raise ValueError(
            f"{item.get('id', '?')}: no object was found in any view, so the map is empty"
        )
    cogmap = align_view_names(
        cogmap,
        str(item.get("question", "")),
        reconstructed=selected if isinstance(selected, list) else None,
        item_order=image_paths,
    )

    if "grounded_cogmap" in item and FIELD_ANNOTATED_COGMAP not in item:
        item[FIELD_ANNOTATED_COGMAP] = item["grounded_cogmap"]
    item["grounded_cogmap"] = json.dumps(cogmap, ensure_ascii=False, indent=2)
    if keep_debug:
        item["pseudo_cogmap_debug"] = _jsonable(meta.get("pseudo_debug"))
    if isinstance(selected, list):
        item["pseudo_cogmap_views"] = selected
    return item


def render_cogmap(cogmap: dict, output_path: str, size: int = 512) -> str:
    """Draw a cognitive map as a bird's-eye PNG, for inspecting a bad map.

    Args:
        cogmap: A cognitive map.
        output_path: Where to write the image.
        size: Image side length in pixels.

    Returns:
        The path written.
    """
    from .render import BEVCogMapRenderConfig, render_bev_cogmap_image

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    render_bev_cogmap_image(cogmap, cfg=BEVCogMapRenderConfig(render_size=size)).save(output_path)
    return output_path


def parse_cogmap(item: dict) -> dict | None:
    """Return an item's cognitive map as a dictionary, or ``None`` if unparseable."""
    try:
        parsed = json.loads(item.get("grounded_cogmap", ""))
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
