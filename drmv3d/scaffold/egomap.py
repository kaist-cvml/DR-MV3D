"""Egocentric maps: the scene as it looks from a given viewpoint.

An allocentric cognitive map states where things are in the room. A question,
though, asks what is to *my* left or behind *me*, so the map has to be
re-expressed relative to a viewpoint and its facing. That conversion is what
this module performs, both for the viewpoint a question selects and for every
other viewpoint in the scene.
"""

import json

from .question_parsing import (
    describe_item_setting,
    extract_options,
    extract_view_from_question,
    find_object_by_name,
    find_view_by_index,
    get_answer_option_text,
    object_name_matches,
    parse_action_sequence,
    parse_mental_turn,
    parse_start_view_index,
)
from .spatial import compute_ego_direction, move_camera, rotate_facing

# Two criteria for "the camera can see it", used for different things and
# deliberately different. The strict one decides which image the reasoning cites as
# evidence for an option, where a thing off to the side is poor evidence. The wide
# one decides where the answer object is visible at all, and a real camera's field
# of view does extend to either side.

#: Strictly ahead of the camera.
FORWARD_DIRECTIONS: frozenset[str] = frozenset({"forward", "forward-left", "forward-right"})

#: Ahead of the camera or level with it, which is what a wide lens covers.
IN_FRAME_DIRECTIONS: frozenset[str] = FORWARD_DIRECTIONS | {"left", "right", "at_camera"}


def views_showing_object(
    cogmap: dict,
    answer_object_name: str,
) -> list[str]:
    """Return the viewpoints an object is visible from.

    Uses the wider of the two criteria, :data:`IN_FRAME_DIRECTIONS`, because the
    question is whether the object appears in that photograph at all.

    Args:
        cogmap: Cognitive map of the scene.
        answer_object_name: Name of the object to look for.

    Returns:
        Viewpoint names, in map order. Empty if the map does not hold the object.
    """
    objects = cogmap.get("objects", [])
    views = cogmap.get("views", [])
    answer_pos: list[int] | None = None
    for obj in objects:
        name = obj.get("name", "")
        if object_name_matches(name, answer_object_name):
            answer_pos = obj.get("position")
            break
    if answer_pos is None:
        return []

    visible_views = []
    for view in views:
        cam_pos = view.get("position", [5, 5])
        facing = view.get("facing", "up")
        if compute_ego_direction(answer_pos, cam_pos, facing) in IN_FRAME_DIRECTIONS:
            visible_views.append(view.get("name", ""))
    return visible_views


def format_primary_egomap(egomap: dict) -> str:
    """Serialise the selected viewpoint's egocentric map.

    Written by hand rather than with :func:`json.dumps` so the layout matches
    what the model is trained to produce, one object per line.

    Args:
        egomap: Egocentric map fields.

    Returns:
        A JSON object as a string.
    """
    result = "{\n"
    result += f'  "selected_view": "{egomap["selected_view"]}",\n'
    result += f'  "selected_view_position": {json.dumps(egomap["selected_view_position"])},\n'
    result += f'  "selected_view_facing": "{egomap["selected_view_facing"]}",\n'

    if egomap.get("actions"):
        result += f'  "actions": {json.dumps(egomap["actions"])},\n'

    if "answer_object" in egomap and egomap["answer_object"]:
        result += f'  "answer_object": "{egomap["answer_object"]}",\n'
    if "answer_visible_in_view" in egomap and egomap["answer_visible_in_view"]:
        result += f'  "answer_visible_in_view": "{egomap["answer_visible_in_view"]}",\n'

    result += '  "objects": [\n'
    for i, obj in enumerate(egomap["objects"]):
        result += '    ' + json.dumps(obj, ensure_ascii=False)
        if i < len(egomap["objects"]) - 1:
            result += ','
        result += '\n'
    result += '  ]'

    if "other_views" in egomap and egomap["other_views"]:
        result += ',\n  "other_views": [\n'
        for i, view in enumerate(egomap["other_views"]):
            result += '    ' + json.dumps(view, ensure_ascii=False)
            if i < len(egomap["other_views"]) - 1:
                result += ','
            result += '\n'
        result += '  ]'

    result += '\n}'
    return result


def build_primary_egomap(
    cogmap_str: str,
    question: str,
    gt_answer: str = "",
    item_id: str = "",
) -> tuple[str, dict]:
    """Build the egocentric map for the viewpoint a question selects.

    The question names a viewpoint and may state a hypothetical movement from
    it. Both are applied to obtain the effective pose, and every object is then
    re-expressed relative to that pose.

    Args:
        cogmap_str: Allocentric cognitive map, as a JSON string.
        question: Full question text.
        gt_answer: Ground-truth option letter, used to mark the answer object.
        item_id: Item identifier, used to recognise the benchmark setting.

    Returns:
        The egocentric map as a JSON string, and metadata about the selected
        viewpoint that later stages reuse.
    """
    meta: dict = {
        "selected_view_name": "", "selected_view_pos": [], "selected_view_facing": "",
        "original_pos": [], "original_facing": "",
        "actions": [], "has_actions": False,
        "answer_object": "", "answer_visible_in_view": "",
        "question_type": "",
    }

    try:
        cogmap = json.loads(cogmap_str)
    except json.JSONDecodeError:
        return "{}", meta

    objects = cogmap.get("objects", [])
    views = cogmap.get("views", [])
    if not views:
        return "{}", meta

    meta["question_type"] = describe_item_setting(item_id)

    # Find the view mentioned in the question
    vp_idx = parse_start_view_index(question)
    if vp_idx is not None:
        selected_view = find_view_by_index(views, vp_idx)
    else:
        selected_view = extract_view_from_question(question, views)
    if not selected_view:
        # A question that names no viewpoint reasons from the first one.
        selected_view = views[0]

    original_pos = list(selected_view.get("position", [5, 5]))
    original_facing = selected_view.get("facing", "up")
    view_name = selected_view.get("name", "Image 1")

    meta["selected_view_name"] = view_name
    meta["original_pos"] = original_pos
    meta["original_facing"] = original_facing

    # Parse and apply actions
    actions = parse_action_sequence(question)
    if not actions and meta["question_type"] == "rotation":
        turn_desc = parse_mental_turn(question)
        if turn_desc != 'none':
            if '90 degrees to the left' in turn_desc:
                actions = ["TURN_LEFT_90"]
            elif '90 degrees to the right' in turn_desc:
                actions = ["TURN_RIGHT_90"]
            elif '180 degrees around' in turn_desc:
                actions = ["TURN_180"]

    meta["actions"] = actions
    meta["has_actions"] = bool(actions)

    effective_pos = list(original_pos)
    effective_facing = original_facing
    for action in actions:
        if action.startswith("TURN"):
            effective_facing = rotate_facing(effective_facing, action)
        else:
            effective_pos = move_camera(effective_pos, effective_facing, action)

    meta["selected_view_pos"] = effective_pos
    meta["selected_view_facing"] = effective_facing

    # Compute ego_directions from effective camera pose
    ego_objects = []
    for obj in objects:
        obj_position = obj.get("position", [0, 0])
        ego_direction = compute_ego_direction(obj_position, effective_pos, effective_facing)
        ego_obj = {
            "name": obj.get("name", ""),
            "position": obj_position,
            "ego_direction": ego_direction,
        }
        if "facing" in obj:
            ego_obj["facing"] = obj["facing"]
        ego_objects.append(ego_obj)

    other_views = []
    for view in views:
        if view.get("name") == view_name:
            continue
        view_position = view.get("position", [0, 0])
        ego_direction = compute_ego_direction(view_position, effective_pos, effective_facing)
        other_views.append({
            "name": view.get("name", ""),
            "position": view_position,
            "facing": view.get("facing", "up"),
            "ego_direction": ego_direction,
        })

    # Build egomap dict
    action_descs = [a.replace("_", " ").lower() for a in actions]

    egomap_dict: dict = {
        "selected_view": view_name,
        "selected_view_position": effective_pos,
        "selected_view_facing": effective_facing,
        "objects": ego_objects,
    }

    if action_descs:
        egomap_dict["actions"] = action_descs

    if other_views:
        egomap_dict["other_views"] = other_views

    # answer_object and answer_visible_in_view
    answer_obj_name = get_answer_option_text(question, gt_answer) if gt_answer else None
    if answer_obj_name:
        egomap_dict["answer_object"] = answer_obj_name
        meta["answer_object"] = answer_obj_name
        visible = views_showing_object(cogmap, answer_obj_name)
        if visible:
            chosen = view_name if view_name in visible else visible[0]
            egomap_dict["answer_visible_in_view"] = chosen
            meta["answer_visible_in_view"] = chosen
        else:
            meta["answer_visible_in_view"] = ""
    else:
        meta["answer_object"] = answer_obj_name or ""
        meta["answer_visible_in_view"] = ""

    return format_primary_egomap(egomap_dict), meta


def compute_option_primary_images(
    options_detail: dict, per_view_objects: dict
) -> dict[str, str]:
    """Pick, for each answer option, the image that best shows it.

    An option is grounded in the first viewpoint whose field of view contains
    the matching object, so the written reasoning can point at a real image.

    Args:
        options_detail: Per-option text and egocentric direction.
        per_view_objects: Viewpoint name to the objects visible from it.

    Returns:
        Option label to image name; an empty string when no image shows it.
    """
    option_images: dict[str, str] = {}

    for label, det in options_detail.items():
        opt_text = det.get("text", "").strip().lower()
        if not opt_text:
            option_images[label] = ""
            continue

        found_image = ""
        for vn in sorted(per_view_objects.keys()):
            for vo in per_view_objects.get(vn, []):
                if object_name_matches(vo.get("name", ""), opt_text):
                    found_image = vn
                    break
            if found_image:
                break

        option_images[label] = found_image

    return option_images


def format_multi_view_egomap(
    multi_view_ego_maps: list[dict],
    egomap_meta: dict,
) -> str:
    """Serialise one egocentric map per viewpoint as a JSON array.

    The entry for the selected viewpoint additionally carries the answer object
    and the image that shows it.

    Args:
        multi_view_ego_maps: One egocentric map per viewpoint.
        egomap_meta: Metadata from :func:`build_primary_egomap`.

    Returns:
        A JSON array as a string.
    """
    answer_obj = egomap_meta.get("answer_object", "")
    answer_in_view = egomap_meta.get("answer_visible_in_view", "")

    result = "[\n"
    for idx, vmap in enumerate(multi_view_ego_maps):
        result += "  {\n"
        result += f'    "view": "{vmap["view"]}",\n'
        result += f'    "position": {json.dumps(vmap["position"])},\n'
        result += f'    "facing": "{vmap["facing"]}",\n'
        result += f'    "is_primary": {json.dumps(vmap["is_primary"])},\n'
        if vmap.get("actions"):
            result += f'    "actions": {json.dumps(vmap["actions"])},\n'
        if vmap["is_primary"] and answer_obj:
            result += f'    "answer_object": "{answer_obj}",\n'
        if vmap["is_primary"] and answer_in_view:
            result += f'    "answer_visible_in_view": "{answer_in_view}",\n'
        result += '    "objects": [\n'
        for i, obj in enumerate(vmap["objects"]):
            result += '      ' + json.dumps(obj, ensure_ascii=False)
            if i < len(vmap["objects"]) - 1:
                result += ','
            result += '\n'
        result += '    ]\n'
        result += '  }'
        if idx < len(multi_view_ego_maps) - 1:
            result += ','
        result += '\n'
    result += ']'
    return result


def build_multi_view_egomap(item: dict) -> tuple[str, dict]:
    """Describe the scene from every viewpoint, not just the selected one.

    The selected viewpoint uses the pose the question's movement leads to; the
    others keep their own. Giving the model all of them at once lets it
    cross-check a relation instead of trusting a single perspective.

    Args:
        item: Data item with a cognitive map, a question and an answer.

    Returns:
        The egocentric maps as a JSON array string, and the metadata the
        reasoning stage consumes. Both are empty if the item has no cognitive
        map, which is the only case this stage tolerates.
    """
    cogmap_str = item.get("grounded_cogmap", "")
    question = item.get("question", "")
    gt_answer = item.get("gt_answer", "")
    item_id = item.get("id", "")

    if not cogmap_str:
        return "", {}

    # The selected viewpoint gets the detailed treatment first.
    egomap_str, egomap_meta = build_primary_egomap(
        cogmap_str, question, gt_answer=gt_answer, item_id=item_id,
    )

    cogmap = json.loads(cogmap_str)
    objects = cogmap.get("objects", [])
    views = cogmap.get("views", [])

    # What each viewpoint can see: an object is in frame if it lies in that
    # camera's forward hemisphere.
    per_view_objects: dict = {}
    for v in views:
        vn = v.get("name", "")
        cam_pos = v.get("position", [5, 5])
        facing = v.get("facing", "up")
        visible = []
        for obj in objects:
            ego_dir = compute_ego_direction(obj.get("position", [0, 0]), cam_pos, facing)
            if ego_dir in FORWARD_DIRECTIONS:
                visible.append({"name": obj.get("name", ""), "ego_direction": ego_dir})
        per_view_objects[vn] = visible
    egomap_meta["per_view_objects"] = per_view_objects
    egomap_meta["cogmap_views"] = views

    # Scene objects (allocentric positions)
    scene_objects = []
    for obj in objects:
        scene_objects.append({
            "name": obj.get("name", ""),
            "position": obj.get("position", []),
        })
    egomap_meta["scene_objects"] = scene_objects

    # Primary view info
    view_name = egomap_meta.get("selected_view_name", "")
    effective_pos = egomap_meta.get("selected_view_pos", [5, 5])
    effective_facing = egomap_meta.get("selected_view_facing", "up")
    has_actions = egomap_meta.get("has_actions", False)

    # ── Build multi-view ego maps ────────────────────────────────────────────
    multi_view_ego_maps = []

    for v in views:
        vn = v.get("name", "")
        cam_pos = list(v.get("position", [5, 5]))
        facing = v.get("facing", "up")

        # For the primary view, use effective pose (after actions)
        if vn == view_name:
            ego_pos = list(effective_pos)
            ego_facing = effective_facing
            is_primary = True
        else:
            ego_pos = cam_pos
            ego_facing = facing
            is_primary = False

        # Compute ego objects from this view's pose
        ego_objs = []
        for obj in objects:
            obj_pos = obj.get("position", [0, 0])
            ego_dir = compute_ego_direction(obj_pos, ego_pos, ego_facing)
            ego_objs.append({
                "name": obj.get("name", ""),
                "position": obj_pos,
                "ego_direction": ego_dir,
            })

        # Compute other views from this perspective
        other_vs = []
        for ov in views:
            if ov.get("name") == vn:
                continue
            ov_pos = ov.get("position", [0, 0])
            ov_dir = compute_ego_direction(ov_pos, ego_pos, ego_facing)
            other_vs.append({
                "name": ov.get("name", ""),
                "position": ov_pos,
                "facing": ov.get("facing", "up"),
                "ego_direction": ov_dir,
            })

        view_ego_map = {
            "view": vn,
            "position": ego_pos,
            "facing": ego_facing,
            "is_primary": is_primary,
            "objects": ego_objs,
        }
        if has_actions and is_primary:
            action_descs = [a.replace("_", " ").lower() for a in egomap_meta.get("actions", [])]
            view_ego_map["actions"] = action_descs

        multi_view_ego_maps.append(view_ego_map)

    egomap_meta["multi_view_ego_maps"] = multi_view_ego_maps

    # Format multi-view egomap as JSON array
    egomap_json = format_multi_view_egomap(multi_view_ego_maps, egomap_meta)

    # Resolve each answer option to a map object and an egocentric direction.
    try:
        primary_egomap = json.loads(egomap_str) if egomap_str else {}
    except json.JSONDecodeError:
        primary_egomap = {}
    ego_objects = primary_egomap.get("objects", [])

    options_map = extract_options(question)
    options_detail: dict = {}
    for label, option_text in options_map.items():
        matched_object = find_object_by_name(objects, option_text)
        matched_ego_object = find_object_by_name(ego_objects, option_text)

        object_position = matched_object.get("position") if matched_object else None
        if object_position is not None:
            ego_direction = compute_ego_direction(
                object_position, effective_pos, effective_facing
            )
        elif matched_ego_object:
            ego_direction = matched_ego_object.get("ego_direction", "?")
        else:
            ego_direction = "?"

        options_detail[label] = {
            "text": option_text,
            "cogmap_position": object_position,
            "ego_direction": ego_direction,
        }

    egomap_meta["options_map"] = options_map
    egomap_meta["options_detail"] = options_detail
    egomap_meta["option_primary_images"] = compute_option_primary_images(
        options_detail,
        egomap_meta.get("per_view_objects", {}),
    )

    # When the question asks for a hypothetical turn in place, the resulting pose
    # may coincide with another viewpoint; name it so the reasoning can cite it.
    rotation_target_view = ""
    if has_actions:
        for view in views:
            if (view.get("name", "") != view_name
                    and view.get("position", []) == list(effective_pos)
                    and view.get("facing", "") == effective_facing):
                rotation_target_view = view.get("name", "")
                break
    egomap_meta["rotation_target_view"] = rotation_target_view

    return egomap_json, egomap_meta
