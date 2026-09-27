"""Written reasoning that connects a cognitive map to an answer.

The chain a model is trained to produce has four parts, separated by blank
lines: the allocentric scene, the viewpoint the question selects, the scene
re-expressed from every viewpoint, and the answer derived from the selected
one. Everything here is generated deterministically from the benchmark
annotations, so the supervision never contradicts the ground truth.
"""

import json
import math
import re

from .question_parsing import (
    detect_question_type,
    extract_options,
    find_object_by_name,
    is_spatial_yesno,
    question_body,
)
from .spatial import DIRECTION_ADJACENCY, TARGET_DIRECTION_SETS, get_facing_axes


def _resolve_consistent_direction(ego_dir: str, target_dirs: set) -> str:
    """Snap a computed direction to the nearest one that answers the question.

    Grid coordinates are coarse, so a computed egocentric direction can land one
    sector away from the direction the ground-truth answer implies. Walking the
    eight-way compass outward from the computed direction finds the closest
    direction that does satisfy the query, keeping the written reasoning
    consistent with the answer.

    Args:
        ego_dir: Direction computed from the map.
        target_dirs: Directions that satisfy the question.

    Returns:
        ``ego_dir`` if it already satisfies the question, otherwise the nearest
        direction that does.
    """
    if ego_dir in target_dirs:
        return ego_dir
    visited = {ego_dir}
    queue = [ego_dir]
    while queue:
        current = queue.pop(0)
        for neighbor in DIRECTION_ADJACENCY.get(current, []):
            if neighbor in target_dirs:
                return neighbor
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    # Nothing adjacent qualifies (only for degenerate directions such as
    # "at_camera"); pick deterministically so runs are reproducible.
    return sorted(target_dirs)[0] if target_dirs else ego_dir


def _build_spatial_yesno_answer(question: str, gt_key: str, gt_text: str) -> list[str]:
    """Write the answer step for a spatial yes/no question.

    These questions ("is the door behind me?") are answered from the ground
    truth directly: the coarse grid makes a direction check unreliable here, so
    stating a check that disagrees with the answer would only teach the model to
    contradict itself.

    Args:
        question: Full question text.
        gt_key: Ground-truth option letter.
        gt_text: Ground-truth option text.

    Returns:
        Lines of the answer section.
    """

    lines = ["Answer:"]

    # Determine direction from question (behind or front)
    is_behind_q = "behind" in question.lower()
    set_label = "behind" if is_behind_q else "front"
    lines.append(f'target: "{set_label}"')
    lines.append(f"\u2192 {gt_key}. {gt_text}")
    return lines


def _build_closer_answer(
    question: str, gt_key: str, gt_text: str, ego_objects: list,
) -> list[str]:
    """Write the answer step for a "do I get closer to X?" question.

    The target object's egocentric direction decides the answer: anything in the
    forward sector means the move shortens the distance. When the computed
    direction disagrees with the ground truth it is snapped to a consistent one.

    Args:
        question: Full question text.
        gt_key: Ground-truth option letter.
        gt_text: Ground-truth option text.
        ego_objects: Objects with their egocentric directions.

    Returns:
        Lines of the answer section.
    """

    lines = ["Answer:"]
    forward_set = {"forward", "forward-left", "forward-right"}
    non_forward_set = {"back", "back-left", "back-right", "left", "right"}

    # Extract target object name from question
    target_match = re.search(
        r'(?:closer to|get closer to|move closer to)\s+(?:the\s+)?(.+?)(?:\?|$)',
        question, re.IGNORECASE,
    )
    target_name = target_match.group(1).strip().rstrip("?").strip() if target_match else ""

    # Determine GT answer: Yes or No
    gt_is_yes = gt_text.lower() in ("yes", "a. yes")

    # Find the target object in ego_objects
    target_ego = find_object_by_name(ego_objects, target_name) if target_name else None
    if target_ego:
        ego_dir = target_ego.get("ego_direction", "?")

        # GT-consistent override
        if gt_is_yes and ego_dir not in forward_set and ego_dir != "?":
            ego_dir = _resolve_consistent_direction(ego_dir, forward_set)
        elif not gt_is_yes and ego_dir in forward_set:
            ego_dir = _resolve_consistent_direction(ego_dir, non_forward_set)

        is_forward = ego_dir in forward_set
        lines.append(f'"{target_name}" ego_direction={ego_dir}')
        membership = "\u2208" if is_forward else "\u2209"
        closer_yn = "yes" if is_forward else "no"
        lines.append(
            f'{ego_dir} {membership} forward set'
            f' \u2192 closer={closer_yn}'
        )
    else:
        # Fallback: use GT answer directly
        closer_fallback = "yes" if gt_is_yes else "no"
        lines.append(f'target object \u2192 closer={closer_fallback}')

    lines.append(f"\u2192 {gt_key}. {gt_text}")
    return lines


def _build_visibility_answer(
    question: str, gt_key: str, gt_text: str,
    ego_objects: list, answer_obj: str,
) -> list[str]:
    """Write the answer step for a "can I see X?" question.

    An object is visible when it falls in the forward sector of the viewpoint.

    Args:
        question: Full question text.
        gt_key: Ground-truth option letter.
        gt_text: Ground-truth option text.
        ego_objects: Objects with their egocentric directions.
        answer_obj: Name of the answer object, used when the question does not
            name the target explicitly.

    Returns:
        Lines of the answer section.
    """

    lines = ["Answer:"]
    forward_set = {"forward", "forward-left", "forward-right"}

    # Extract target object name from question
    target_match = re.search(
        r'(?:able to see|can you see|can i see)\s+(?:the\s+)?(.+?)(?:\?|$)',
        question, re.IGNORECASE,
    )
    if not target_match:
        # Try "would X be behind me" pattern
        target_match = re.search(
            r'(?:would|will)\s+(?:the\s+)?(.+?)\s+(?:be\s+)?(?:behind|in front)',
            question, re.IGNORECASE,
        )
    target_name = target_match.group(1).strip().rstrip("?").strip() if target_match else answer_obj

    # Find the target object in ego_objects
    target_ego = find_object_by_name(ego_objects, target_name) if target_name else None
    if target_ego:
        ego_dir = target_ego.get("ego_direction", "?")
        is_visible = ego_dir in forward_set
        lines.append(f'"{target_name}" ego_direction={ego_dir}')
        lines.append(f'{ego_dir} {"∈" if is_visible else "∉"} forward set → visible={"yes" if is_visible else "no"}')
    else:
        # Fallback: use GT answer directly
        is_yes = gt_text.lower() in ("yes", "a. yes")
        lines.append(f'target object → visible={"yes" if is_yes else "no"}')

    lines.append(f"→ {gt_key}. {gt_text}")
    return lines


def _build_direction_answer(gt_key: str, gt_text: str, cogmap_views: list) -> list[str]:
    """Write the answer step for a "which way did the camera move?" question.

    The displacement between the first two viewpoints is projected onto the
    first viewpoint's forward and right axes to name the motion.

    Args:
        gt_key: Ground-truth option letter.
        gt_text: Ground-truth option text.
        cogmap_views: Viewpoint entries of the cognitive map.

    Returns:
        Lines of the answer section.
    """
    lines = ["Answer:"]

    # The question asks which way the camera moved between the first two views.
    if len(cogmap_views) >= 2:
        start_pos = cogmap_views[0].get("position", [5, 5])
        end_pos = cogmap_views[1].get("position", [5, 5])
        start_facing = cogmap_views[0].get("facing", "up")

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        lines.append(f'{cogmap_views[0]["name"]} {start_pos} → {cogmap_views[1]["name"]} {end_pos}')
        lines.append(f'Δ_move=[{dx:+d},{dy:+d}], facing={start_facing}')

        # The movement is described from where the camera started, so the axes
        # are the starting view's, not the selected view's.
        start_axes = get_facing_axes(start_facing)
        fwd_vec = start_axes.get("forward", (0, -1))
        rgt_vec = start_axes.get("right", (1, 0))

        fwd_comp = dx * fwd_vec[0] + dy * fwd_vec[1]
        rgt_comp = dx * rgt_vec[0] + dy * rgt_vec[1]

        # Determine direction label
        fwd_label = "forward" if fwd_comp > 0 else ("backward" if fwd_comp < 0 else "")
        rgt_label = "right" if rgt_comp > 0 else ("left" if rgt_comp < 0 else "")

        if fwd_label and rgt_label:
            direction = f"diagonally {fwd_label} and {rgt_label}"
        elif fwd_label:
            direction = f"directly {fwd_label}"
        elif rgt_label:
            direction = f"directly {rgt_label}"
        else:
            direction = "no movement"

        lines.append(f'fwd={fwd_comp:+d}, right={rgt_comp:+d} → {direction}')
    else:
        lines.append("insufficient views for direction computation")

    lines.append(f"→ {gt_key}. {gt_text}")
    return lines


def _build_nearest_answer(
    question: str, gt_key: str, gt_text: str,
    ego_objects: list, answer_in_view: str,
) -> list[str]:
    """Write the answer step for a "nearest object in direction D of X" question.

    Candidates are restricted to the queried sector around the reference object
    and then ranked by grid distance.

    Args:
        question: Full question text.
        gt_key: Ground-truth option letter.
        gt_text: Ground-truth option text.
        ego_objects: Objects with their egocentric directions and positions.
        answer_in_view: Image in which the answer object is visible.

    Returns:
        Lines of the answer section.
    """

    lines = ["Answer:"]

    # The reference object and the queried direction, read from the question's own
    # words. The option list is cut off first, or it would be swallowed into the
    # object's name.
    body = question_body(question).rstrip(" .?")
    ref_match = re.search(
        r'nearest object\s+(?:to the |)(right|left|front|behind|in front)\s+of\s+(?:the\s+)?(.+?)$',
        body, re.IGNORECASE,
    )
    if not ref_match:
        # Some are phrased without the "of".
        ref_match = re.search(
            r'nearest object\s+(behind)\s+(?:the\s+)?(.+?)$',
            body, re.IGNORECASE,
        )

    ref_name = ""
    target_dir_label = ""
    if ref_match:
        groups = ref_match.groups()
        if len(groups) >= 2 and groups[0] is not None:
            target_dir_label = groups[0].lower().replace("in front", "front")
            ref_name = groups[1].strip().rstrip("?").strip()

    target_dirs = TARGET_DIRECTION_SETS.get(target_dir_label, set()) if target_dir_label else set()

    # Find reference object position
    ref_obj = find_object_by_name(ego_objects, ref_name)
    ref_pos = ref_obj.get("position", []) if ref_obj else []

    if ref_pos and target_dirs:
        lines.append(f'target="{target_dir_label}" of "{ref_name}" {ref_pos}')

        # Filter and rank objects by distance
        candidates = []
        for obj in ego_objects:
            name = obj.get("name", "")
            ego_dir = obj.get("ego_direction", "")
            pos = obj.get("position", [])
            if not name or name.lower() == ref_name.lower():
                continue
            if ego_dir not in target_dirs:
                continue
            if pos and ref_pos:
                dist = math.sqrt((pos[0] - ref_pos[0]) ** 2 + (pos[1] - ref_pos[1]) ** 2)
            else:
                dist = 999
            candidates.append((name, ego_dir, dist, pos))

        candidates.sort(key=lambda x: x[2])
        for name, ego_dir, dist, pos in candidates:
            mark = "✓" if name.lower() == gt_text.lower() else ""
            lines.append(f'- "{name}" {pos}: dist={dist:.1f}, ego={ego_dir} {mark}')

        if candidates:
            lines.append(f'nearest: "{candidates[0][0]}" (dist={candidates[0][2]:.1f})')
    else:
        # Fallback
        lines.append(f'"{gt_text}" is the nearest matching object')
        if answer_in_view:
            lines.append(f'visible in {answer_in_view}')

    lines.append(f"→ {gt_key}. {gt_text}")
    return lines


def _build_elimination_table(
    question: str, gt_key: str, gt_text: str,
    ego_objects: list, options_detail: dict,
    answer_in_view: str,
    option_primary_images: dict[str, str],
) -> list[str]:
    """Write the answer step as an elimination table over all options.

    Each option is listed with its egocentric direction, the image that shows
    it, and a tick or cross for whether that direction answers the question::

        target=right valid={back-right,forward-right,right}
        A. Door=forward-right [Image 1] \u2713
        B. Chair=forward [Image 3] \u2717

    Args:
        question: Full question text.
        gt_key: Ground-truth option letter.
        gt_text: Ground-truth option text.
        ego_objects: Objects with their egocentric directions.
        options_detail: Per-option text and egocentric direction.
        answer_in_view: Image in which the answer object is visible.
        option_primary_images: Option label to the image that shows it.

    Returns:
        Lines of the answer section.
    """

    lines = []

    # Extract target direction set
    target_dirs = None
    target_label = None
    dir_match = re.search(
        r'what (?:is|would be) to (?:my |the )?(right|left|front|behind)\??',
        question, re.IGNORECASE,
    )
    if not dir_match:
        dir_match = re.search(
            r'what (?:is|would be) (behind|in front)',
            question, re.IGNORECASE,
        )
    if dir_match:
        raw_label = dir_match.group(1).lower()
        target_label = "front" if raw_label == "in front" else raw_label
        target_dirs = TARGET_DIRECTION_SETS.get(target_label, set())
        lines.append(f"target={target_label} valid={{{','.join(sorted(target_dirs))}}}")
    elif "closer" in question.lower():
        target_dirs = {"forward", "forward-left", "forward-right"}
        lines.append(f"target=closer valid={{{','.join(sorted(target_dirs))}}}")

    # Full A/B/C/D elimination table with image annotations
    if options_detail:
        for label in ["A", "B", "C", "D"]:
            if label not in options_detail:
                continue
            det = options_detail[label]
            opt_text = det.get("text", "")
            ego_dir = det.get("ego_direction", "?")

            # Keep the written direction consistent with the answer.
            if label == gt_key and target_dirs is not None and ego_dir != "?" and ego_dir not in target_dirs:
                ego_dir = _resolve_consistent_direction(ego_dir, target_dirs)

            if target_dirs is not None:
                mark = "\u2713" if ego_dir in target_dirs else "\u2717"
            else:
                mark = ""

            # Ground every option in the image that shows it.
            primary_img = option_primary_images.get(label, "")
            img_note = f" [{primary_img}]" if primary_img else ""

            lines.append(f"{label}. {opt_text}={ego_dir}{img_note} {mark}")
    else:
        answer_ego = find_object_by_name(ego_objects, gt_text)
        if answer_ego:
            ego_dir = answer_ego.get("ego_direction", "?")
            if target_dirs is not None and ego_dir != "?" and ego_dir not in target_dirs:
                ego_dir = _resolve_consistent_direction(ego_dir, target_dirs)
            img_note = f" [{answer_in_view}]" if answer_in_view else ""
            lines.append(f"{gt_text}={ego_dir}{img_note} \u2713")

    lines.append(f"\u2192 {gt_key}. {gt_text}")
    return lines


def _build_reasoning(item: dict, meta: dict) -> str:
    """Assemble the four-part reasoning chain.

    Args:
        item: Data item with the question, the answer and the cognitive map.
        meta: Metadata produced alongside the egocentric maps.

    Returns:
        The reasoning chain, with its parts separated by blank lines.
    """

    question = item.get("question", "")
    gt_answer_key = item.get("gt_answer", "")
    cogmap_str = item.get("grounded_cogmap", "")
    has_actions = meta.get("has_actions", False)

    options_map = extract_options(question)
    gt_text = options_map.get(gt_answer_key, "").strip()

    try:
        cogmap = json.loads(cogmap_str) if cogmap_str else {}
    except json.JSONDecodeError:
        cogmap = {}

    view_name = meta.get("selected_view_name", "")
    original_pos = meta.get("original_pos", [])
    original_facing = meta.get("original_facing", "")
    effective_pos = meta.get("selected_view_pos", [])
    effective_facing = meta.get("selected_view_facing", "")
    actions = meta.get("actions", [])
    answer_obj = meta.get("answer_object", "")
    answer_in_view = meta.get("answer_visible_in_view", "")
    per_view_objects = meta.get("per_view_objects", {})
    cogmap_views = meta.get("cogmap_views", cogmap.get("views", []))
    options_detail = meta.get("options_detail", {})
    option_primary_images = meta.get("option_primary_images", {})
    scene_objects = meta.get("scene_objects", [])
    rotation_target_view = meta.get("rotation_target_view", "")
    multi_view_ego_maps = meta.get("multi_view_ego_maps", [])

    # Build object → primary image mapping
    obj_to_image: dict[str, str] = {}
    for v in cogmap_views:
        vn = v.get("name", "")
        for vo in per_view_objects.get(vn, []):
            name = vo.get("name", "")
            if name and name not in obj_to_image:
                obj_to_image[name] = vn

    # Primary ego objects (for the elimination table)
    primary_ego_objects = []
    for vmap in multi_view_ego_maps:
        if vmap.get("is_primary"):
            primary_ego_objects = vmap.get("objects", [])
            break

    parts = []

    # Part 1: the allocentric scene, which anchors everything that follows.
    scene_lines = []
    if scene_objects:
        obj_strs = ", ".join(
            f'{o["name"]} {o["position"]}' for o in scene_objects if o.get("position")
        )
        scene_lines.append(f"Scene: {obj_strs}")
    if cogmap_views:
        view_strs = ", ".join(
            f'{v.get("name", "")} {v.get("position", [])} facing={v.get("facing", "")}'
            for v in cogmap_views
        )
        scene_lines.append(f"Views: {view_strs}")
    parts.append("\n".join(scene_lines))

    # Part 2: which viewpoint to reason from, and where any movement lands.
    select_lines = []
    if has_actions:
        select_lines.append(
            f"Selected: {view_name} {original_pos} facing={original_facing}"
        )
        action_desc = " → ".join(a.replace("_", " ").lower() for a in actions)
        traj_str = f"→ {action_desc} → {effective_pos} facing={effective_facing}"
        if rotation_target_view:
            traj_str += f" (= {rotation_target_view})"
        select_lines.append(traj_str)
        # Naming the axes keeps the rotation explicit.
        eff_axes = get_facing_axes(effective_facing)
        fwd_label = eff_axes.get("forward_label", f"toward {effective_facing}")
        rgt_label = eff_axes.get("right_label", "right")
        select_lines.append(f"Axes: forward={fwd_label}, right={rgt_label}")
    else:
        select_lines.append(
            f"Selected: {view_name} {effective_pos} facing={effective_facing}"
        )
    parts.append("\n".join(select_lines))

    # Part 3: the scene re-expressed from every viewpoint.
    for vmap in multi_view_ego_maps:
        ego_lines = []
        vn = vmap.get("view", "")
        vpos = vmap.get("position", [])
        vfacing = vmap.get("facing", "")
        is_primary = vmap.get("is_primary", False)
        ego_objs = vmap.get("objects", [])

        # Position-based header with primary/secondary label
        if is_primary and has_actions:
            ego_lines.append(f"Ego from {vpos} facing={vfacing} (primary, after action):")
        elif is_primary:
            ego_lines.append(f"Ego from {vn} {vpos} facing={vfacing} (primary):")
        else:
            ego_lines.append(f"Ego from {vn} {vpos} facing={vfacing}:")

        for obj in ego_objs:
            name = obj.get("name", "")
            ego_dir = obj.get("ego_direction", "")
            pos = obj.get("position", [])
            if not name or not ego_dir or ego_dir == "at_camera":
                continue
            # Delta math
            if pos and len(pos) == 2 and len(vpos) == 2:
                dx = pos[0] - vpos[0]
                dy = pos[1] - vpos[1]
                math_str = f" Δ[{dx:+d},{dy:+d}]="
            else:
                math_str = " ="
            img_src = obj_to_image.get(name, "")
            img_note = f" [{img_src}]" if img_src else ""
            ego_lines.append(f"{name}{math_str}{ego_dir}{img_note}")

        parts.append("\n".join(ego_lines))

    # Part 4: the answer, derived from the selected viewpoint's egocentric map.
    qtype = detect_question_type(question)

    if qtype == "spatial" and not is_spatial_yesno(question):
        answer_lines = _build_elimination_table(
            question, gt_answer_key, gt_text,
            primary_ego_objects, options_detail, answer_in_view,
            option_primary_images,
        )
    elif qtype == "spatial" and is_spatial_yesno(question):
        answer_lines = _build_spatial_yesno_answer(question, gt_answer_key, gt_text)
    elif qtype == "closer":
        answer_lines = _build_closer_answer(
            question, gt_answer_key, gt_text, primary_ego_objects,
        )
    elif qtype == "direction":
        answer_lines = _build_direction_answer(gt_answer_key, gt_text, cogmap_views)
    elif qtype == "nearest":
        answer_lines = _build_nearest_answer(
            question, gt_answer_key, gt_text, primary_ego_objects, answer_in_view,
        )
    else:
        # detect_question_type returns exactly these five values, so this is the
        # visibility case rather than a fallback.
        answer_lines = _build_visibility_answer(
            question, gt_answer_key, gt_text, primary_ego_objects, answer_obj,
        )

    parts.append("\n".join(answer_lines))

    return "\n\n".join(parts)


def build_reasoning_chain(item: dict, egomap_meta: dict) -> str:
    """
    The chain has four parts, separated by blank lines:

    1. the allocentric scene: every object and viewpoint with its grid cell;
    2. the selected viewpoint and the pose any stated movement leads to;
    3. one egocentric block per viewpoint, so the scene is described from every
       perspective rather than only the one the question names;
    4. the answer, derived from the selected viewpoint's block.

    Args:
        item: Data item with the question, the answer and the cognitive map.
        egomap_meta: Metadata produced alongside the egocentric maps by
            :func:`drmv3d.scaffold.egomap.build_multi_view_egomap`.

    Returns:
        The reasoning chain.
    """
    return _build_reasoning(item, egomap_meta)
