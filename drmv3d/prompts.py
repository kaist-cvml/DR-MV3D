"""Turning a scaffolded item into the prompt and target the model is trained on.

The prompt is the task description, the cognitive-map instruction built by
:mod:`drmv3d.scaffold.cogmap`, and the question. The target is the four blocks
the model must produce, in order: the allocentric map, the per-view egocentric
maps, the reasoning chain, and the answer.

Both strings are released artifacts, so their wording and whitespace are fixed.
"""

from .scaffold.question_parsing import format_answer_option

#: Header that separates the question from the task description.
QUESTION_HEADER = "[Question]\n"

#: Placeholder in :data:`INSTRUCTION` for the per-item cognitive-map instruction.
COGMAP_INSTRUCTION_PLACEHOLDER = "<cogmap_gen_instruction>"

#: The task description. It states the four output blocks, how to derive the
#: egocentric maps, and the JSON schema of the egocentric-map block.
INSTRUCTION = '''[Task]
Given multiple images showing a 3D scene from different viewpoints, reason carefully about the spatial layout and answer the question.

[Answer Instruction]
Output 4 blocks in order:
   1. <cogmap>: Bird's-eye map with all object positions and view positions/facings.
   2. <egomap>: Multi-view ego-centric maps — a JSON array with one ego map per view.
      Each ego map includes "view", "position", "facing", "is_primary", per-object "ego_direction".
      The primary view (question-referenced, after any actions) also includes "answer_object" and "answer_visible_in_view".
   3. <think>: 4-part reasoning with multi-view ego maps:
      Part 1 — Scene layout: list all objects with allocentric positions, then all views.
        Scene: obj1 [x1,y1], obj2 [x2,y2], ...
        Views: Image 1 [x,y] facing=dir, Image 2 [x,y] facing=dir, ...
      Part 2 — Primary view selection: identify the question-relevant viewpoint.
        If actions exist: Selected: Image X [pos] facing=dir → action → [eff_pos] facing=eff_dir
        Then axis mapping: Axes: forward=..., right=...
        If no actions: Selected: Image X [pos] facing=dir
      Part 3 — Multi-view ego maps: compute ego-centric layout from EACH view.
        For the primary view (after any actions):
          Ego from [eff_pos] facing=eff_dir (primary, after action):
          obj Δ[dx,dy]=ego_dir [Image X]
        For each secondary view:
          Ego from Image Y [pos] facing=dir:
          obj Δ[dx,dy]=ego_dir [Image X]
        This provides complete spatial context from all perspectives.
      Part 4 — Elimination: using the PRIMARY ego map directions, check each option A/B/C/D.
        target=dir valid={...}
        A. name=ego_dir [Image X] ✓/✗
   4. <answer>: Final answer as "X. option_text".
Format: '<cogmap>...</cogmap><egomap>...</egomap><think>...</think><answer>X. option_text</answer>'
Make sure `X. option_text` matches exactly one of the options in the question.

<cogmap_gen_instruction>

[Multi-View Ego Map Instruction]
Generate ego-centric maps from the cognitive map for ALL views in the scene.
The primary view (question-referenced) uses the effective pose (after any rotation/translation actions).
Secondary views use their original poses from the cognitive map.
Each ego map computes ego_direction for all objects using Δ=[obj_x−cam_x, obj_y−cam_y] projected onto forward/right axes.

[View Arrangement Reference]
The question preamble describes the camera arrangement:
- 4-view scenes: "front, left, back, and right" → Image 1=front, Image 2=left, Image 3=back, Image 4=right
- 3-view scenes: view labels are stated in the question (e.g., "back, left, and right")
- Rotation scenes: all images are from the same position with different facing directions.
  After rotation, the effective facing may or may not match an existing image.
  Use the position and effective facing for ego-centric computation — do NOT assume it matches any image.

[Rotation Axis Rules]
When the camera facing changes (via turn/rotation), the coordinate axes rotate:
- facing=up:    forward=[-y] (north), right=[+x] (east)
- facing=right: forward=[+x] (east),  right=[+y] (south)
- facing=down:  forward=[+y] (south), right=[-x] (west)
- facing=left:  forward=[-x] (west),  right=[-y] (north)
Always state "Axes: forward=..., right=..." after rotation to make the mapping explicit.

[Rules]
1. Grid: [0,0]=top-left, [9,9]=bottom-right.
2. Parse the viewpoint from the question; apply actions to get the effective camera pose.
3. Compute "ego_direction" using Δ=[obj_x−cam_x, obj_y−cam_y], project onto forward/right axes.
4. Build ego maps for ALL views, not just the primary view.
5. Use the PRIMARY ego map for final answer elimination.
6. The final answer must match one of the provided multiple choice options.

[JSON Schema — Multi-View EgoMap Array]
[
  {
    "view": "Image k",
    "position": [x, y],
    "facing": "up|right|down|left",
    "is_primary": true,
    "actions": ["turn right 90"],
    "answer_object": "ObjectName",
    "answer_visible_in_view": "Image k",
    "objects": [
      {"name": "...", "position": [x, y], "ego_direction": "forward|back|left|right|forward-left|..."},
      ...
    ]
  },
  {
    "view": "Image m",
    "position": [x, y],
    "facing": "up|right|down|left",
    "is_primary": false,
    "objects": [...]
  },
  ...
]
'''


def build_prompt(item: dict) -> str:
    """Build the prompt shown to the model.

    Args:
        item: Scaffolded item, which must carry the cognitive-map instruction.

    Returns:
        The full prompt, ending with the ``[Answer]`` header that cues
        generation.
    """
    instruction = INSTRUCTION.replace(
        COGMAP_INSTRUCTION_PLACEHOLDER, item.get("cogmap_instruction", "")
    )
    return "\n".join([instruction, QUESTION_HEADER + item.get("question", ""), "\n[Answer]"])


def build_target(item: dict) -> str:
    """Build the four-block target the model is trained to produce.

    Args:
        item: Scaffolded item carrying the cognitive map, the egocentric maps and
            the reasoning chain.

    Returns:
        The target string.

    Raises:
        ValueError: If the ground-truth letter names no option in the question.
    """
    answer = format_answer_option(item.get("question", ""), item.get("gt_answer", ""))
    return (
        f"<cogmap>{item.get('grounded_cogmap', '')}</cogmap>"
        f"<egomap>{item.get('grounded_egomap', '')}</egomap>"
        f"<think>{item.get('reasoning_chain', '')}</think>"
        f"<answer>{answer}</answer>"
    )


def add_prompt(item: dict) -> dict:
    """Add the prompt and the target to a scaffolded item, in place.

    Args:
        item: Scaffolded item.

    Returns:
        The same item, with ``input_prompt`` and ``grounded_output`` added.
    """
    item["input_prompt"] = build_prompt(item)
    item["grounded_output"] = build_target(item)
    return item
