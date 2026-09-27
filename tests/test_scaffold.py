"""The scaffold stages, end to end on a synthetic scene."""

import json
import random

import pytest

from drmv3d.constants import SCAFFOLD_SEED
from drmv3d.prompts import build_prompt, build_target
from drmv3d.scaffold.cogmap import add_cogmap, build_cogmap, detect_setting, format_cogmap
from drmv3d.scaffold.pipeline import add_scaffold, add_scaffold_to_all


@pytest.fixture
def rng() -> random.Random:
    return random.Random(SCAFFOLD_SEED)


def test_detect_setting():
    assert detect_setting("among_group001_q1") == "among"
    assert detect_setting("around_abc_q2") == "around"
    assert detect_setting("rotation_group001_q3") == "rotation"
    assert detect_setting("mystery_group001") is None


def test_among_cogmap_places_five_objects_and_four_views(among_item):
    cogmap_str, objects, oriented = build_cogmap(among_item, quiet=True)
    cogmap = json.loads(cogmap_str)
    assert len(cogmap["objects"]) == 5
    assert len(cogmap["views"]) == 4
    assert objects == among_item["meta_info"][0]
    # Only the two objects the annotation orients get a facing.
    assert oriented == ["black sneaker", "light purple sofa"]
    assert [entry["name"] for entry in cogmap["objects"] if "facing" in entry] == oriented


def test_among_cameras_look_outward_from_inside_the_ring(among_item):
    cogmap = json.loads(build_cogmap(among_item, quiet=True)[0])
    views = {view["name"]: view for view in cogmap["views"]}
    # Image 1 was shot from the front, so it stands below centre looking up.
    assert views["Image 1"]["position"] == [5, 6]
    assert views["Image 1"]["facing"] == "up"
    # Image 3 was shot from the back, so it stands above centre looking down.
    assert views["Image 3"]["position"] == [5, 4]
    assert views["Image 3"]["facing"] == "down"


def test_rotation_cameras_share_one_cell(rotation_item):
    cogmap = json.loads(build_cogmap(rotation_item)[0])
    assert len({tuple(view["position"]) for view in cogmap["views"]}) == 1
    assert [view["facing"] for view in cogmap["views"]] == ["left", "up", "right"]


def test_rotation_annotates_no_object_facings(rotation_item):
    assert build_cogmap(rotation_item)[2] == []


def test_object_count_mismatch_is_rejected(among_item):
    among_item["meta_info"][0] = among_item["meta_info"][0][:4]
    with pytest.raises(AssertionError):
        build_cogmap(among_item, quiet=True)


def test_unknown_setting_is_rejected(among_item):
    among_item["id"] = "mystery_group001_q1"
    with pytest.raises(ValueError, match="no known setting"):
        build_cogmap(among_item, quiet=True)


def test_format_cogmap_round_trips():
    cogmap = {
        "objects": [{"name": "sofa", "position": [5, 5], "facing": "down"}],
        "views": [{"name": "Image 1", "position": [5, 6], "facing": "up"}],
    }
    text = format_cogmap(cogmap)
    assert json.loads(text) == cogmap
    # One entry per line, so a map stays readable inside a prompt.
    entry_lines = [line for line in text.splitlines() if line.startswith('    {')]
    assert len(entry_lines) == 2
    assert entry_lines[0] == '    {"name": "sofa", "position": [5, 5], "facing": "down"}'


def test_cogmap_instruction_names_every_object(among_item, rng):
    add_cogmap(among_item, rng, quiet=True)
    instruction = among_item["cogmap_instruction"]
    for name in among_item["meta_info"][0]:
        assert name in instruction
    # The two oriented objects are asked for a facing; the others are not.
    assert "determine their facing direction" in instruction
    assert "black sneaker" in instruction.split("4. For objects [")[1]


def test_scaffold_does_not_reorder_the_annotation(among_item, rng):
    original = list(among_item["meta_info"][0])
    add_cogmap(among_item, rng, quiet=True)
    assert among_item["meta_info"][0] == original


def test_full_scaffold_adds_every_field(among_item, rng):
    add_scaffold(among_item, rng, quiet=True)
    for field in ("grounded_cogmap", "cogmap_instruction", "grounded_egomap", "reasoning_chain"):
        assert among_item[field], f"{field} is empty"


def test_egomap_has_one_entry_per_view_with_exactly_one_primary(among_item, rng):
    add_scaffold(among_item, rng, quiet=True)
    egomap = json.loads(among_item["grounded_egomap"])
    assert isinstance(egomap, list)
    assert len(egomap) == 4
    assert sum(bool(entry.get("is_primary")) for entry in egomap) == 1
    primary = next(entry for entry in egomap if entry["is_primary"])
    # The question reasons from image 2.
    assert primary["view"] == "Image 2"


def test_reasoning_chain_reaches_the_ground_truth_answer(among_item, rng):
    add_scaffold(among_item, rng, quiet=True)
    assert among_item["reasoning_chain"].rstrip().endswith("C. Light purple sofa")


def test_target_holds_the_four_blocks_in_order(among_item, rng):
    add_scaffold(among_item, rng, quiet=True)
    target = build_target(among_item)
    positions = [target.index(f"<{tag}>") for tag in ("cogmap", "egomap", "think", "answer")]
    assert positions == sorted(positions)
    assert target.endswith("<answer>C. Light purple sofa</answer>")


def test_prompt_carries_the_instruction_and_the_question(among_item, rng):
    add_scaffold(among_item, rng, quiet=True)
    prompt = build_prompt(among_item)
    assert among_item["cogmap_instruction"] in prompt
    assert among_item["question"] in prompt
    assert "<cogmap_gen_instruction>" not in prompt
    assert prompt.rstrip().endswith("[Answer]")


def test_scaffolding_a_file_is_reproducible(among_item, rotation_item):
    items = [among_item, rotation_item]
    first = add_scaffold_to_all([dict(item) for item in items], quiet=True)
    second = add_scaffold_to_all([dict(item) for item in items], quiet=True)
    assert [item["cogmap_instruction"] for item in first] == [
        item["cogmap_instruction"] for item in second
    ]
