"""The dense reward and the pieces it is built from."""

import json

import pytest

from drmv3d.grpo.cogmap_similarity import (
    SAME_CELL,
    compare_cogmaps,
    normalise_facing,
    relative_direction,
)
from drmv3d.grpo.parsing import (
    extract_block,
    parse_json_relaxed,
    predicted_view_trajectory,
    primary_egomap,
    view_number,
)
from drmv3d.grpo.reward import (
    KEY_REFERENCE_ANSWER_VIEW,
    KEY_REFERENCE_COGMAP,
    KEY_REFERENCE_SELECTED_VIEW,
    METRIC_KEYS,
    WEIGHT_ANSWER,
    answer_reward,
    format_reward,
    local_reward,
    score_trajectory,
)

REFERENCE_COGMAP = {
    "objects": [
        {"name": "sneaker", "position": [5, 5], "facing": "left"},
        {"name": "sofa", "position": [5, 8], "facing": "down"},
        {"name": "curtains", "position": [2, 5]},
        {"name": "tv", "position": [5, 2]},
    ],
    "views": [
        {"name": "Image 1", "position": [5, 6], "facing": "up"},
        {"name": "Image 2", "position": [4, 5], "facing": "right"},
    ],
}

EGOMAP = [
    {"view": "Image 1", "position": [5, 6], "facing": "up", "is_primary": False, "objects": []},
    {
        "view": "Image 2",
        "position": [4, 5],
        "facing": "right",
        "is_primary": True,
        "answer_object": "sofa",
        "answer_visible_in_view": "Image 1",
        "objects": [{"name": "sofa", "position": [5, 8], "ego_direction": "forward-right"}],
    },
]


def good_response() -> str:
    return (
        f"<cogmap>{json.dumps(REFERENCE_COGMAP)}</cogmap>"
        f"<egomap>{json.dumps(EGOMAP)}</egomap>"
        f"<think>Image 2 faces right, so the sofa is forward-right.</think>"
        f"<answer>C. Light purple sofa</answer>"
    )


def reference() -> dict:
    return {
        KEY_REFERENCE_COGMAP: json.dumps(REFERENCE_COGMAP),
        KEY_REFERENCE_SELECTED_VIEW: "Image 2",
        KEY_REFERENCE_ANSWER_VIEW: "Image 1",
    }


# --- parsing ---------------------------------------------------------------

def test_extract_block():
    assert extract_block("<think>abc</think>", "think") == "abc"
    assert extract_block("<think>abc</think>", "answer") is None


def test_parse_json_survives_what_a_learning_policy_writes():
    assert parse_json_relaxed('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_relaxed("Here it is: {'a': 1} done") == {"a": 1}
    assert parse_json_relaxed("[1, 2]") == [1, 2]
    assert parse_json_relaxed("not json at all") is None
    assert parse_json_relaxed("") is None


def test_primary_egomap_prefers_the_flagged_entry():
    assert primary_egomap(EGOMAP)["view"] == "Image 2"


def test_primary_egomap_falls_back_to_the_first_entry():
    unflagged = [{"view": "Image 3", "objects": []}, {"view": "Image 4", "objects": []}]
    assert primary_egomap(unflagged)["view"] == "Image 3"


def test_primary_egomap_accepts_the_single_view_form():
    assert primary_egomap({"selected_view": "Image 1"})["selected_view"] == "Image 1"


def test_view_numbers_compare_across_naming():
    assert view_number("Image 2") == view_number("View 2") == "2"
    assert view_number("nothing") is None


def test_predicted_trajectory_reads_the_multi_view_block():
    assert predicted_view_trajectory(good_response()) == ("Image 2", "Image 1")


def test_predicted_trajectory_is_empty_without_an_egomap_block():
    assert predicted_view_trajectory("<answer>A. Yes</answer>") == ("", "")


# --- similarity ------------------------------------------------------------

def test_identical_maps_score_one():
    result = compare_cogmaps(REFERENCE_COGMAP, REFERENCE_COGMAP)
    assert result.valid
    assert result.overall == pytest.approx(1.0)
    assert result.coverage == pytest.approx(1.0)


def test_translating_a_whole_map_changes_nothing():
    shifted = {
        key: [{**entry, "position": [entry["position"][0] + 3, entry["position"][1] + 3]}
              for entry in entries]
        for key, entries in REFERENCE_COGMAP.items()
    }
    assert compare_cogmaps(shifted, REFERENCE_COGMAP).overall == pytest.approx(1.0)


def test_omitting_objects_costs_score():
    partial = {"objects": REFERENCE_COGMAP["objects"][:2], "views": REFERENCE_COGMAP["views"]}
    result = compare_cogmaps(partial, REFERENCE_COGMAP)
    assert result.coverage < 1.0
    assert result.overall < 1.0, "an almost-empty map must not score as a perfect one"


def test_wrong_view_facing_costs_only_the_facing_part():
    wrong = {
        "objects": REFERENCE_COGMAP["objects"],
        "views": [{**view, "facing": "down"} for view in REFERENCE_COGMAP["views"]],
    }
    result = compare_cogmaps(wrong, REFERENCE_COGMAP)
    assert result.directional == pytest.approx(1.0)
    assert result.facing == pytest.approx(0.0)
    assert result.overall == pytest.approx(0.7)


def test_malformed_maps_are_invalid_not_zero_scored():
    assert not compare_cogmaps({"objects": "not a list"}, REFERENCE_COGMAP).valid
    assert not compare_cogmaps(REFERENCE_COGMAP, {"objects": [], "views": []}).valid
    assert not compare_cogmaps("not a map", REFERENCE_COGMAP).valid


def test_relative_direction():
    assert relative_direction((5, 5), (5, 2)) == "up"
    assert relative_direction((5, 5), (5, 8)) == "down"
    assert relative_direction((5, 5), (8, 5)) == "right"
    assert relative_direction((5, 5), (2, 5)) == "left"
    assert relative_direction((5, 5), (5, 5)) == SAME_CELL


def test_facing_aliases_are_folded():
    assert normalise_facing("North") == "up"
    assert normalise_facing(" EAST ") == "right"
    assert normalise_facing("up") == "up"
    assert normalise_facing("sideways") is None
    assert normalise_facing(None) is None


# --- reward terms ----------------------------------------------------------

def test_answer_reward():
    assert answer_reward(good_response(), "C") == 1.0
    assert answer_reward(good_response(), "A") == 0.0
    assert answer_reward("", "C") == 0.0


def test_format_reward_is_full_for_a_well_formed_response():
    assert format_reward(good_response()) == pytest.approx(1.0)


def test_format_reward_is_partial_when_a_block_is_missing():
    without_cogmap = good_response().split("</cogmap>")[1]
    partial = format_reward(without_cogmap)
    assert 0.0 < partial < 1.0


def test_format_reward_credits_the_multi_view_egomap():
    # The block is a list of per-view maps, which is the released output format.
    with_egomap = f"<egomap>{json.dumps(EGOMAP)}</egomap>"
    without_egomap = "<egomap>{}</egomap>"
    assert format_reward(with_egomap) > format_reward(without_egomap)


def test_local_reward_counts_matched_trajectory_steps():
    score, selected, answer = local_reward(good_response(), reference())
    assert (score, selected, answer) == (1.0, True, True)


def test_local_reward_is_half_when_one_step_is_wrong():
    wrong = good_response().replace('"answer_visible_in_view": "Image 1"',
                                    '"answer_visible_in_view": "Image 9"')
    score, selected, answer = local_reward(wrong, reference())
    assert score == pytest.approx(0.5)
    assert selected and not answer


def test_local_reward_is_zero_without_a_reference_trajectory():
    assert local_reward(good_response(), {}) == (0.0, False, False)


def test_view_naming_does_not_affect_the_local_reward():
    renamed = good_response().replace("Image 2", "View 2").replace("Image 1", "View 1")
    assert local_reward(renamed, reference())[0] == pytest.approx(1.0)


# --- the whole reward ------------------------------------------------------

def test_a_perfect_response_earns_every_term():
    result = score_trajectory(good_response(), "C", reference())
    assert set(METRIC_KEYS) <= set(result)
    assert result["answer_reward"] == pytest.approx(WEIGHT_ANSWER)
    assert result["score"] == pytest.approx(
        result["global_reward"] + result["local_reward"]
        + result["answer_reward"] + result["format_reward"]
    )


def test_every_metric_is_present_even_for_an_empty_response():
    result = score_trajectory("", "C", reference())
    assert set(METRIC_KEYS) <= set(result)
    assert result["score"] == pytest.approx(0.0)


def test_the_answer_term_outweighs_all_shaping():
    right_answer_only = "<answer>C. Light purple sofa</answer>"
    everything_but_the_answer = good_response().replace(
        "<answer>C. Light purple sofa</answer>", "<answer>A. Tv</answer>"
    )
    assert (score_trajectory(right_answer_only, "C", reference())["score"]
            > score_trajectory(everything_but_the_answer, "C", reference())["score"])


def test_an_unparseable_map_is_worse_than_no_map():
    no_block = "<answer>C. Light purple sofa</answer>"
    broken = "<cogmap>{not json</cogmap><answer>C. Light purple sofa</answer>"
    assert (score_trajectory(broken, "C", reference())["global_reward"]
            < score_trajectory(no_block, "C", reference())["global_reward"])


def test_reward_works_without_any_reference_signals():
    # Falls back to answer and format only, rather than failing.
    result = score_trajectory(good_response(), "C")
    assert result["global_reward"] == 0.0
    assert result["local_reward"] == 0.0
    assert result["answer_reward"] == pytest.approx(WEIGHT_ANSWER)
