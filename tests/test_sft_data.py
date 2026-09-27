"""Converting prompts into training conversations."""

import pytest

from drmv3d.grpo.data import reference_view_trajectory, to_rl_sample
from drmv3d.training.sft_data import IMAGE_KEY, to_conversation, to_conversations


def prompt_item() -> dict:
    return {
        "id": "among_group001_q1",
        "images": ["a.jpg", "b.jpg"],
        "input_prompt": "[Task]\nthe task\n[Question]\nthe question? A. Yes B. No\n\n[Answer]",
        "grounded_output": "<cogmap>{}</cogmap><egomap>[]</egomap><think>t</think><answer>A. Yes</answer>",
        "gt_answer": "A",
        "grounded_cogmap": '{"objects": [], "views": []}',
        "grounded_egomap": '[{"view": "Image 1", "is_primary": true, '
                           '"answer_visible_in_view": "Image 2", "objects": []}]',
    }


def test_one_placeholder_per_image():
    conversation = to_conversation(prompt_item())
    human = conversation["conversations"][0]["value"]
    assert human.startswith("<image>\n<image>\n")
    assert human.count("<image>") == 2
    assert conversation[IMAGE_KEY] == ["a.jpg", "b.jpg"]


def test_paths_go_under_the_key_the_trainer_reads():
    # Under any other key the trainer takes its text-only path, sets no pixel
    # values, and trains without ever seeing the images.
    assert IMAGE_KEY == "image"
    assert IMAGE_KEY in to_conversation(prompt_item())


def test_the_assistant_turn_is_the_target():
    conversation = to_conversation(prompt_item())
    assert conversation["conversations"][1] == {
        "from": "gpt",
        "value": prompt_item()["grounded_output"],
    }


def test_images_must_be_a_list():
    item = prompt_item()
    item["images"] = "a.jpg"
    with pytest.raises(TypeError):
        to_conversation(item)


def test_a_missing_field_is_reported_not_silently_dropped():
    item = prompt_item()
    del item["grounded_output"]
    with pytest.raises(KeyError):
        to_conversation(item)


def test_conversations_keep_file_order():
    items = [prompt_item(), {**prompt_item(), "images": ["c.jpg"]}]
    assert [len(c[IMAGE_KEY]) for c in to_conversations(items)] == [2, 1]


def test_reference_trajectory_comes_from_the_primary_egomap():
    assert reference_view_trajectory(prompt_item()) == ("Image 1", "Image 2")


def test_reference_trajectory_is_empty_without_an_egomap():
    assert reference_view_trajectory({"grounded_egomap": ""}) == ("", "")


def test_rl_sample_carries_every_reference_signal():
    row = to_rl_sample(prompt_item(), 0, "data")
    extra = row["extra_info"]
    assert row["reward_model"] == {"style": "rule", "ground_truth": "A"}
    assert extra["reference_selected_view"] == "Image 1"
    assert extra["reference_answer_view"] == "Image 2"
    assert extra["reference_cogmap"] == prompt_item()["grounded_cogmap"]
    assert extra["num_images"] == 2


def test_rl_sample_makes_image_paths_absolute():
    # The trainer's workers do not share this process's working directory.
    row = to_rl_sample(prompt_item(), 0, "data")
    assert all(image["image"].startswith("/") for image in row["images"])
