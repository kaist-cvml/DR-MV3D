"""Scoring model responses."""

import pytest

from drmv3d.eval.accuracy import format_scores, score_responses, setting_of
from drmv3d.eval.answers import extract_answer_letter


def test_answer_letter_comes_from_the_answer_block():
    # The reasoning mentions every option; only the answer block decides.
    response = (
        "<think>A. Tv=forward-left ✗ B. Table=forward ✗ C. Sofa=right ✓ D. Curtains=back ✗</think>"
        "<answer>C. Light purple sofa</answer>"
    )
    assert extract_answer_letter(response) == "C"


def test_answer_letter_falls_back_to_the_last_option_mentioned():
    assert extract_answer_letter("After checking each one, the answer is D. Curtains") == "D"


def test_bare_letter_is_accepted():
    assert extract_answer_letter("<answer>B</answer>") == "B"


def test_response_naming_no_option_is_unparseable():
    assert extract_answer_letter("I cannot tell from these images.") is None
    assert extract_answer_letter("") is None


def test_setting_of():
    assert setting_of("around_abc_q2_7") == "around"
    assert setting_of("among_group693_q1") == "among"
    assert setting_of("rotation_group000_q3") == "rotation"
    assert setting_of("mystery_1") == "other"


def test_score_responses_counts_per_setting():
    items = [
        {"id": "among_1", "gt_answer": "A", "answer": "<answer>A. Yes</answer>"},
        {"id": "among_2", "gt_answer": "B", "answer": "<answer>A. Yes</answer>"},
        {"id": "rotation_1", "gt_answer": "C", "answer": "<answer>C. Door</answer>"},
    ]
    scores = score_responses(items)
    assert scores["total"] == 3
    assert scores["correct"] == 2
    assert scores["accuracy"] == pytest.approx(2 / 3)
    assert scores["settings"]["among"]["accuracy"] == pytest.approx(0.5)
    assert scores["settings"]["rotation"]["accuracy"] == pytest.approx(1.0)
    assert "around" not in scores["settings"], "settings with no items are left out"


def test_unparseable_responses_are_counted_and_scored_wrong():
    items = [{"id": "among_1", "gt_answer": "A", "answer": "I cannot tell."}]
    scores = score_responses(items)
    assert scores["unparseable"] == 1
    assert scores["correct"] == 0


def test_items_without_ground_truth_are_skipped():
    items = [
        {"id": "among_1", "gt_answer": "A", "answer": "<answer>A. Yes</answer>"},
        {"id": "among_2", "answer": "<answer>A. Yes</answer>"},
    ]
    assert score_responses(items)["total"] == 1


def test_empty_input_does_not_divide_by_zero():
    scores = score_responses([])
    assert scores == {"total": 0, "correct": 0, "accuracy": 0.0, "unparseable": 0, "settings": {}}


def test_format_scores_renders_every_setting_and_an_overall_row():
    items = [
        {"id": "among_1", "gt_answer": "A", "answer": "<answer>A. Yes</answer>"},
        {"id": "rotation_1", "gt_answer": "C", "answer": "<answer>C. Door</answer>"},
    ]
    table = format_scores(score_responses(items))
    assert "among" in table
    assert "rotation" in table
    assert "overall" in table
    assert "100.00%" in table
