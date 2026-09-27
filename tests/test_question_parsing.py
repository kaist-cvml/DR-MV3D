"""Recovering structure from a benchmark question."""

import pytest

from drmv3d.scaffold.question_parsing import (
    detect_question_type,
    extract_options,
    format_answer_option,
    get_answer_option_text,
    is_spatial_yesno,
    object_name_matches,
    parse_action_sequence,
    parse_mental_turn,
    parse_start_view_index,
)

QUESTION = (
    "Based on these four images showing the same scene from different viewpoints "
    "(front, left, back, and right): From the viewpoint presented in image 2, "
    "which object is to my right? A. TV B. Wooden dining table "
    "C. Light purple sofa D. Brown curtains and windows"
)


def test_extract_options():
    assert extract_options(QUESTION) == {
        "A": "TV",
        "B": "Wooden dining table",
        "C": "Light purple sofa",
        "D": "Brown curtains and windows",
    }


def test_extract_options_without_an_option_list():
    assert extract_options("Is the sofa behind me") == {}


def test_answer_option_text_drops_the_label():
    assert get_answer_option_text(QUESTION, "C") == "Light purple sofa"


def test_answer_option_keeps_the_label():
    assert format_answer_option(QUESTION, "C") == "C. Light purple sofa"


def test_answer_option_rejects_a_letter_the_question_does_not_offer():
    with pytest.raises(ValueError):
        format_answer_option(QUESTION, "F")


def test_start_view_index():
    assert parse_start_view_index(QUESTION) == 2
    assert parse_start_view_index("From image 4, what do I see?") == 4
    assert parse_start_view_index("Which object is on the left?") is None


@pytest.mark.parametrize(
    "clause,expected",
    [
        ("then I turn right, what do I see?", ["TURN_RIGHT_90"]),
        ("then I turn 90 degrees to the left, what do I see?", ["TURN_LEFT_90"]),
        ("then I turn 180 degrees around, what do I see?", ["TURN_180"]),
        ("then I turn right and move forward, what do I see?",
         ["TURN_RIGHT_90", "MOVE_FORWARD"]),
        ("what do I see?", []),
    ],
)
def test_parse_action_sequence(clause, expected):
    assert parse_action_sequence(clause) == expected


def test_movement_outside_the_hypothetical_is_not_an_instruction():
    # The preamble describes how the photographs were taken, not an instruction.
    assert parse_action_sequence("The camera moved forward between shots. What is left?") == []


def test_parse_mental_turn():
    assert parse_mental_turn("and turn 90 degrees to the right?") == "90 degrees to the right"
    assert parse_mental_turn("what is behind me?") == "none"


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Which object is closer to me?", "closer"),
        ("Which direction did the camera move?", "direction"),
        ("Which object is nearest the sofa?", "nearest"),
        ("Am I able to see the TV?", "visibility"),
        ("Is the sofa behind me? A. Yes B. No", "spatial"),
    ],
)
def test_detect_question_type(question, expected):
    assert detect_question_type(question) == expected


def test_yes_no_question_is_spatial_not_visibility():
    # "behind me" is a spatial relation even though it is answered Yes or No.
    question = "Is the sofa behind me? A. Yes B. No"
    assert detect_question_type(question) == "spatial"
    assert is_spatial_yesno(question)


def test_object_names_match_across_independent_wording():
    assert object_name_matches("Light purple sofa", "light purple sofa")
    assert object_name_matches("sofa", "Light purple sofa")
    assert not object_name_matches("TV", "sofa")
    assert not object_name_matches("", "sofa")
