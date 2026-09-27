"""Fixtures shared by the tests: one small synthetic scene in the benchmark schema."""

import json

import pytest

#: A four-view ``among`` scene: five objects ringing the centre, cameras inside
#: the ring looking outward. Written by hand so the tests need no downloaded data.
AMONG_ITEM = {
    "id": "among_group001_q1_5_2",
    "category": ["among", "O-O", "meanwhile", "self"],
    "type": 1,
    "meta_info": [
        ["black sneaker", "light purple sofa", "brown curtains", "tv", "dining table"],
        ["left", "face", None, None, None],
    ],
    "images": [
        "other_all_image/among/scene001/front_000.jpg",
        "other_all_image/among/scene001/left_001.jpg",
        "other_all_image/among/scene001/back_002.jpg",
        "other_all_image/among/scene001/right_003.jpg",
    ],
    "question": (
        "Based on these four images showing the same scene from different viewpoints "
        "(front, left, back, and right): From the viewpoint presented in image 2, "
        "which object is to my right? A. Tv B. Dining table C. Light purple sofa "
        "D. Brown curtains"
    ),
    "gt_answer": "C",
}

#: A three-view ``rotation`` scene: one camera turning on the spot.
ROTATION_ITEM = {
    "id": "rotation_group001_q3_6",
    "category": ["rotation", "O-O", "meanwhile", "self"],
    "type": "three_view",
    "meta_info": ["bookshelf", "window", "door"],
    "images": [
        "other_all_image/rotation/scene001/0.jpg",
        "other_all_image/rotation/scene001/1.jpg",
        "other_all_image/rotation/scene001/2.jpg",
    ],
    "question": (
        "Based on these three images taken from the same position: From the viewpoint "
        "presented in image 2, if I turn 90 degrees to the right, which object is in "
        "front of me? A. Bookshelf B. Window C. Door D. None of them"
    ),
    "gt_answer": "C",
}


@pytest.fixture
def among_item() -> dict:
    """A fresh copy of the synthetic ``among`` item."""
    return json.loads(json.dumps(AMONG_ITEM))


@pytest.fixture
def rotation_item() -> dict:
    """A fresh copy of the synthetic ``rotation`` item."""
    return json.loads(json.dumps(ROTATION_ITEM))
