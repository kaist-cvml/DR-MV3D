"""Grid geometry: facings, turns, moves, and egocentric directions."""

import pytest

from drmv3d.scaffold.spatial import (
    compute_ego_direction,
    get_facing_axes,
    move_camera,
    rotate_facing,
)


@pytest.mark.parametrize(
    "facing,action,expected",
    [
        ("up", "TURN_RIGHT_90", "right"),
        ("right", "TURN_RIGHT_90", "down"),
        ("up", "TURN_LEFT_90", "left"),
        ("left", "TURN_LEFT_90", "down"),
        ("up", "TURN_180", "down"),
        ("left", "TURN_180", "right"),
        ("up", "MOVE_FORWARD", "up"),
    ],
)
def test_rotate_facing(facing, action, expected):
    assert rotate_facing(facing, action) == expected


def test_four_right_turns_return_to_start():
    facing = "up"
    for _ in range(4):
        facing = rotate_facing(facing, "TURN_RIGHT_90")
    assert facing == "up"


@pytest.mark.parametrize(
    "facing,action,expected",
    [
        ("up", "MOVE_FORWARD", [5, 4]),
        ("up", "MOVE_BACK", [5, 6]),
        ("up", "MOVE_RIGHT", [6, 5]),
        ("up", "MOVE_LEFT", [4, 5]),
        ("right", "MOVE_FORWARD", [6, 5]),
        ("right", "MOVE_RIGHT", [5, 6]),
        ("down", "MOVE_FORWARD", [5, 6]),
        ("left", "MOVE_FORWARD", [4, 5]),
    ],
)
def test_move_camera(facing, action, expected):
    assert move_camera([5, 5], facing, action) == expected


def test_move_then_reverse_returns_to_start():
    for facing in ("up", "down", "left", "right"):
        moved = move_camera([5, 5], facing, "MOVE_FORWARD")
        assert move_camera(moved, facing, "MOVE_BACK") == [5, 5]


@pytest.mark.parametrize(
    "target,facing,expected",
    [
        ([5, 3], "up", "forward"),
        ([5, 7], "up", "back"),
        ([7, 5], "up", "right"),
        ([3, 5], "up", "left"),
        ([7, 3], "up", "forward-right"),
        ([3, 7], "up", "back-left"),
        # Turning the camera right turns the whole frame with it.
        ([5, 3], "right", "left"),
        ([7, 5], "right", "forward"),
        ([5, 3], "down", "back"),
        ([5, 3], "left", "right"),
    ],
)
def test_compute_ego_direction(target, facing, expected):
    assert compute_ego_direction(target, [5, 5], facing) == expected


def test_object_at_camera_has_no_direction():
    assert compute_ego_direction([5, 5], [5, 5], "up") == "at_camera"


def test_unknown_facing_is_reported_not_guessed():
    assert compute_ego_direction([5, 3], [5, 5], "sideways") == "unknown"


def test_facing_axes_are_perpendicular():
    for facing in ("up", "down", "left", "right"):
        axes = get_facing_axes(facing)
        fx, fy = axes["forward"]
        rx, ry = axes["right"]
        assert fx * rx + fy * ry == 0, f"{facing} axes are not perpendicular"
