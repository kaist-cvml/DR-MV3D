"""The parts of the estimated-cognitive-map path that need no external models."""

import pytest

from drmv3d.pseudo.pipeline import align_view_names, object_names, resolve_image_paths


def test_object_names_for_an_among_item():
    # among puts the names in meta_info[0].
    item = {"meta_info": [["sneaker", "sofa", "curtains"], ["left", "face", None]]}
    assert object_names(item) == ["sneaker", "sofa", "curtains"]


def test_object_names_for_a_rotation_item():
    # rotation makes meta_info itself the list.
    assert object_names({"meta_info": ["bookshelf", "window", "door"]}) == [
        "bookshelf", "window", "door",
    ]


def test_object_names_for_an_around_item():
    # around puts the view count and size words in meta_info[0], and the names in
    # meta_info[1][1]. Reading the wrong slot finds no objects at all.
    item = {"meta_info": [[4, "None", "Heavey", "Little"],
                          [2, ["potted plant", "floor lamp"], ["None", "None"], []]]}
    assert object_names(item) == ["potted plant", "floor lamp"]


def test_direction_and_size_words_are_not_objects():
    item = {"meta_info": [["sofa", "left", "None", "Little", "tv"], [None] * 5]}
    assert object_names(item) == ["sofa", "tv"]


def test_duplicate_names_are_dropped():
    assert object_names({"meta_info": ["sofa", "sofa", "tv"]}) == ["sofa", "tv"]


def test_no_metadata_yields_no_objects():
    assert object_names({}) == []
    assert object_names({"meta_info": []}) == []


def test_image_paths_resolve_against_the_root():
    # A relative path is joined to the root; an absolute one is left alone.
    paths = resolve_image_paths({"images": ["a/b.jpg", "/tmp/c.jpg"]}, "data")
    assert paths == ["data/a/b.jpg", "/tmp/c.jpg"]


def cogmap() -> dict:
    return {
        "objects": [{"name": "sofa", "position": [5, 5]}],
        "views": [
            {"name": "Image 1", "position": [5, 6], "facing": "up"},
            {"name": "Image 2", "position": [4, 5], "facing": "right"},
            {"name": "Image 3", "position": [5, 4], "facing": "down"},
        ],
    }


def test_view_names_follow_the_word_the_question_uses():
    assert [v["name"] for v in align_view_names(cogmap(), "these views")["views"]] == [
        "View 1", "View 2", "View 3",
    ]
    assert [v["name"] for v in align_view_names(cogmap(), "these images")["views"]] == [
        "Image 1", "Image 2", "Image 3",
    ]


def test_reordering_for_reconstruction_does_not_renumber_the_views():
    # The estimator puts the front view first; the question still counts the
    # item's own order, so a viewpoint must keep the number the question uses.
    item_order = ["s/back.jpg", "s/front.jpg", "s/left.jpg"]
    reconstructed = ["s/front.jpg", "s/back.jpg", "s/left.jpg"]
    views = align_view_names(cogmap(), "these images", reconstructed, item_order)["views"]
    by_name = {v["name"]: v["position"] for v in views}
    # front was reconstructed first, but it is the item's second image.
    assert by_name["Image 2"] == [5, 6]
    assert by_name["Image 1"] == [4, 5]
    assert by_name["Image 3"] == [5, 4]


def test_views_are_listed_in_the_items_own_order():
    item_order = ["s/back.jpg", "s/front.jpg", "s/left.jpg"]
    reconstructed = ["s/front.jpg", "s/back.jpg", "s/left.jpg"]
    views = align_view_names(cogmap(), "these images", reconstructed, item_order)["views"]
    assert [v["name"] for v in views] == ["Image 1", "Image 2", "Image 3"]


def test_a_reconstructed_image_from_another_item_is_rejected():
    with pytest.raises(ValueError, match="not one of the item's"):
        align_view_names(cogmap(), "these images", ["other/x.jpg"], ["s/back.jpg"])


def test_without_the_orders_the_numbering_is_left_alone():
    views = align_view_names(cogmap(), "these images")["views"]
    assert [v["name"] for v in views] == ["Image 1", "Image 2", "Image 3"]
