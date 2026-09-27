"""The dense reward: four verifiable terms scoring one sampled trajectory.

Optimising answer correctness alone gives one bit of feedback per rollout, which
is thin for a four-stage reasoning chain. Each term here scores a different stage,
and each is computed from the benchmark annotations rather than a learned model,
so the signal is verifiable:

``global``
    How well the predicted allocentric map matches a reference map of the scene.
``local``
    How much of the reference view trajectory the response actually reasoned from.
``answer``
    Whether the chosen option is right. This is the terminal objective.
``format``
    Whether the four blocks are present and parseable.

The weights below make the answer term dominant, so shaping guides the policy
without overriding what it is being trained to do.
"""

import os
from typing import Any

from .cogmap_similarity import CogmapSimilarity, compare_cogmaps
from .parsing import extract_block, parse_json_relaxed, predicted_view_trajectory, view_number

# ---------------------------------------------------------------------------
# Term weights, the lambdas of the reward equation
# ---------------------------------------------------------------------------
#: Weight of the global map term. Its own value is a similarity in [0, 1], or
#: minus :data:`INVALID_COGMAP_PENALTY` for a map block that does not parse.
WEIGHT_GLOBAL = float(os.getenv("DRMV3D_WEIGHT_GLOBAL", "1.0"))

#: Weight of the local trajectory term. Its own value is a fraction in [0, 1].
WEIGHT_LOCAL = float(os.getenv("DRMV3D_WEIGHT_LOCAL", "1.0"))

#: Weight of the answer term, the largest so that shaping cannot outvote it.
WEIGHT_ANSWER = float(os.getenv("DRMV3D_WEIGHT_ANSWER", "5.0"))

#: Weight of the format term. Its own value is a fraction in [0, 1].
WEIGHT_FORMAT = float(os.getenv("DRMV3D_WEIGHT_FORMAT", "1.0"))

#: Subtracted from the global term when the response opens a map block but emits
#: something unparseable, which is worse than omitting the block.
INVALID_COGMAP_PENALTY = float(os.getenv("DRMV3D_INVALID_COGMAP_PENALTY", "0.1"))

#: How the format term splits over the four blocks. The answer block weighs most
#: because it is the one the answer term depends on.
FORMAT_WEIGHTS: dict[str, float] = {
    "cogmap": 0.2,
    "egomap": 0.2,
    "think": 0.2,
    "answer": 0.4,
}

#: Reference-signal keys the data-preparation step writes into each sample.
KEY_REFERENCE_COGMAP = "reference_cogmap"
KEY_REFERENCE_SELECTED_VIEW = "reference_selected_view"
KEY_REFERENCE_ANSWER_VIEW = "reference_answer_view"

#: Metric names always present in the returned dictionary. The training framework
#: requires every sample in a batch to return the same keys.
METRIC_KEYS: tuple[str, ...] = (
    "score",
    "global_reward",
    "local_reward",
    "answer_reward",
    "format_reward",
    "selected_view_match",
    "answer_view_match",
    "cogmap_comparable",
    "cogmap_coverage",
    "cogmap_similarity",
    "cogmap_directional_similarity",
    "cogmap_facing_similarity",
)


def answer_reward(response: str, ground_truth: str) -> float:
    """Score whether the response chose the right option.

    Args:
        response: The model's full response.
        ground_truth: The correct option letter.

    Returns:
        ``1.0`` if the letters match, else ``0.0``.
    """
    from ..eval.answers import extract_answer_letter

    predicted = extract_answer_letter(response)
    if not predicted or not ground_truth:
        return 0.0
    return 1.0 if predicted.upper() == ground_truth.strip().upper() else 0.0


def format_reward(response: str) -> float:
    """Score how much of the required output format the response produced.

    A block earns its weight when it is present and, for the two JSON blocks, when
    it parses and carries the keys that make it usable.

    Args:
        response: The model's full response.

    Returns:
        A fraction in ``[0, 1]``.
    """
    if not response:
        return 0.0

    earned = 0.0

    if extract_answer_present(response):
        earned += FORMAT_WEIGHTS["answer"]

    think = extract_block(response, "think")
    if think and think.strip():
        earned += FORMAT_WEIGHTS["think"]

    cogmap = parse_json_relaxed(extract_block(response, "cogmap"))
    if isinstance(cogmap, dict) and "objects" in cogmap and "views" in cogmap:
        earned += FORMAT_WEIGHTS["cogmap"]

    egomap = parse_json_relaxed(extract_block(response, "egomap"))
    if _egomap_is_usable(egomap):
        earned += FORMAT_WEIGHTS["egomap"]

    return min(1.0, earned)


def extract_answer_present(response: str) -> bool:
    """Test whether the response has an answer block naming an option.

    Args:
        response: The model's full response.

    Returns:
        ``True`` if an ``<answer>`` block names an option letter.
    """
    from ..eval.answers import extract_answer_letter

    block = extract_block(response, "answer")
    return bool(block) and extract_answer_letter(block) is not None


def _egomap_is_usable(egomap: Any) -> bool:
    """Test whether an egocentric-map block carries the fields the reward reads."""
    if isinstance(egomap, dict):
        # The single-view form earlier checkpoints emit.
        return "objects" in egomap
    if not isinstance(egomap, list) or not egomap:
        return False
    entries = [entry for entry in egomap if isinstance(entry, dict)]
    return bool(entries) and all("view" in entry and "objects" in entry for entry in entries)


def local_reward(response: str, reference: dict) -> tuple[float, bool, bool]:
    """Score how much of the reference view trajectory the response reasoned from.

    The reference trajectory has two steps, both derived from the benchmark
    annotation: the viewpoint the question reasons from, and the viewpoint that
    shows the answer object. The score is the fraction matched.

    Args:
        response: The model's full response.
        reference: Sample metadata carrying the reference viewpoints.

    Returns:
        The fraction matched, and whether each of the two steps matched. The
        fraction is ``0.0`` when the metadata names no reference trajectory.
    """
    wanted_selected = reference.get(KEY_REFERENCE_SELECTED_VIEW) or ""
    wanted_answer = reference.get(KEY_REFERENCE_ANSWER_VIEW) or ""
    steps = [step for step in (wanted_selected, wanted_answer) if step]
    if not steps:
        return 0.0, False, False

    got_selected, got_answer = predicted_view_trajectory(response)
    selected_match = bool(
        wanted_selected and view_number(got_selected) == view_number(wanted_selected)
    )
    answer_match = bool(
        wanted_answer and view_number(got_answer) == view_number(wanted_answer)
    )

    matched = int(selected_match) + int(answer_match)
    return matched / len(steps), selected_match, answer_match


def global_reward(response: str, reference: dict) -> tuple[float, CogmapSimilarity | None]:
    """Score the predicted allocentric map against the reference map.

    Args:
        response: The model's full response.
        reference: Sample metadata carrying the reference map.

    Returns:
        The score and the similarity breakdown. The score is ``0.0`` with no
        breakdown when the metadata carries no reference map. Opening a map block
        and emitting something unparseable costs :data:`INVALID_COGMAP_PENALTY`.
    """
    reference_map = _parse_reference_cogmap(reference.get(KEY_REFERENCE_COGMAP))
    if reference_map is None:
        return 0.0, None

    block = extract_block(response, "cogmap")
    predicted_map = parse_json_relaxed(block)
    if not isinstance(predicted_map, dict):
        return (-INVALID_COGMAP_PENALTY if block is not None else 0.0), None

    similarity = compare_cogmaps(predicted_map, reference_map)
    if not similarity.valid:
        return -INVALID_COGMAP_PENALTY, similarity
    return similarity.overall, similarity


def _parse_reference_cogmap(value: Any) -> dict | None:
    """Return the reference map as a dictionary, accepting a JSON string."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = parse_json_relaxed(value)
        return parsed if isinstance(parsed, dict) else None
    return None


def score_trajectory(
    response: str,
    ground_truth: str,
    reference: dict | None = None,
) -> dict[str, Any]:
    """Score one sampled response with the full dense reward.

    Args:
        response: The model's full response.
        ground_truth: The correct option letter.
        reference: Sample metadata with the reference map and view trajectory.
            Terms whose metadata is absent contribute nothing, so this can be
            omitted to fall back to answer and format only.

    Returns:
        The weighted total under ``score``, each term's own contribution, and the
        diagnostics listed in :data:`METRIC_KEYS`. Every key is always present.
    """
    reference = reference or {}

    fmt = format_reward(response)
    ans = answer_reward(response, ground_truth)
    loc, selected_match, answer_match = local_reward(response, reference)
    glob, similarity = global_reward(response, reference)

    score = (
        WEIGHT_GLOBAL * glob
        + WEIGHT_LOCAL * loc
        + WEIGHT_ANSWER * ans
        + WEIGHT_FORMAT * fmt
    )

    return {
        "score": score,
        "global_reward": WEIGHT_GLOBAL * glob,
        "local_reward": WEIGHT_LOCAL * loc,
        "answer_reward": WEIGHT_ANSWER * ans,
        "format_reward": WEIGHT_FORMAT * fmt,
        "selected_view_match": float(selected_match),
        "answer_view_match": float(answer_match),
        # Whether the map block parsed *and* was well formed enough to compare.
        # Zero also means the sample carried no reference map to compare against.
        "cogmap_comparable": float(similarity is not None and similarity.valid),
        "cogmap_coverage": similarity.coverage if similarity else 0.0,
        "cogmap_similarity": similarity.overall if similarity else 0.0,
        "cogmap_directional_similarity": similarity.directional if similarity else 0.0,
        "cogmap_facing_similarity": similarity.facing if similarity else 0.0,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Entry point the training framework calls once per sampled response.

    Args:
        data_source: Dataset name, unused; part of the framework's signature.
        solution_str: The model's full response.
        ground_truth: The correct option letter.
        extra_info: Per-sample metadata written by
            ``scripts/prepare_grpo_data.py``, carrying the reference map and view
            trajectory.
        **kwargs: Ignored; the framework passes extras that differ by version.

    Returns:
        The result of :func:`score_trajectory`, plus ``reward`` as an alias of
        ``score`` for framework versions that read that name.
    """
    result = score_trajectory(solution_str, ground_truth, extra_info)
    result["reward"] = result["score"]
    return result
