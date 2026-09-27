"""Accuracy of the chosen option, overall and per question setting."""

from typing import Any

from ..constants import SETTINGS
from .answers import extract_answer_letter


def setting_of(item_id: str) -> str:
    """Return the question setting an item id names.

    Args:
        item_id: Identifier such as ``"rotation_group000_q3_6"``.

    Returns:
        One of :data:`drmv3d.constants.SETTINGS`, or ``"other"``.
    """
    lowered = item_id.lower()
    for setting in SETTINGS:
        if setting in lowered:
            return setting
    return "other"


def score_responses(items: list[dict]) -> dict[str, Any]:
    """Score a file of model responses.

    An item counts as correct when the letter recovered from its response equals
    its ground-truth letter. A response that names no option counts as wrong and
    is also reported separately, since a high unparseable count means the output
    format broke rather than the reasoning.

    Args:
        items: Response records, each with ``id``, ``gt_answer`` and ``answer``.

    Returns:
        ``total``, ``correct``, ``accuracy`` and ``unparseable`` overall, plus the
        same per setting under ``settings``.
    """
    order = list(SETTINGS) + ["other"]
    per_setting: dict[str, dict[str, int]] = {
        name: {"total": 0, "correct": 0, "unparseable": 0} for name in order
    }

    for item in items:
        truth: str | None = item.get("gt_answer")
        if not truth:
            continue
        stats = per_setting[setting_of(item.get("id", ""))]
        stats["total"] += 1

        predicted = extract_answer_letter(item.get("answer", ""))
        if predicted is None:
            stats["unparseable"] += 1
        elif predicted.upper() == truth.strip().upper():
            stats["correct"] += 1

    total = sum(s["total"] for s in per_setting.values())
    correct = sum(s["correct"] for s in per_setting.values())
    unparseable = sum(s["unparseable"] for s in per_setting.values())

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "unparseable": unparseable,
        "settings": {
            name: {
                **stats,
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
            }
            for name, stats in per_setting.items()
            if stats["total"]
        },
    }


def format_scores(scores: dict[str, Any]) -> str:
    """Render scores as a table.

    Args:
        scores: Result of :func:`score_responses`.

    Returns:
        A table with one row per setting and an overall row.
    """
    header = f"{'setting':<10} {'items':>6} {'correct':>8} {'accuracy':>9} {'no answer':>10}"
    rule = "-" * len(header)
    lines = [header, rule]
    for name, stats in scores["settings"].items():
        lines.append(
            f"{name:<10} {stats['total']:>6} {stats['correct']:>8} "
            f"{stats['accuracy'] * 100:>8.2f}% {stats['unparseable']:>10}"
        )
    lines.append(rule)
    lines.append(
        f"{'overall':<10} {scores['total']:>6} {scores['correct']:>8} "
        f"{scores['accuracy'] * 100:>8.2f}% {scores['unparseable']:>10}"
    )
    return "\n".join(lines)
