"""Reading the four output blocks back out of a generated response.

During training the response comes from a policy that is still learning the
format, so every reader here tolerates what a partly-formed response looks like:
a missing block, a code fence around the JSON, single quotes instead of double,
or prose wrapped around the object. Tolerating those is what lets the reward
distinguish "nearly right" from "wrong", which is the whole point of shaping it.
"""

import ast
import json
import re
from typing import Any

#: Keys that have named the question-referenced viewpoint across output formats.
_SELECTED_VIEW_KEYS = ("view", "selected_view", "target_view", "start_view")

#: Keys that have named the viewpoint showing the answer object.
_ANSWER_VIEW_KEYS = ("answer_visible_in_view", "answer_view")


def extract_block(response: str, tag: str) -> str | None:
    """Return the contents of one output block.

    Args:
        response: The model's full response.
        tag: Block name, for example ``"cogmap"``.

    Returns:
        The text between the tags, or ``None`` if the block is absent.
    """
    if not response:
        return None
    match = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else None


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding ```` ```json ... ``` ```` fence, if present."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    newline = text.find("\n")
    if newline != -1:
        text = text[newline + 1:]
    if "```" in text:
        text = text.rsplit("```", 1)[0]
    return text.strip()


def parse_json_relaxed(text: str | None) -> Any | None:
    """Parse JSON that a still-learning policy produced.

    Tried in order: the text as JSON; the text as a Python literal, which accepts
    single-quoted keys; and the outermost bracketed span, which drops prose around
    the object.

    Args:
        text: Block contents, or ``None``.

    Returns:
        The parsed object, or ``None`` if nothing parsed.
    """
    if not text:
        return None
    text = _strip_code_fence(text)
    if not text:
        return None

    for candidate in _candidates(text):
        for parse in (json.loads, ast.literal_eval):
            try:
                return parse(candidate)
            except (ValueError, SyntaxError):
                continue
    return None


def _candidates(text: str) -> list[str]:
    """Return the strings worth trying to parse, most faithful first.

    After the text itself come the bracketed spans, the one matching the text's
    own first bracket first so an array is not mistaken for the objects inside it.
    Last comes the array with its complete elements kept and the closing bracket
    restored, which is what a response cut off at the token budget looks like.
    """
    candidates = [text]

    spans = {}
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            spans[opener] = text[start:end + 1]

    first = min((text.find(b) for b in "[{" if text.find(b) != -1), default=-1)
    order = "[{" if first != -1 and text[first] == "[" else "{["
    candidates += [spans[b] for b in order if b in spans]

    truncated = _close_truncated_array(text)
    if truncated is not None:
        candidates.append(truncated)
    return candidates


def _close_truncated_array(text: str) -> str | None:
    """Rebuild an array whose tail was cut off, dropping the incomplete element.

    Args:
        text: Text that may begin an array and stop partway through it.

    Returns:
        The array with its complete elements and a closing bracket, or ``None`` if
        the text begins no array or no element completed.
    """
    start = text.find("[")
    if start == -1:
        return None

    elements: list[str] = []
    depth = 0
    element_start = None
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth == 2 and element_start is None:
                element_start = index
        elif char in "]}":
            depth -= 1
            if depth == 1 and element_start is not None:
                elements.append(text[element_start:index + 1])
                element_start = None
            elif depth == 0:
                break

    return "[" + ",".join(elements) + "]" if elements else None


def primary_egomap(egomap: Any) -> dict | None:
    """Return the egocentric map of the viewpoint the question reasons from.

    The block holds one map per viewpoint, and exactly one is flagged
    ``is_primary``. A response that omits the flag still usually puts the
    question's viewpoint first, so the first entry is the fallback.

    Args:
        egomap: The parsed egocentric-map block.

    Returns:
        The primary map, or ``None`` if the block holds none.
    """
    if isinstance(egomap, dict):
        return egomap
    if not isinstance(egomap, list):
        return None
    entries = [entry for entry in egomap if isinstance(entry, dict)]
    for entry in entries:
        if entry.get("is_primary"):
            return entry
    return entries[0] if entries else None


def _first_value(entry: dict, keys: tuple[str, ...]) -> str:
    """Return the first non-empty string among ``keys``."""
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def predicted_view_trajectory(response: str) -> tuple[str, str]:
    """Return the viewpoints a response says it reasoned from.

    Args:
        response: The model's full response.

    Returns:
        The viewpoint the response selected and the viewpoint it claims shows the
        answer object. Either is an empty string if the response does not say.
    """
    primary = primary_egomap(parse_json_relaxed(extract_block(response, "egomap")))
    if primary is None:
        return "", ""
    return _first_value(primary, _SELECTED_VIEW_KEYS), _first_value(primary, _ANSWER_VIEW_KEYS)


def view_number(name: str) -> str | None:
    """Return the number in a viewpoint name.

    Comparing numbers rather than whole names absorbs the difference between
    ``"Image 2"`` and ``"View 2"``, which are the same viewpoint.

    Args:
        name: A viewpoint name.

    Returns:
        The digits as a string, or ``None`` if the name holds none.
    """
    if not isinstance(name, str):
        return None
    match = re.search(r"\d+", name)
    return match.group(0) if match else None
