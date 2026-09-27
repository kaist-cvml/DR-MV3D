"""Reading structure out of a benchmark question.

A question states which viewpoint to reason from, optionally a hypothetical
movement from it, and a list of answer options. These helpers recover that
structure so the scaffold stage can place the camera and score the options.

This module is a leaf: it imports nothing from the rest of the package.
"""

import re

#: Option labels the benchmark uses, in order.
OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D")


#: Where a question's option list begins: the last "A." that starts a word and is
#: followed by a space. Anchoring on the list itself rather than on the question
#: mark matters, because some questions end in a full stop instead.
_OPTION_LIST_START = re.compile(r"\bA\.\s")


def question_body(question: str) -> str:
    """Return the question without its option list.

    Anything scanning the question for an object name has to stop before the
    options, or it picks the whole list up as part of the name. Splitting on the
    question mark is not enough: some questions end in a full stop.

    Args:
        question: Full question text.

    Returns:
        The text before the option list, or the whole question if it has none.
    """
    starts = list(_OPTION_LIST_START.finditer(question))
    return question[:starts[-1].start()].strip() if starts else question.strip()


def extract_options(question: str) -> dict[str, str]:
    """Split the trailing option list of a question.

    Args:
        question: Full question text, ending with ``"A. ...  B. ..."``.

    Returns:
        Mapping from option label to option text. Empty if the question has no
        recognisable option list.
    """
    options: dict[str, str] = {}
    starts = list(_OPTION_LIST_START.finditer(question))
    if not starts:
        return options
    option_text = question[starts[-1].start():].strip()
    for label in OPTION_LABELS:
        match = re.search(rf"\b{label}\.\s*(.+?)(?=\s+[A-Z]\.|$)", option_text)
        if match:
            options[label] = match.group(1).strip()
    return options


def get_answer_option_text(question: str, gt_answer: str) -> str | None:
    """Return the option text the ground-truth letter refers to.

    Args:
        question: Full question text.
        gt_answer: Ground-truth option letter, for example ``"C"``.

    Returns:
        The option text, for example ``"Light purple sofa"``, or ``None`` if the
        letter is not a single ``A``-``F`` or the option cannot be found.
    """
    if not question or not gt_answer:
        return None
    answer_key = gt_answer.strip().upper()
    if len(answer_key) != 1 or answer_key not in "ABCDEF":
        return None
    # Option text may itself contain capital letters, so split on the next
    # "<letter>." rather than on whitespace.
    for match in re.findall(r"([A-Z]\.\s*.*?)(?=\s+[A-Z]\.|$)", question):
        candidate = match.strip()
        if candidate.upper().startswith(answer_key + "."):
            text = candidate.split(".", 1)[1].strip()
            return text or None
    return None


def format_answer_option(question: str, gt_answer: str) -> str:
    """Return the ground-truth option as it must appear in the answer block.

    Unlike :func:`get_answer_option_text` this keeps the label, because the
    answer block of a training target is the full ``"C. Light purple sofa"``.

    Args:
        question: Full question text.
        gt_answer: Ground-truth option letter, for example ``"C"``.

    Returns:
        Label and text joined as the question writes them.

    Raises:
        ValueError: If the question has no such option, which would otherwise
            produce a target the model can never match.
    """
    for match in re.findall(r"([A-Z]\.\s+.*?)(?=[A-Z]\.|$)", question):
        option = match.strip()
        if option.startswith(gt_answer + "."):
            return option
    raise ValueError(f"option {gt_answer!r} not found in question: {question!r}")


def object_name_matches(object_name: str, option_text: str) -> bool:
    """Test whether a map object and an answer option denote the same thing.

    Matching is case-insensitive and accepts either string containing the other,
    because option text and map labels are written independently, for example
    ``"Light purple sofa"`` and ``"light purple sofa"``.

    Args:
        object_name: Name of an object in the cognitive map.
        option_text: Text of an answer option.

    Returns:
        ``True`` if the two names refer to the same object.
    """
    if not object_name or not option_text:
        return False
    left = object_name.strip().lower()
    right = option_text.strip().lower()
    return left == right or right in left or left in right


def find_object_by_name(objects: list[dict], name: str) -> dict | None:
    """Return the first map object whose name matches ``name``.

    Args:
        objects: Object entries of a cognitive map.
        name: Name to look for, typically an answer option text.

    Returns:
        The matching object entry, or ``None``.
    """
    if not name:
        return None
    for obj in objects:
        if object_name_matches(obj.get("name", ""), name):
            return obj
    return None


def find_view_by_index(views: list[dict], index: int) -> dict | None:
    """Return the viewpoint whose name carries a given number.

    Args:
        views: Viewpoint entries of a cognitive map.
        index: Number to look for, for example ``2`` for ``"Image 2"``.

    Returns:
        The matching viewpoint entry, or ``None``.
    """
    for view in views:
        if str(index) in view.get("name", ""):
            return view
    return None


def extract_view_from_question(question: str, views: list[dict]) -> dict | None:
    """Return the viewpoint the question refers to.

    Args:
        question: Full question text.
        views: Viewpoint entries of a cognitive map.

    Returns:
        The referenced viewpoint, the first viewpoint if the question names
        none, or ``None`` if there are no viewpoints at all.
    """
    for pattern in (r"[Ii]mage\s*(\d+)", r"[Vv]iew\s*(\d+)"):
        match = re.search(pattern, question)
        if match:
            view = find_view_by_index(views, int(match.group(1)))
            if view is not None:
                return view
    return views[0] if views else None


def parse_start_view_index(question: str) -> int | None:
    """Return the number of the viewpoint the reasoning should start from.

    Args:
        question: Full question text.

    Returns:
        The viewpoint number, or ``None`` if the question does not name one
        with a recognised phrasing.
    """
    patterns = (
        r"(?:viewpoint presented in|facing the same direction as shown in) image (\d+)",
        r"shown in image\s*(\d+)",
        r"from image\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


_ACTION_TOKENS = re.compile(
    r"(?P<turn180>turn\s*180(?:\s*degrees)?(?:\s*around)?)|"
    r"(?P<turnright>turn\s*(?:90\s*degrees\s*to\s*the\s*right|right))|"
    r"(?P<turnleft>turn\s*(?:90\s*degrees\s*to\s*the\s*left|left))|"
    r"(?P<moveforward>move\s*forward)|"
    r"(?P<moveback>move\s*(?:back|backward))|"
    r"(?P<moveleft>move\s*(?:to\s*the\s*)?left)|"
    r"(?P<moveright>move\s*(?:to\s*the\s*)?right)"
)

_ACTION_NAMES = {
    "turn180": "TURN_180",
    "turnright": "TURN_RIGHT_90",
    "turnleft": "TURN_LEFT_90",
    "moveforward": "MOVE_FORWARD",
    "moveback": "MOVE_BACK",
    "moveleft": "MOVE_LEFT",
    "moveright": "MOVE_RIGHT",
}


def parse_action_sequence(question: str) -> list[str]:
    """Return the ordered camera actions stated in a question's hypothetical.

    Only the clause introduced by "then I ..." is considered, so movements
    described elsewhere in the preamble are not mistaken for instructions.

    Args:
        question: Full question text.

    Returns:
        Action names such as ``["TURN_RIGHT_90", "MOVE_FORWARD"]``; empty if the
        question states no hypothetical movement.
    """
    match = re.search(r"then i\s+(.*?)(?:,|\?)", question.lower())
    if not match:
        return []
    actions = [_ACTION_NAMES.get(token.lastgroup, "") for token in _ACTION_TOKENS.finditer(match.group(1))]
    return [action for action in actions if action]


def parse_mental_turn(question: str) -> str:
    """Return the turn described by a rotation question, as free text.

    Rotation questions phrase the turn differently from the generic "then I ..."
    clause handled by :func:`parse_action_sequence`, so they get their own
    pattern.

    Args:
        question: Full question text.

    Returns:
        One of ``"90 degrees to the left"``, ``"90 degrees to the right"``,
        ``"180 degrees around"``, or ``"none"``.
    """
    match = re.search(
        r"(?:and|then)(?: I)? turn "
        r"(90 degrees to the left|90 degrees to the right|180 degrees around)",
        question,
        re.IGNORECASE,
    )
    return match.group(1).lower() if match else "none"


def detect_question_type(question: str) -> str:
    """Classify a question so the reasoning stage can pick an answer template.

    Args:
        question: Full question text.

    Returns:
        One of ``closer``, ``direction``, ``nearest``, ``visibility`` or
        ``spatial``. Yes/no questions such as "is X behind me?" are ``spatial``,
        not ``visibility``; only explicit "can I see" phrasings are visibility
        questions.
    """
    q = question.lower()
    if "closer" in q or "will i get" in q or "will i move" in q:
        return "closer"
    if "which direction" in q or "did the camera move" in q or "did i move" in q:
        return "direction"
    if "nearest" in q:
        return "nearest"
    if "able to see" in q or "can you see" in q or "can i see" in q:
        return "visibility"
    return "spatial"


def is_spatial_yesno(question: str) -> bool:
    """Test whether a spatial question is answered with Yes/No options.

    Args:
        question: Full question text.

    Returns:
        ``True`` if the first option is "Yes" or "No".
    """
    # Some option lists write "No." with a full stop, so punctuation goes first.
    first = extract_options(question).get("A", "").strip().strip(".").strip().lower()
    return first in ("yes", "no")


def describe_item_setting(item_id: str) -> str:
    """Return the benchmark setting encoded in an item id.

    Args:
        item_id: Identifier such as ``"rotation_group000_q3_6"``.

    Returns:
        ``rotation``, ``translation``, ``among``, ``around``, or an empty string.
    """
    lowered = item_id.lower()
    if "rotation" in lowered:
        return "rotation"
    if "translation" in lowered or "linear" in lowered:
        return "translation"
    if "among" in lowered:
        return "among"
    if "around" in lowered:
        return "around"
    return ""
