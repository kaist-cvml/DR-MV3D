"""Recovering the chosen option from a model response.

A response ends with an ``<answer>`` block holding the option as the question
writes it, for example ``"C. Light purple sofa"``. Reading the letter back out of
that is all scoring needs. The fallbacks below exist because a partly-formed
response still often states an option, and counting those as wrong rather than as
unparseable would overstate how often the format breaks.
"""

import re

#: The answer block of a well-formed response.
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

#: An option written as a label and its text, e.g. ``"C. Light purple sofa"``.
_LABELLED_OPTION = re.compile(r"([A-E])\.")

#: A bare option letter, for a response that gives the label and nothing else.
_BARE_LABEL = re.compile(r"\b([A-E])\b")


def extract_answer_letter(response: str) -> str | None:
    """Return the option letter a response settles on.

    The answer block is read first. If it holds no option, the whole response is
    searched, on the assumption that the last option mentioned is the conclusion.

    Args:
        response: The model's full response text.

    Returns:
        A letter in ``A``-``E``, or ``None`` if the response names no option.
    """
    if not response:
        return None

    block = _ANSWER_BLOCK.search(response)
    candidates = [block.group(1), response] if block else [response]
    for text in candidates:
        for pattern in (_LABELLED_OPTION, _BARE_LABEL):
            found = pattern.findall(text)
            if found:
                return found[-1]
    return None
