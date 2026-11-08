import re
import string
from collections import Counter


def normalize(text):
    text = text.lower()
    text = "".join(
        char
        for char in text
        if char not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return float(
        normalize(prediction) == normalize(reference)
    )


def answer_f1(prediction, reference):
    predicted = normalize(prediction).split()
    expected = normalize(reference).split()

    if not predicted or not expected:
        return float(predicted == expected)

    overlap = sum(
        (
            Counter(predicted)
            & Counter(expected)
        ).values()
    )

    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted)
    recall = overlap / len(expected)

    return (
        2 * precision * recall
        / (precision + recall)
    )


def evidence_f1(evidence, supporting_facts):
    predicted = {
        (row["title"], row["index"])
        for row in evidence
    }

    expected = {
        (title, int(index))
        for title, index in supporting_facts
    }

    if not predicted or not expected:
        return float(predicted == expected)

    overlap = len(predicted & expected)

    if not overlap:
        return 0.0

    precision = overlap / len(predicted)
    recall = overlap / len(expected)

    return (
        2 * precision * recall
        / (precision + recall)
    )
