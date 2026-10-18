import re
from collections import Counter
from typing import Dict, List

LABELS = ["positive", "negative", "neutral"]
UNKNOWN = "unknown"

_LABEL_PATTERN = "|".join(LABELS)
# Phrases that introduce the model's final answer, e.g. "Overall sentiment: negative"
_CUE_RE = re.compile(
    rf"(?:sentiment|determination|only|leans?|leaning)\s*(?:is|:)?\s*(?:towards?\s+)?({_LABEL_PATTERN})\b"
)
_LABEL_RE = re.compile(rf"\b({_LABEL_PATTERN})\b")


def extract_label(output: str) -> str:
    """Pull the sentiment label out of a model reply ('unknown' if unclear)."""
    text = output.lower()

    mentioned = _LABEL_RE.findall(text)
    if len(set(mentioned)) == 1:
        return mentioned[0]

    # Several labels mentioned: trust the last "sentiment: X" style phrase
    cues = _CUE_RE.findall(text)
    if cues:
        return cues[-1]

    return UNKNOWN


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def rouge_1(output: str, expected: str) -> float:
    """ROUGE-1 F1: word overlap between the output and the expected text."""
    out_counts = Counter(_tokens(output))
    exp_counts = Counter(_tokens(expected))
    overlap = sum((out_counts & exp_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(out_counts.values())
    recall = overlap / sum(exp_counts.values())
    return 2 * precision * recall / (precision + recall)


def confusion_matrix(expected: List[str], predicted: List[str]) -> Dict[str, Dict[str, int]]:
    """Count predictions per expected label: matrix[expected][predicted]."""
    matrix = {exp: {pred: 0 for pred in LABELS + [UNKNOWN]} for exp in LABELS}
    for exp, pred in zip(expected, predicted):
        matrix[exp][pred] += 1
    return matrix


def classification_report(expected: List[str], predicted: List[str]) -> Dict[str, Dict[str, float]]:
    """Precision, recall and F1 for each label."""
    report = {}
    for label in LABELS:
        true_pos = sum(1 for e, p in zip(expected, predicted) if e == label and p == label)
        predicted_count = sum(1 for p in predicted if p == label)
        support = sum(1 for e in expected if e == label)

        precision = true_pos / predicted_count if predicted_count else 0.0
        recall = true_pos / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        report[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    return report
