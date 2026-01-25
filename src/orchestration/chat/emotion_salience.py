from __future__ import annotations

import math
import re
from typing import Tuple


POSITIVE_WORDS = {"great", "good", "love", "nice", "wonderful", "happy", "excited"}
NEGATIVE_WORDS = {"bad", "sad", "angry", "upset", "terrible", "hate", "awful", "tired"}
INTENSIFIERS = {"very", "extremely", "incredibly", "totally", "really", "so", "super"}

_POS = POSITIVE_WORDS
_NEG = NEGATIVE_WORDS

_TOKEN_RE = re.compile(r"\w+")


def _normalize(v: float, lo: float, hi: float) -> float:
    if hi - lo == 0:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def estimate_salience_and_valence(text: str) -> Tuple[float, float]:
    """
    Lightweight heuristic for salience & emotional valence.
    Returns (salience:0..1, valence:-1..1)
    """
    if not text:
        return 0.0, 0.0
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    length_factor = min(1.0, len(tokens) / 25.0)
    punctuation_count = text.count("!") + text.count("?")
    punctuation_boost = min(0.25, 0.07 * punctuation_count)
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    emphasis_boost = min(0.25, uppercase_ratio * 0.6)
    intensifier_hits = sum(1 for t in tokens if t in INTENSIFIERS)
    intensifier_boost = min(0.2, intensifier_hits * 0.05)
    salience = max(
        0.0,
        min(
            1.0,
            0.25
            + length_factor * 0.4
            + punctuation_boost
            + emphasis_boost
            + intensifier_boost,
        ),
    )

    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        valence = 0.0
    else:
        valence = (pos - neg) / total  # -1..1
    # Smooth extreme single-word swings
    valence = math.tanh(valence)
    return salience, valence
