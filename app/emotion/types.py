"""Shared emotion types and the canonical label set.

Both the text model and the audio model are normalized into this one taxonomy so
they can be fused. The audio model only distinguishes a subset (neutral/joy/anger/
sadness); the remaining labels come from the text model only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# One shared taxonomy for text + audio + fused results.
CANONICAL_EMOTIONS = [
    "joy",
    "anger",
    "sadness",
    "fear",
    "surprise",
    "disgust",
    "neutral",
]


@dataclass
class EmotionResult:
    """A single emotion reading.

    label:  the winning emotion (highest score)
    score:  confidence of the winning label (0..1)
    scores: full distribution across CANONICAL_EMOTIONS
    source: "text" | "audio" | "fused"
    """

    label: str
    score: float
    scores: Dict[str, float] = field(default_factory=dict)
    source: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "source": self.source,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


def empty_scores() -> Dict[str, float]:
    return {e: 0.0 for e in CANONICAL_EMOTIONS}


def top_label(scores: Dict[str, float]) -> str:
    return max(scores, key=scores.get)
