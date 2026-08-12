"""Fuse the text-based and tone-based emotion readings into one result.

Strategy (simple, documented, tunable): a weighted average of the two score
distributions over the canonical labels.

    fused[e] = tw * text[e] + aw * audio[e]      (weights renormalized to sum 1)

Defaults: text 0.6 / audio 0.4 — text is generally the more reliable signal, but
tone meaningfully corrects it (sarcasm, shouting, a shaky voice).

Caveat: the audio model only scores {neutral, joy, anger, sadness}. For the labels
it can't see (fear, surprise, disgust) the audio contribution is 0, so those come
from the text side only (scaled by tw). This is intentional for the MVP; document
it when tuning.
"""
from __future__ import annotations

from typing import Optional

from app.emotion.types import CANONICAL_EMOTIONS, EmotionResult, top_label

DEFAULT_TEXT_WEIGHT = 0.6
DEFAULT_AUDIO_WEIGHT = 0.4


def fuse(
    text_result: Optional[EmotionResult],
    audio_result: Optional[EmotionResult],
    text_weight: float = DEFAULT_TEXT_WEIGHT,
    audio_weight: float = DEFAULT_AUDIO_WEIGHT,
) -> Optional[EmotionResult]:
    """Combine text + audio emotion. Either may be None."""
    if text_result is None and audio_result is None:
        return None

    if audio_result is None:
        return EmotionResult(
            label=text_result.label,
            score=text_result.score,
            scores=dict(text_result.scores),
            source="fused",
        )
    if text_result is None:
        return EmotionResult(
            label=audio_result.label,
            score=audio_result.score,
            scores=dict(audio_result.scores),
            source="fused",
        )

    total = text_weight + audio_weight
    tw = text_weight / total
    aw = audio_weight / total

    scores = {}
    for e in CANONICAL_EMOTIONS:
        scores[e] = tw * text_result.scores.get(e, 0.0) + aw * audio_result.scores.get(e, 0.0)

    label = top_label(scores)
    return EmotionResult(label=label, score=scores[label], scores=scores, source="fused")
