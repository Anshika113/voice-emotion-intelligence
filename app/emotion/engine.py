"""Convenience wrapper: run text + audio emotion and fuse them in one call.

Reused by the offline pipeline (Phase 2) and the live call orchestrator (Phase 3).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.emotion import audio_emotion, fusion, text_emotion
from app.emotion.types import EmotionResult


def analyze_utterance(
    text: Optional[str] = None,
    waveform: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
) -> dict:
    """Analyze one utterance from text and/or audio.

    Returns a dict with the individual and fused readings (each as_dict or None):
        {"text": ..., "audio": ..., "fused": ...}
    """
    from app.config import settings

    # Text emotion: prefer the context-aware LLM classifier (Sarvam-M); fall back to
    # the local model if the LLM is disabled or errors.
    text_res: Optional[EmotionResult] = None
    if text:
        if settings.emotion_llm and settings.sarvam_api_key:
            from app.emotion import llm_emotion

            text_res = llm_emotion.classify(text)
        if text_res is None:
            text_res = text_emotion.analyze_text(text)

    audio_res: Optional[EmotionResult] = (
        audio_emotion.analyze_audio(waveform, sample_rate) if waveform is not None else None
    )

    # If non-English text is analyzed WITHOUT translation, the text signal is weaker,
    # so lean on the language-agnostic tone model. With translation (or English), the
    # text emotion is reliable, so use balanced weights.
    non_english = bool(text) and text_emotion.looks_non_english(text)
    if non_english and not settings.emotion_use_translation:
        fused_res = fusion.fuse(text_res, audio_res, text_weight=0.35, audio_weight=0.65)
    else:
        fused_res = fusion.fuse(text_res, audio_res)

    return {
        "text": text_res.as_dict() if text_res else None,
        "audio": audio_res.as_dict() if audio_res else None,
        "fused": fused_res.as_dict() if fused_res else None,
    }
