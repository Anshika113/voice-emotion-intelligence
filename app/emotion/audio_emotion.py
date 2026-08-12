"""Emotion from the voice tone (acoustic), independent of the words.

Uses HuggingFace `superb/wav2vec2-base-superb-er` (trained on IEMOCAP), which emits
four classes: neu / hap / ang / sad -> mapped to canonical neutral/joy/anger/sadness.
The model wants a mono float32 waveform in [-1, 1] at 16 kHz; we resample if needed.

Lazy-loaded and cached. Returns None (rather than raising) if the model can't run,
so the pipeline can continue on text alone.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.emotion.types import EmotionResult, empty_scores, top_label

logger = logging.getLogger("voice.emotion.audio")

_MODEL_NAME = "superb/wav2vec2-base-superb-er"
_pipeline = None
TARGET_SR = 16000

_LABEL_MAP = {
    "neu": "neutral",
    "hap": "joy",
    "ang": "anger",
    "sad": "sadness",
    # be tolerant of long-form labels too
    "neutral": "neutral",
    "happy": "joy",
    "angry": "anger",
    "sadness": "sadness",
}


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline  # imported lazily (heavy)

        logger.info("Loading audio emotion model: %s", _MODEL_NAME)
        _pipeline = pipeline("audio-classification", model=_MODEL_NAME, top_k=None)
    return _pipeline


def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resample (good enough for 8k->16k speech)."""
    if sr_in == sr_out:
        return wav
    duration = wav.shape[0] / float(sr_in)
    n_out = max(1, int(round(duration * sr_out)))
    x_old = np.linspace(0.0, duration, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def _to_mono_float(waveform: np.ndarray) -> np.ndarray:
    wav = np.asarray(waveform)
    if wav.ndim > 1:  # (samples, channels) -> mono
        wav = wav.mean(axis=1)
    if wav.dtype.kind in ("i", "u"):  # int PCM -> float [-1, 1]
        max_val = float(np.iinfo(wav.dtype).max)
        wav = wav.astype(np.float32) / max_val
    else:
        wav = wav.astype(np.float32)
    return wav


def analyze_audio(waveform: np.ndarray, sample_rate: int) -> Optional[EmotionResult]:
    """Return an EmotionResult from a raw waveform, or None if empty/unavailable."""
    if waveform is None or len(waveform) == 0:
        return None

    wav = _to_mono_float(waveform)
    if sample_rate != TARGET_SR:
        wav = _resample(wav, sample_rate, TARGET_SR)

    try:
        out = _load_pipeline()({"raw": wav, "sampling_rate": TARGET_SR})
        if out and isinstance(out[0], list):
            out = out[0]
        scores = empty_scores()
        for item in out:
            lbl = _LABEL_MAP.get(str(item["label"]).lower())
            if lbl:
                scores[lbl] = float(item["score"])
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        logger.warning("Audio emotion model unavailable (%s); skipping tone signal", exc)
        return None

    label = top_label(scores)
    return EmotionResult(label=label, score=scores[label], scores=scores, source="audio")
