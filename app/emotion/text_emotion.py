"""Emotion from the transcript text — multilingual (English, Hindi, Hinglish).

Two models, chosen per input:
  * English (Latin script, detected as English): the fine-grained 7-emotion model
    `j-hartmann/emotion-english-distilroberta-base` — best granularity.
  * Everything else (Devanagari Hindi, or non-English detected): a multilingual
    sentiment model, mapped onto the canonical emotions by valence.

Why this split: the English model is sharp but English-only. For Hindi/Hinglish we
lean on the multilingual model for valence, and the (language-agnostic) audio tone
model — see audio_emotion.py — sharpens *which* negative emotion it is (anger vs
sadness) during fusion.

All models are lazy-loaded and cached.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.config import settings
from app.emotion.types import CANONICAL_EMOTIONS, EmotionResult, empty_scores, top_label

logger = logging.getLogger("voice.emotion.text")

_EN_MODEL = "j-hartmann/emotion-english-distilroberta-base"
# XLM-RoBERTa multilingual sentiment — robust across Hindi/Hinglish/English.
_MULTI_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

_en_pipe = None
_multi_pipe = None
_vader = None

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# HF English model labels -> canonical (already aligned).
_EN_LABEL_MAP = {
    "anger": "anger", "disgust": "disgust", "fear": "fear", "joy": "joy",
    "neutral": "neutral", "sadness": "sadness", "surprise": "surprise",
}

# How to spread a "negative" sentiment across canonical emotions (audio refines it).
_NEG_SPLIT = {"anger": 0.45, "sadness": 0.35, "fear": 0.10, "disgust": 0.10}


def _norm_sentiment(label: str) -> str:
    """Map a sentiment label (any casing, or LABEL_0/1/2) to pos/neg/neu."""
    l = label.lower()
    if "pos" in l or l.endswith("_2") or l == "2":
        return "pos"
    if "neg" in l or l.endswith("_0") or l == "0":
        return "neg"
    return "neu"


# ---- language routing ----

def looks_non_english(text: str) -> bool:
    """True for Hindi (Devanagari) or any language langdetect flags as not English."""
    if _DEVANAGARI.search(text):  # Hindi in Devanagari
        return True
    try:
        from langdetect import detect

        return detect(text) != "en"
    except Exception:  # noqa: BLE001 — langdetect missing or too-short text
        return False


# ---- English fine-grained model ----

def _english_scores(text: str) -> dict:
    global _en_pipe
    if _en_pipe is None:
        from transformers import pipeline

        logger.info("Loading English emotion model: %s", _EN_MODEL)
        _en_pipe = pipeline("text-classification", model=_EN_MODEL, top_k=None)
    out = _en_pipe(text)
    if out and isinstance(out[0], list):
        out = out[0]
    scores = empty_scores()
    for item in out:
        lbl = _EN_LABEL_MAP.get(str(item["label"]).lower())
        if lbl:
            scores[lbl] = float(item["score"])
    return scores


# ---- multilingual sentiment model ----

def _multilingual_scores(text: str) -> dict:
    global _multi_pipe
    if _multi_pipe is None:
        from transformers import pipeline

        logger.info("Loading multilingual sentiment model: %s", _MULTI_MODEL)
        _multi_pipe = pipeline("text-classification", model=_MULTI_MODEL, top_k=None)
    out = _multi_pipe(text)
    if out and isinstance(out[0], list):
        out = out[0]

    pos = neg = neu = 0.0
    for item in out:
        bucket = _norm_sentiment(str(item["label"]))
        score = float(item["score"])
        if bucket == "pos":
            pos = score
        elif bucket == "neg":
            neg = score
        else:
            neu = score

    scores = empty_scores()
    scores["joy"] = pos
    scores["neutral"] = neu
    for emo, weight in _NEG_SPLIT.items():
        scores[emo] = neg * weight
    return scores


# ---- VADER fallback (English) ----

def _vader_fallback(text: str) -> dict:
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _vader = SentimentIntensityAnalyzer()
    compound = _vader.polarity_scores(text)["compound"]
    scores = empty_scores()
    if compound >= 0.25:
        scores["joy"] = min(1.0, compound)
        scores["neutral"] = 1.0 - scores["joy"]
    elif compound <= -0.25:
        neg = min(1.0, -compound)
        for emo, weight in _NEG_SPLIT.items():
            scores[emo] = neg * weight
        scores["neutral"] = 1.0 - neg
    else:
        scores["neutral"] = 1.0
    return scores


def _non_english_scores(text: str) -> dict:
    """For Hindi/Hinglish: translate to English + use the sharp English emotion model.
    Falls back to the multilingual sentiment model if translation is unavailable."""
    if settings.emotion_use_translation:
        from app.stt.sarvam_stt import translate_to_english

        english = translate_to_english(text)
        if english:
            logger.info("Emotion via translation: %r -> %r", text, english)
            return _english_scores(english)
    return _multilingual_scores(text)


def analyze_text(text: str) -> Optional[EmotionResult]:
    """Return an EmotionResult for `text` (any of English/Hindi/Hinglish), or None."""
    text = (text or "").strip()
    if not text:
        return None

    try:
        if looks_non_english(text):
            scores = _non_english_scores(text)
        else:
            scores = _english_scores(text)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        logger.warning("Text emotion model unavailable (%s); using VADER fallback", exc)
        scores = _vader_fallback(text)

    label = top_label(scores)
    return EmotionResult(label=label, score=scores[label], scores=scores, source="text")
