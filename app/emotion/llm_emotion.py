"""Context-aware emotion classification via an LLM (Sarvam-M).

The small text model classifies by emotional *words*; it misses implied/contextual
emotion ("everyone stood there with a cake" = joy). An LLM understands the situation,
so this is much more accurate on nuanced sentences — in English, Hindi, and Hinglish.

Uses Sarvam's OpenAI-compatible chat endpoint with the same Sarvam key used for STT.
Returns None on any failure so the caller can fall back to the local model.
"""
from __future__ import annotations

import json
import logging
import re

import requests

from app.config import settings
from app.emotion.types import CANONICAL_EMOTIONS, EmotionResult, empty_scores

logger = logging.getLogger("voice.emotion.llm")

_ENDPOINT = "https://api.sarvam.ai/v1/chat/completions"

_SYSTEM = (
    "You are an expert emotion classifier. Read the caller's sentence and decide the "
    "single most fitting emotion, using CONTEXT and implied feeling — not just keywords. "
    "The sentence may be English, Hindi, or Hinglish. "
    "Choose exactly ONE label from: joy, anger, sadness, fear, surprise, disgust, neutral. "
    'Reply with ONLY compact JSON, no extra text: {"emotion": "<label>", "confidence": <0..1>}.'
)


def _post(payload: dict):
    """Try both Sarvam auth styles; return parsed JSON or None."""
    key = settings.sarvam_api_key
    for headers in (
        {"Authorization": f"Bearer {key}"},
        {"api-subscription-key": key},
    ):
        try:
            r = requests.post(
                _ENDPOINT, headers={**headers, "Content-Type": "application/json"},
                json=payload, timeout=15,
            )
            if r.status_code in (401, 403):
                continue  # wrong auth style — try the other
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as exc:
            logger.warning("Sarvam LLM HTTP error: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sarvam LLM request failed: %s", exc)
            return None
    return None


def _extract(content: str):
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            pass
    low = content.lower()
    for e in CANONICAL_EMOTIONS:  # fallback: a bare label word
        if e in low:
            return {"emotion": e, "confidence": 0.85}
    return None


def classify(text: str):
    """Return an EmotionResult from the LLM, or None to fall back to the local model."""
    text = (text or "").strip()
    if not text or not settings.sarvam_api_key:
        return None

    data = _post(
        {
            "model": settings.emotion_llm_model,
            "temperature": 0,
            "max_tokens": 60,  # just a short JSON label — keeps free-tier tokens low
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": text},
            ],
        }
    )
    if not data:
        return None

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.warning("Sarvam LLM: unexpected response shape")
        return None

    parsed = _extract(content or "")
    if not parsed:
        return None

    label = str(parsed.get("emotion", "")).lower().strip()
    if label not in CANONICAL_EMOTIONS:
        return None
    try:
        conf = max(0.05, min(1.0, float(parsed.get("confidence", 0.85))))
    except (TypeError, ValueError):
        conf = 0.85

    scores = empty_scores()
    scores[label] = conf
    if label != "neutral":
        scores["neutral"] = round(1.0 - conf, 4)
    logger.info("LLM emotion: %r -> %s (%.2f)", text, label, conf)
    return EmotionResult(label=label, score=conf, scores=scores, source="text-llm")
