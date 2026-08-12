"""Speech-to-text via Sarvam AI — excellent Hindi/Hinglish (Indian-language ASR).

REST API: we wrap the buffered utterance as a 16 kHz mono WAV and POST it. Blocking
`requests` call, meant to run in a thread executor. Returns '' on failure so the
pipeline degrades gracefully.

Docs: https://docs.sarvam.ai  (endpoint: POST /speech-to-text)
"""
from __future__ import annotations

import io
import logging
import wave

import requests

from app.audio import convert
from app.config import settings

logger = logging.getLogger("voice.stt.sarvam")

_ENDPOINT = "https://api.sarvam.ai/speech-to-text"
_TRANSLATE_ENDPOINT = "https://api.sarvam.ai/translate"


def _pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # PCM16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def transcribe_pcm16(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
    """Transcribe PCM16 audio via Sarvam. Returns '' on empty/failure."""
    if not pcm_bytes:
        return ""
    if not settings.sarvam_api_key:
        logger.warning("SARVAM_API_KEY not set; cannot transcribe")
        return ""

    pcm16k, _ = convert.resample_pcm16(pcm_bytes, sample_rate, 16000)
    wav_bytes = _pcm16_to_wav_bytes(pcm16k, 16000)

    try:
        resp = requests.post(
            _ENDPOINT,
            headers={"api-subscription-key": settings.sarvam_api_key},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={
                "model": settings.sarvam_model,
                "language_code": settings.sarvam_language or "unknown",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return (resp.json().get("transcript") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sarvam STT failed: %s", exc)
        return ""


def translate_to_english(text: str, source: str = "hi-IN") -> str:
    """Translate Indian-language text to English (Sarvam Mayura). '' on failure."""
    if not text or not settings.sarvam_api_key:
        return ""
    try:
        resp = requests.post(
            _TRANSLATE_ENDPOINT,
            headers={
                "api-subscription-key": settings.sarvam_api_key,
                "Content-Type": "application/json",
            },
            json={
                "input": text,
                "source_language_code": source,
                "target_language_code": "en-IN",
                "model": "mayura:v1",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return (resp.json().get("translated_text") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sarvam translate failed: %s", exc)
        return ""
