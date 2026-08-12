"""Minimal client for Sarvam's OpenAI-compatible chat endpoint (Sarvam-M LLM).

Used to generate context-aware bot replies with the same Sarvam key as STT. Blocking
`requests` call — run it in a thread executor. Returns None on any failure.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import requests

from app.config import settings

logger = logging.getLogger("voice.bot.sarvam")

_ENDPOINT = "https://api.sarvam.ai/v1/chat/completions"


def chat(messages: List[dict], model: Optional[str] = None,
         max_tokens: int = 160, temperature: float = 0.5) -> Optional[str]:
    key = settings.sarvam_api_key
    if not key:
        return None
    payload = {
        "model": model or settings.emotion_llm_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for headers in ({"Authorization": f"Bearer {key}"}, {"api-subscription-key": key}):
        try:
            r = requests.post(
                _ENDPOINT, headers={**headers, "Content-Type": "application/json"},
                json=payload, timeout=20,
            )
            if r.status_code in (401, 403):
                continue  # try the other auth style
            r.raise_for_status()
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        except requests.HTTPError as exc:
            logger.warning("Sarvam chat HTTP error: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sarvam chat failed: %s", exc)
            return None
    return None
