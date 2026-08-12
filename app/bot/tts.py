"""Text-to-speech returning μ-law 8 kHz audio for Twilio — free, no API key.

Uses `edge-tts` (Microsoft Edge's online TTS): free, no account/key, multilingual
(Hindi + English + more). It returns MP3, which we decode and convert to μ-law:

    text --(edge-tts)--> MP3 --(soundfile)--> PCM16 --(resample)--> 8 kHz --(μ-law)--> Twilio

Voice is picked by script: Devanagari -> a Hindi voice, otherwise an English voice.
Blocking wrapper (`synthesize_mulaw`) is meant to run in a thread executor.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re

from app.audio import convert
from app.config import settings

logger = logging.getLogger("voice.bot.tts")

_TARGET_SR = 8000
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _pick_voice(text: str) -> str:
    return settings.tts_voice_hindi if _DEVANAGARI.search(text) else settings.tts_voice_english


def _synthesize_mp3(text: str, voice: str) -> bytes:
    """Run edge-tts (async) to completion and return MP3 bytes."""
    import edge_tts

    async def _run() -> bytes:
        buf = bytearray()
        communicate = edge_tts.Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        return bytes(buf)

    return asyncio.run(_run())  # we're in a worker thread; safe to own a loop


def _mp3_to_mulaw(mp3_bytes: bytes) -> bytes:
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(mp3_bytes), dtype="int16", always_2d=True)
    mono = data[:, 0]
    pcm8, _ = convert.resample_pcm16(mono.astype(np.int16).tobytes(), sr, _TARGET_SR)
    return convert.pcm16_to_mulaw(pcm8)


def synthesize_mulaw(text: str) -> bytes:
    """Return raw μ-law 8 kHz bytes for `text`, or b"" on failure/empty input."""
    text = (text or "").strip()
    if not text:
        return b""
    try:
        mp3 = _synthesize_mp3(text, _pick_voice(text))
    except Exception as exc:  # noqa: BLE001 — never crash the call
        logger.warning("TTS (edge-tts) failed: %s", exc)
        return b""
    if not mp3:
        return b""
    try:
        return _mp3_to_mulaw(mp3)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MP3 decode/convert failed: %s", exc)
        return b""
