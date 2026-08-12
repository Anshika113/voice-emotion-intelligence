"""Audio format glue between Twilio, Sarvam STT, and the ML models.

Twilio speaks G.711 μ-law @ 8 kHz. Sarvam STT wants PCM16 (as a WAV). The tone model
wants a float32 waveform. These helpers translate between those.

Uses the stdlib `audioop` module (removed in Python 3.13 — target 3.11/3.12, or
install `audioop-lts`). We import it defensively so the failure message is clear.
"""
from __future__ import annotations

from typing import Optional, Tuple

try:
    import audioop  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "The stdlib `audioop` module is missing. It was removed in Python 3.13. "
        "Use Python 3.11/3.12, or `pip install audioop-lts`."
    ) from exc

import numpy as np

SAMPLE_WIDTH = 2  # PCM16 = 2 bytes/sample
CHANNELS = 1

def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """G.711 μ-law bytes -> linear PCM16 bytes (same sample rate)."""
    return audioop.ulaw2lin(mulaw_bytes, SAMPLE_WIDTH)

def pcm16_to_mulaw(pcm_bytes: bytes) -> bytes:
    """Linear PCM16 bytes -> G.711 μ-law bytes (same sample rate)."""
    return audioop.lin2ulaw(pcm_bytes, SAMPLE_WIDTH)

def resample_pcm16(
    pcm_bytes: bytes,
    sr_in: int,
    sr_out: int,
    state: Optional[object] = None,
) -> Tuple[bytes, object]:
    """Resample PCM16. Returns (converted_bytes, state) — pass `state` back in for
    the next chunk of a continuous stream to avoid boundary clicks."""
    if sr_in == sr_out:
        return pcm_bytes, state
    converted, new_state = audioop.ratecv(pcm_bytes, SAMPLE_WIDTH, CHANNELS, sr_in, sr_out, state)
    return converted, new_state

def pcm16_to_float_array(pcm_bytes: bytes) -> np.ndarray:
    """PCM16 bytes -> float32 numpy waveform in [-1, 1] (mono)."""
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0