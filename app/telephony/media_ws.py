"""WebSocket handlers for audio input — the per-call orchestrator.

Two input paths share the same buffering/VAD/transcribe/emotion/reply logic:

  * /media  — Twilio Media Streams: base64 μ-law @ 8 kHz (real phone calls).
  * /mic    — local hi-fi testing: raw PCM16 @ 16 kHz binary frames (much better
              transcription, since the audio isn't crushed to telephone quality).

Flow per utterance: buffer audio -> energy VAD detects the pause -> Sarvam STT
transcribes (in a thread) -> emotion engine (text + tone) -> dashboard broadcast
-> emotion-aware reply (spoken back only on the Twilio path).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from app.audio import convert
from app.bot import llm, tts
from app.config import settings
from app.dashboard.ws import broadcast
from app.stt import sarvam_stt

logger = logging.getLogger("voice.media")

_SPEECH_RMS = 500.0          # PCM16 RMS above this counts as speech
_SILENCE_HANGOVER_MS = 800   # end the utterance after this much trailing silence
_MIN_UTTERANCE_MS = 400      # ignore blips shorter than this


def _rms(pcm16: bytes) -> float:
    if not pcm16:
        return 0.0
    a = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


class CallSession:
    """State + behaviour for a single call/session (Twilio or local mic)."""

    def __init__(self, websocket: WebSocket, input_sr: int = 8000, reply_audio: bool = True) -> None:
        self.ws = websocket
        self.input_sr = input_sr          # sample rate of incoming PCM16
        self.reply_audio = reply_audio     # stream TTS back over this socket?
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None
        # Current utterance audio (PCM16 @ input_sr) + VAD state.
        self._utterance_pcm = bytearray()
        self._has_speech = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        # LLM history + guards.
        self.history: list[dict] = []
        self._last_final = ""
        self._closed = False
        self._replying = False
        self._loop = asyncio.get_event_loop()

    # ---- audio ingest + VAD (shared by both paths) ----

    def _feed(self, pcm16: bytes) -> None:
        self._utterance_pcm.extend(pcm16)
        samples = len(pcm16) // 2
        frame_ms = samples / (self.input_sr / 1000.0)
        if _rms(pcm16) >= _SPEECH_RMS:
            self._has_speech = True
            self._speech_ms += frame_ms
            self._silence_ms = 0.0
        elif self._has_speech:
            self._silence_ms += frame_ms
            if self._silence_ms >= _SILENCE_HANGOVER_MS:
                self._end_utterance()

    async def handle_media(self, data: dict) -> None:
        """Twilio path: base64 μ-law @ 8 kHz."""
        payload = data.get("media", {}).get("payload")
        if payload:
            self._feed(convert.mulaw_to_pcm16(base64.b64decode(payload)))

    def handle_pcm(self, pcm16: bytes) -> None:
        """Local hi-fi path: raw PCM16 @ input_sr."""
        if pcm16:
            self._feed(pcm16)

    async def handle_start(self, data: dict) -> None:
        start = data.get("start", {})
        self.stream_sid = start.get("streamSid")
        self.call_sid = start.get("callSid")
        logger.info("Stream started: streamSid=%s callSid=%s", self.stream_sid, self.call_sid)

    def _end_utterance(self) -> None:
        pcm = bytes(self._utterance_pcm)
        spoke_ms = self._speech_ms
        self._utterance_pcm.clear()
        self._has_speech = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        if spoke_ms >= _MIN_UTTERANCE_MS:
            asyncio.create_task(self._process(pcm))

    async def flush(self) -> None:
        """Transcribe any trailing speech at end of call."""
        if self._has_speech and self._speech_ms >= _MIN_UTTERANCE_MS:
            pcm = bytes(self._utterance_pcm)
            self._utterance_pcm.clear()
            self._has_speech = False
            await self._process(pcm)

    # ---- transcribe -> emotion -> reply ----

    async def _process(self, pcm: bytes) -> None:
        try:
            text = await self._loop.run_in_executor(
                None, sarvam_stt.transcribe_pcm16, pcm, self.input_sr
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Transcription failed: %s", exc)
            return
        text = (text or "").strip()
        if not text:
            return

        norm = text.lower()
        if norm == self._last_final:
            return
        self._last_final = norm
        logger.info("FINAL   : %s", text)

        try:
            result = await self._loop.run_in_executor(None, self._run_engine, text, pcm)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Emotion analysis failed: %s", exc)
            return

        fused = result.get("fused") if result else None
        if fused:
            logger.info("EMOTION : %s (%.2f) | text=%r", fused["label"], fused["score"], text)

        await broadcast(
            {
                "type": "final",
                "transcript": text,
                "emotion": result,
                "callSid": self.call_sid,
                "ts": time.time(),
            }
        )

        if settings.enable_bot_reply and not self._replying:
            self._replying = True
            try:
                await self._speak_reply(text, fused)
            finally:
                self._replying = False

    def _run_engine(self, text: str, pcm: bytes) -> dict:
        """Blocking; runs in a thread executor."""
        from app.emotion.engine import analyze_utterance

        waveform = convert.pcm16_to_float_array(pcm) if pcm else None
        use_audio = settings.enable_audio_emotion and waveform is not None and len(waveform) > 0
        return analyze_utterance(
            text=text,
            waveform=waveform if use_audio else None,
            sample_rate=self.input_sr,
        )

    async def _speak_reply(self, transcript: str, fused: Optional[dict]) -> None:
        label = fused["label"] if fused else "neutral"
        reply = await llm.generate_reply(transcript, label, self.history)

        self.history.append({"role": "user", "content": transcript})
        self.history.append({"role": "assistant", "content": reply})
        del self.history[:-10]
        logger.info("BOT     : %s", reply)

        await broadcast({"type": "bot", "reply": reply, "callSid": self.call_sid})
        if self.reply_audio:
            mulaw = await self._loop.run_in_executor(None, tts.synthesize_mulaw, reply)
            await self.send_audio_to_twilio(mulaw)

    async def send_audio_to_twilio(self, mulaw: bytes) -> None:
        """Stream μ-law bytes back to the caller as Twilio outbound media frames."""
        if not mulaw or self.stream_sid is None or self._closed:
            return
        chunk_size = 640  # ~80 ms of 8 kHz μ-law
        try:
            for i in range(0, len(mulaw), chunk_size):
                if self._closed:
                    break
                chunk = mulaw[i : i + chunk_size]
                await self.ws.send_json(
                    {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": base64.b64encode(chunk).decode("ascii")},
                    }
                )
        except RuntimeError:
            self._closed = True  # caller hung up mid-reply

    async def close(self) -> None:
        self._closed = True
        logger.info("Call session closed: callSid=%s", self.call_sid)


async def twilio_media_ws(websocket: WebSocket) -> None:
    """FastAPI WebSocket endpoint at /media — Twilio Media Streams (8 kHz μ-law)."""
    await websocket.accept()
    session = CallSession(websocket, input_sr=8000, reply_audio=True)
    logger.info("Twilio media WebSocket accepted")
    try:
        while True:
            data = json.loads(await websocket.receive_text())
            event = data.get("event")
            if event == "start":
                await session.handle_start(data)
            elif event == "media":
                await session.handle_media(data)
            elif event == "stop":
                logger.info("Stream stopped by Twilio")
                await session.flush()
                break
            elif event == "connected":
                logger.debug("Twilio media protocol connected")
    except WebSocketDisconnect:
        logger.info("Twilio media WebSocket disconnected")
        await session.flush()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Media WebSocket error: %s", exc)
    finally:
        await session.close()


async def local_mic_ws(websocket: WebSocket) -> None:
    """FastAPI WebSocket endpoint at /mic — local hi-fi PCM16 @ 16 kHz (best accuracy)."""
    await websocket.accept()
    session = CallSession(websocket, input_sr=16000, reply_audio=False)
    session.stream_sid = "local-mic"
    session.call_sid = "local-mic"
    logger.info("Local mic WebSocket accepted (16 kHz PCM)")
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                session.handle_pcm(msg["bytes"])
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("event") == "stop":
                    await session.flush()
                    break
    except WebSocketDisconnect:
        await session.flush()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Mic WebSocket error: %s", exc)
    finally:
        await session.close()
