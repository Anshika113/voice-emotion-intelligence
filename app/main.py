"""FastAPI application entry point.

Phase 1 wires up the app, config, and a health check. Later phases add the
Twilio webhook (/voice), the media WebSocket (/media), and the dashboard.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.dashboard.ws import manager as dashboard_manager
from app.telephony.media_ws import local_mic_ws, twilio_media_ws
from app.telephony.twiml import router as twiml_router

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("voice")

app = FastAPI(title="Voice Emotion Detection", version="0.1.0")

# Twilio voice webhook (POST /voice)
app.include_router(twiml_router)

@app.websocket("/media")
async def media(websocket: WebSocket) -> None:
    """Twilio Media Streams connect here (see app/telephony/media_ws.py)."""
    await twilio_media_ws(websocket)

@app.websocket("/mic")
async def mic(websocket: WebSocket) -> None:
    """Local hi-fi mic testing — raw PCM16 @ 16 kHz (best transcription accuracy)."""
    await local_mic_ws(websocket)

@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket) -> None:
    """Browsers connect here to receive live transcript + emotion updates."""
    await dashboard_manager.connect(websocket)
    try:
        while True:
            # We don't expect inbound messages; this keeps the socket open.
            await websocket.receive_text()
    except WebSocketDisconnect:
        dashboard_manager.disconnect(websocket)
    except Exception:  # noqa: BLE001
        dashboard_manager.disconnect(websocket)

def _warmup_models() -> None:
    """Load the emotion models in the background so the first utterance isn't slow.
    (STT is Sarvam — a cloud call — so there's no STT model to load.)"""
    try:
        import numpy as np

        from app.emotion import audio_emotion, text_emotion

        text_emotion.analyze_text("hello there")          # English emotion model
        text_emotion.analyze_text("नमस्ते कैसे हो")         # multilingual model
        if settings.enable_audio_emotion:
            audio_emotion.analyze_audio(np.zeros(8000, dtype=np.float32), 8000)  # tone model
        logger.info("Emotion models warmed up")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model warmup skipped: %s", exc)

@app.on_event("startup")
async def _on_startup() -> None:
    logger.info("Voice Emotion Detection starting up")
    logger.info("Public host: %s", settings.public_host)
    logger.info("Stream WS URL (for Twilio): %s", settings.stream_ws_url)
    logger.info(
        "Feature toggles: audio_emotion=%s bot_reply=%s",
        settings.enable_audio_emotion,
        settings.enable_bot_reply,
    )
    import threading

    threading.Thread(target=_warmup_models, daemon=True).start()

@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe. Also reports which integrations have keys configured."""
    return JSONResponse(
        {
            "status": "ok",
            "service": "voice-emotion-detection",
            "version": app.version,
            "config": {
                "public_host": settings.public_host,
                "sarvam_language": settings.sarvam_language,
                "sarvam_model": settings.sarvam_model,
                "emotion_use_translation": settings.emotion_use_translation,
                "enable_audio_emotion": settings.enable_audio_emotion,
                "enable_bot_reply": settings.enable_bot_reply,
                "keys_present": {
                    "sarvam": bool(settings.sarvam_api_key),
                    "anthropic": bool(settings.anthropic_api_key),
                    "twilio": bool(settings.twilio_account_sid and settings.twilio_auth_token),
                },
            },
        }
    )

@app.get("/")
async def root() -> FileResponse:
    """Serve the live emotion dashboard."""
    return FileResponse(STATIC_DIR / "index.html")