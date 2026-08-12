"""Twilio voice webhook: returns TwiML that opens a bidirectional media stream.

When someone calls the Twilio number, Twilio issues POST /voice. We reply with
<Connect><Stream> pointing at our /media WebSocket, which keeps the call connected
and streams the caller's audio to us (and lets us stream audio back in Phase 5).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import Connect, VoiceResponse

from app.config import settings

logger = logging.getLogger("voice.telephony")

router = APIRouter()


@router.post("/voice")
async def voice(request: Request) -> Response:
    form = await request.form()
    logger.info("Incoming call: From=%s CallSid=%s", form.get("From"), form.get("CallSid"))

    response = VoiceResponse()
    response.say("You are now connected. Please start speaking.")

    connect = Connect()
    connect.stream(url=settings.stream_ws_url)
    response.append(connect)

    return Response(content=str(response), media_type="application/xml")
