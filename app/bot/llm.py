"""Generate the bot's spoken reply with Claude, adapted to the caller's emotion.

Replies are kept short and plain (no emojis/markdown) because they will be spoken
aloud by the TTS engine. The caller's detected emotion is injected so the model can
de-escalate anger, comfort sadness, match happiness, etc.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import List, Optional

from app.config import settings

_SARVAM_REPLY_SYSTEM = (
    "You are a warm, empathetic voice assistant on a live phone call. "
    "Understand what the caller actually says and respond helpfully and naturally — "
    "answer their questions, acknowledge their situation. "
    "Reply in the SAME language they used (Hindi, Hinglish, or English). "
    "Keep it to one or two short spoken sentences. No emojis, markdown, or lists."
)

logger = logging.getLogger("voice.bot.llm")

_client = None

SYSTEM_PROMPT = (
    "You are a warm, empathetic voice assistant on a live phone call. "
    "Each caller message is prefixed with their detected emotion in brackets. "
    "Adapt your tone to it: if they are angry, stay calm, apologize, and de-escalate; "
    "if they are sad or fearful, be gentle and reassuring; if they are happy, match "
    "their energy. "
    "Reply in the SAME language the caller used — if they speak Hindi, reply in Hindi; "
    "if they speak Hinglish (Hindi written in Latin letters), reply in Hinglish; "
    "if English, reply in English. "
    "Keep every reply very short — one or two sentences — natural and conversational, "
    "because it will be spoken aloud. Do not use emojis, markdown, or special characters."
)

# Free, no-API-key fallback replies: emotion- and language-aware templates used when
# the Anthropic API is unavailable (no key / no credit). Several phrasings per emotion
# so the bot doesn't repeat the same line; one is chosen at random (avoiding a repeat).
_LOCAL_REPLIES = {
    "en": {
        "anger": [
            "I hear that you're upset, and I'm sorry. Let's fix this together.",
            "Your frustration is completely valid. Tell me more and I'll help.",
            "I understand you're angry. Let's stay calm and work this out.",
            "I'm sorry for the trouble. How can I make this right?",
        ],
        "sadness": [
            "I'm really sorry you're feeling this way. I'm here for you.",
            "That sounds genuinely hard. You're not alone in this.",
            "I'm sorry to hear that. Take heart — I'm right here with you.",
            "That's tough, and it's okay to feel this way. I'm listening.",
        ],
        "fear": [
            "Take a slow breath, it's alright. I'm right here with you.",
            "Don't worry, you're safe. I'm here, tell me what's wrong.",
            "It's okay to feel scared. We'll get through this together.",
            "Stay calm, one step at a time. I've got you.",
        ],
        "joy": [
            "That's wonderful to hear! I'm so glad for you.",
            "Awesome — that's really great news!",
            "I love that! Thanks for sharing it with me.",
            "That's fantastic! Tell me more.",
        ],
        "surprise": [
            "Oh wow! Tell me more about that.",
            "Really? That's fascinating — go on.",
            "That's quite surprising! What happened next?",
        ],
        "disgust": [
            "I understand, that sounds unpleasant. Let's sort it out.",
            "That's really off-putting. Tell me how I can help.",
            "I get it, that's not okay. Let's make it better.",
        ],
        "neutral": [
            "Got it. Please tell me more.",
            "I see — I'm listening. Go on.",
            "Okay, I understand. Please continue.",
            "Sure, tell me a bit more about that.",
        ],
    },
    "hi": {
        "anger": [
            "मैं समझ सकता हूँ कि आप नाराज़ हैं, और मुझे खेद है। आइए इसे मिलकर सुलझाते हैं।",
            "आपका गुस्सा बिल्कुल जायज़ है। मुझे बताइए, मैं मदद करता हूँ।",
            "मैं समझता हूँ आप परेशान हैं। शांति रखिए, हम हल निकालेंगे।",
            "तकलीफ़ के लिए माफ़ी चाहता हूँ। बताइए, मैं कैसे मदद करूँ?",
        ],
        "sadness": [
            "मुझे खेद है कि आप ऐसा महसूस कर रहे हैं। मैं आपके साथ हूँ।",
            "यह सुनकर सच में दुख हुआ। आप अकेले नहीं हैं, मैं यहीं हूँ।",
            "मैं समझ सकता हूँ यह कठिन है। हिम्मत रखिए, सब ठीक होगा।",
            "आपकी बात सुनकर बुरा लगा। मैं आपके साथ हूँ, बताइए।",
        ],
        "fear": [
            "चिंता मत कीजिए, गहरी साँस लीजिए। मैं आपके साथ हूँ।",
            "घबराइए मत, सब ठीक होगा। मैं यहीं हूँ, बताइए क्या हुआ।",
            "डरने की ज़रूरत नहीं, हम इसे मिलकर संभालेंगे।",
            "शांत रहिए, एक-एक कदम। मैं आपके साथ हूँ।",
        ],
        "joy": [
            "यह सुनकर बहुत अच्छा लगा!",
            "वाह, यह तो बढ़िया खबर है!",
            "बहुत खूब! मुझे आपके लिए खुशी है।",
            "सुनकर दिल खुश हो गया! और बताइए।",
        ],
        "surprise": [
            "अरे वाह! मुझे और बताइए।",
            "सच में? यह तो दिलचस्प है!",
            "ओह, यह तो हैरानी की बात है! आगे कहिए।",
        ],
        "disgust": [
            "मैं समझता हूँ, यह अच्छा नहीं लगा। आइए इसे ठीक करते हैं।",
            "यह वाकई अप्रिय है। मुझे बताइए, मैं मदद करता हूँ।",
            "समझ सकता हूँ, यह ठीक नहीं। इसे सुधारते हैं।",
        ],
        "neutral": [
            "ठीक है, कृपया और बताइए।",
            "जी, मैं सुन रहा हूँ। आगे कहिए।",
            "अच्छा, समझ गया। कृपया जारी रखिए।",
            "जी हाँ, इसके बारे में थोड़ा और बताइए।",
        ],
    },
}


def _is_placeholder_key(key: str) -> bool:
    return (not key) or key.startswith("your_")


def _local_reply(emotion_label: str, transcript: str, avoid: Optional[str] = None) -> str:
    """Emotion + language aware reply, picked at random (never the same as `avoid`)."""
    try:
        from app.emotion.text_emotion import looks_non_english

        lang = "hi" if looks_non_english(transcript) else "en"
    except Exception:  # noqa: BLE001
        lang = "en"
    table = _LOCAL_REPLIES[lang]
    options = table.get(emotion_label) or table["neutral"]
    if avoid and len(options) > 1:
        options = [o for o in options if o != avoid] or options
    return random.choice(options)


def _last_bot_reply(history: Optional[List[dict]]) -> Optional[str]:
    for m in reversed(history or []):
        if m.get("role") == "assistant":
            return m.get("content")
    return None


def _get_client():
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic

        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


async def generate_reply(
    transcript: str,
    emotion_label: str = "neutral",
    history: Optional[List[dict]] = None,
) -> str:
    """Return a short spoken reply. `history` is prior [{role, content}] turns."""
    avoid = _last_bot_reply(history)  # don't repeat the previous template line

    # 1) Anthropic Claude, if a real key is configured.
    if not _is_placeholder_key(settings.anthropic_api_key):
        messages = (history or []) + [
            {"role": "user", "content": f"[caller emotion: {emotion_label}] {transcript}"}
        ]
        try:
            resp = await _get_client().messages.create(
                model=settings.anthropic_model, max_tokens=150,
                system=SYSTEM_PROMPT, messages=messages,
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            logger.warning("Anthropic reply failed (%s)", exc)

    # 2) Sarvam-M (uses the Sarvam key) — real conversational reply in the caller's language.
    if settings.bot_use_sarvam and settings.sarvam_api_key:
        reply = await _sarvam_reply(transcript, emotion_label, history)
        if reply:
            return reply

    # 3) Free emotion/language templates.
    return _local_reply(emotion_label, transcript, avoid)


async def _sarvam_reply(transcript: str, emotion_label: str, history: Optional[List[dict]]) -> Optional[str]:
    from app.bot import sarvam_chat

    messages = (
        [{"role": "system", "content": _SARVAM_REPLY_SYSTEM}]
        + (history or [])[-6:]
        + [{"role": "user", "content": f"(caller sounds {emotion_label}) {transcript}"}]
    )
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            None, lambda: sarvam_chat.chat(messages, temperature=0.5, max_tokens=120)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sarvam reply failed: %s", exc)
        return None
