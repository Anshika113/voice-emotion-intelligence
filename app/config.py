"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Speech-to-text: Sarvam AI (Indian-language ASR) ----
    sarvam_api_key: str = ""
    sarvam_model: str = "saarika:v2.5"   # v2.5 supports Hinglish / code-mixing
    sarvam_language: str = "hi-IN"        # or "unknown" to auto-detect

    # For non-English speech, translate -> English and use the sharp English emotion
    # model (more accurate than the multilingual sentiment model). Falls back safely.
    emotion_use_translation: bool = True

    # Use an LLM (Sarvam-M) to classify emotion with context (much better on nuanced
    # sentences). Falls back to the local model if it fails. Uses the Sarvam key.
    emotion_llm: bool = True
    emotion_llm_model: str = "sarvam-105b-conversations"   # conversational Sarvam LLM

    # ---- Bot LLM ----
    # Prefer Anthropic if a key is set; else use Sarvam-M (your Sarvam key) for real
    # conversational replies; else free emotion/language templates.
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    bot_use_sarvam: bool = True

    # ---- Text-to-speech: edge-tts (free, no key). Voice per language. ----
    tts_voice_hindi: str = "hi-IN-SwaraNeural"
    tts_voice_english: str = "en-US-AriaNeural"

    # ---- Twilio (only for real phone calls; not needed for the mic path) ----
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""

    # ---- Server / networking ----
    public_host: str = "localhost:8123"
    host: str = "127.0.0.1"
    port: int = 8123

    # ---- Feature toggles ----
    enable_audio_emotion: bool = True
    enable_bot_reply: bool = True

    @property
    def stream_ws_url(self) -> str:
        """WebSocket URL Twilio streams media to (wss over the public host)."""
        return f"wss://{self.public_host}/media"


settings = Settings()
