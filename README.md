<<<<<<< HEAD
# Voice Emotion Intelligence

A real-time voice bot that detects the speaker's **emotion** as they talk — in
**Hindi, Hinglish, or English** — shows it live on a polished web dashboard, and
replies conversationally in the speaker's language (text + voice).

**Stack:** Python · FastAPI · Sarvam AI (STT + translation + LLM) · HuggingFace
(local emotion models) · edge-tts (free voice) · vanilla HTML/JS dashboard

> **Full setup & run instructions:** see [HOW_TO_RUN.md](HOW_TO_RUN.md)

## How it works

```
Mic (browser 🎤 button or script) / phone call (Twilio)
  → voice-activity detection (splits speech on pauses)
  → Sarvam STT  (Hindi / Hinglish / English transcript)
  → Emotion:
      1) Sarvam-105b-conversations LLM — context-aware classification  (primary)
      2) local fallback: Sarvam-translate → English 7-emotion model
  → Live dashboard  (emotion ring, distribution bars, transcript bubbles,
                     emotion-over-time chart, session stats)
  → Bot reply:
      1) Anthropic Claude       (if ANTHROPIC_API_KEY set)
      2) Sarvam-105b LLM        (free — real conversational replies)   ← default
      3) emotion/language templates (offline fallback)
  → Voice: browser speech synthesis (dashboard) / edge-tts (phone calls)
```

## Key features

- **Talk from the browser** — 🎤 button on the dashboard streams your mic
  (16 kHz PCM) straight to the server; no phone or extra script needed.
- **Context-aware emotion** — the LLM understands situations ("everyone stood
  there with a cake" → joy), not just emotion keywords.
- **Multilingual** — speak Hindi, Hinglish, or English; the bot replies in kind.
- **Live dashboard** — animated confidence ring, per-emotion distribution bars,
  chat-style transcript with emotion chips, mood-over-time chart, mic level
  meter, light/dark themes, female bot voice.
- **Free-tier friendly** — runs on Sarvam's free key + local models + edge-tts.
  Token budgets are kept tight (emotion ≤60, reply ≤120 tokens per turn).
- **Phone-call ready** — Twilio `<Connect><Stream>` path included (`/voice` +
  `/media`) for real inbound calls via a tunnel (cloudflared/ngrok).

## Requirements

- **Python 3.11 or 3.12** (not 3.13 — stdlib `audioop` was removed there)
- A free **Sarvam API key** → https://dashboard.sarvam.ai
- Optional: Anthropic key (premium replies), Twilio account (real phone calls)

## Quick start

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env       # put your SARVAM_API_KEY in .env
python run.py                # then open http://127.0.0.1:8123/
```

Click **🎤 Start talking** on the dashboard and speak.

## Project layout

```
app/
  main.py            FastAPI app: routes, /health, warmup
  config.py          settings loaded from .env
  telephony/         /voice TwiML webhook + /media (Twilio) & /mic (browser) WebSockets
  stt/               Sarvam STT + translation client
  emotion/           LLM classifier, local models, fusion, engine
  bot/               reply generation (Claude → Sarvam-M → templates), edge-tts, Sarvam chat client
  audio/             μ-law / PCM16 / resample helpers
  dashboard/         WebSocket broadcast to browsers
  static/index.html  the dashboard UI
scripts/
  mic_emotion.py     hi-fi terminal mic client (alternative to the browser button)
  simulate_call.py   simulates Twilio's 8 kHz phone protocol locally
tests/
  offline_pipeline.py  run the emotion engine on text/WAV, no server
  test_emotion.py      unit tests (fusion, taxonomy)
  check_sarvam_llm.py  diagnostic for Sarvam LLM access/models
```

## Configuration highlights (`.env`)

| Key | Recommended | Why |
|-----|-------------|-----|
| `SARVAM_API_KEY` | *(your key)* | STT + translation + LLM |
| `SARVAM_LANGUAGE` | `unknown` | auto-detect Hindi/English per sentence |
| `EMOTION_LLM` | `true` | context-aware emotion via Sarvam LLM |
| `EMOTION_LLM_MODEL` | `sarvam-105b-conversations` | returns clean replies fast (non-reasoning) |
| `EMOTION_USE_TRANSLATION` | `true` | fallback path: translate → English model |
| `ENABLE_AUDIO_EMOTION` | `false` | tone model is noisy on mic audio; text is far more accurate |
| `ENABLE_BOT_REPLY` | `true` | conversational replies on |
| `ANTHROPIC_API_KEY` | *(optional)* | premium replies via Claude |
=======
# voice-emotion-intelligence
>>>>>>> b361f7ff26a8a430d9611a341f59be2212958a70
