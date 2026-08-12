# 🎙️ Voice Emotion Intelligence

> **Real-time multilingual voice emotion detection and conversational AI using Python, FastAPI, Sarvam AI, and HuggingFace.**

Voice Emotion Intelligence is an AI-powered voice application that listens to a speaker through a **browser microphone**, converts speech into text, detects the speaker's emotional state in real time, and responds conversationally in the speaker's language.

The application provides a live dashboard showing the detected emotion, confidence score, transcript, emotion distribution, mood progression, and microphone activity.

It supports **Hindi, Hinglish, and English** conversations.

---

## ✨ Features

### 🎤 Real-Time Voice Interaction

* Capture speech directly from the browser microphone.
* Real-time voice activity detection.
* Speech is processed sentence-by-sentence after natural pauses.
* No separate microphone application is required.

### 🌐 Multilingual AI

Supports:

* 🇮🇳 Hindi
* Hinglish
* 🇬🇧 English

The system can automatically detect the language and generate responses in the speaker's language.

### 🧠 Context-Aware Emotion Detection

Emotion is primarily detected using the **Sarvam `sarvam-105b-conversations` LLM**, allowing the system to understand context rather than relying only on emotion keywords.

A local HuggingFace-based emotion model is available as a fallback.

Supported emotion categories include:

| Emotion     | Description                        |
| ----------- | ---------------------------------- |
| 😊 Joy      | Positive / happy emotional state   |
| 😠 Anger    | Frustration, irritation, anger     |
| 😢 Sadness  | Sad or emotionally low state       |
| 😨 Fear     | Fear, anxiety, nervousness         |
| 😲 Surprise | Unexpected or surprising situation |
| 🤢 Disgust  | Dislike or aversion                |
| 😐 Neutral  | No strong emotional signal         |

### 📊 Live Emotion Dashboard

The dashboard provides:

* Current emotion
* Emotion confidence
* Emotion distribution
* Live transcript
* Emotion-over-time visualization
* Session information
* Microphone activity
* AI-generated response
* Voice response controls
* Light / dark theme

### 🔊 Conversational Voice Response

The detected emotion and transcript are used to generate a conversational response.

Response pipeline:

```text
Anthropic Claude
       ↓
Sarvam 105B LLM
       ↓
Template fallback
```

Voice output uses:

* Browser Speech Synthesis for the web dashboard
* `edge-tts` for the optional phone-call flow

---

# 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │   Browser Microphone │
                    │        🎤            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Voice Activity       │
                    │ Detection            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Sarvam STT       │
                    │ Hindi / Hinglish /   │
                    │ English              │
                    └──────────┬───────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │      Emotion Detection       │
                │                              │
                │ Sarvam 105B LLM              │
                │            │                 │
                │            ▼                 │
                │ Local HuggingFace Fallback   │
                └──────────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Live Dashboard     │
                    │                      │
                    │ Emotion + Confidence │
                    │ Transcript           │
                    │ Distribution         │
                    │ Emotion Timeline     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Conversational LLM   │
                    │ Response Generation  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Voice Reply     │
                    │ Browser Speech /     │
                    │ edge-tts             │
                    └──────────────────────┘
```

---

# 🛠️ Technology Stack

| Layer              | Technology                          |
| ------------------ | ----------------------------------- |
| Language           | Python 3.11 / 3.12                  |
| Backend            | FastAPI                             |
| Server             | Uvicorn                             |
| Speech-to-Text     | Sarvam AI                           |
| Emotion Detection  | Sarvam 105B LLM                     |
| Emotion Fallback   | HuggingFace local model             |
| LLM Response       | Sarvam 105B / Anthropic Claude      |
| Text-to-Speech     | Browser Speech Synthesis / edge-tts |
| Frontend           | HTML, CSS, JavaScript               |
| Communication      | WebSockets                          |
| Optional Telephony | Twilio                              |
| Development Tunnel | Cloudflare Tunnel                   |

Python **3.11 or 3.12** is recommended because the project uses audio functionality affected by the removal of `audioop` in Python 3.13.

---

# 📁 Project Structure

```text
voice-emotion-intelligence/
│
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── telephony/
│   │   ├── twiml.py
│   │   └── media_ws.py
│   │
│   ├── stt/
│   │   └── ...
│   │
│   ├── emotion/
│   │   ├── ...
│   │
│   ├── bot/
│   │   ├── ...
│   │
│   ├── audio/
│   │   └── ...
│   │
│   ├── dashboard/
│   │   └── ...
│   │
│   └── static/
│       └── index.html
│
├── scripts/
│   └── mic_emotion.py
│
├── tests/
│   ├── test_emotion.py
│   ├── offline_pipeline.py
│   └── check_sarvam_llm.py
│
├── requirements.txt
├── .env.example
├── .gitignore
├── run.py
└── README.md
```

---

# 🚀 Getting Started

## 1. Prerequisites

Install:

* Python 3.11 or 3.12
* A Sarvam AI API key
* Working microphone
* Chrome or Microsoft Edge
* Internet connection

A Sarvam API key is required because speech-to-text and the primary AI processing use Sarvam's cloud APIs.

---

## 2. Clone the Repository

```bash
git clone https://github.com/Anshika113/voice-emotion-intelligence.git

cd voice-emotion-intelligence
```

---

## 3. Create a Virtual Environment

### Windows

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1
```

You should now see:

```text
(.venv)
```

in your terminal.

---

## 4. Install Dependencies

```powershell
python -m pip install --upgrade pip
```

```powershell
python -m pip install -r requirements.txt
```

If microphone support requires it:

```powershell
python -m pip install sounddevice
```

The first run may download local fallback emotion models and cache them locally.

---

# 🔐 Environment Configuration

Create your environment file:

```powershell
Copy-Item .env.example .env
```

Open it:

```powershell
notepad .env
```

Add your Sarvam API key:

```env
SARVAM_API_KEY=your_real_sarvam_key

SARVAM_LANGUAGE=unknown

EMOTION_LLM=true
EMOTION_LLM_MODEL=sarvam-105b-conversations

EMOTION_USE_TRANSLATION=true

ENABLE_AUDIO_EMOTION=false

ENABLE_BOT_REPLY=true

PORT=8123
HOST=127.0.0.1
```

Optional:

```env
ANTHROPIC_API_KEY=your_anthropic_key
```

**Never commit `.env` to GitHub.**

Use `.env.example` as the public configuration template.

---

# ▶️ Run the Application

Start the FastAPI server:

```powershell
python run.py
```

Wait until the application finishes warming up.

Then open:

```text
http://127.0.0.1:8123/
```

Click:

**🎤 Start talking**

Allow microphone access and begin speaking.

For better sentence-level processing, speak a sentence, pause briefly, and then continue.

---

# 🖥️ Dashboard

The live dashboard displays:

```text
┌─────────────────────────────────────────────────────┐
│ Current Emotion                                     │
│                                                     │
│              😖  Disgust                            │
│                 51%                                 │
│                                                     │
├───────────────────┬─────────────────────────────────┤
│ Microphone        │ Live Transcript & Timeline      │
│                   │                                 │
│       🎤          │ User → transcript               │
│                   │ AI → response                   │
│ Start Talking     │                                 │
├───────────────────┴─────────────────────────────────┤
│ Emotion Distribution                                │
│                                                     │
│ Joy       ███████                                   │
│ Anger     █████                                     │
│ Sadness   ███                                       │
│ Fear      ███                                       │
│ Surprise  ██                                        │
│ Disgust   █████████████                             │
│ Neutral   ███████████                               │
│                                                     │
├─────────────────────────────────────────────────────┤
│ Emotion Over Time                                   │
│                                                     │
│      ────────────────╲                              │
│                       ╲                             │
│                        ╲                            │
└─────────────────────────────────────────────────────┘
```

---

# 🧪 Testing

## Check Sarvam LLM

```powershell
python tests/check_sarvam_llm.py
```

This can be used to diagnose model access and authentication.

---

## Test Emotion Detection Without Microphone

```powershell
python tests/offline_pipeline.py --text "मुझे बहुत गुस्सा आ रहा है"
```

Example English input:

```powershell
python tests/offline_pipeline.py --text "I am so excited for tomorrow"
```

---

## Run Unit Tests

Install pytest:

```powershell
python -m pip install pytest
```

Run:

```powershell
python -m pytest tests/test_emotion.py -q
```

---

# 📞 Optional: Phone Call Integration

The **browser microphone is the primary interaction mode**.

The project also contains an optional Twilio-based phone-call path.

For phone-call testing, the local server can be exposed through Cloudflare Tunnel and connected to a Twilio phone number.

High-level flow:

```text
Phone Call
    ↓
Twilio
    ↓
FastAPI
    ↓
Audio WebSocket
    ↓
Speech-to-Text
    ↓
Emotion Detection
    ↓
LLM Response
    ↓
edge-tts
    ↓
Twilio
    ↓
Caller
```

The phone-call functionality requires additional Twilio configuration and a public tunnel. The browser dashboard does **not** require a phone number.

---

# ⚙️ Configuration Options

| Variable                  | Purpose                            |
| ------------------------- | ---------------------------------- |
| `SARVAM_API_KEY`          | Sarvam API authentication          |
| `SARVAM_LANGUAGE`         | Speech language configuration      |
| `EMOTION_LLM`             | Enable LLM-based emotion detection |
| `EMOTION_LLM_MODEL`       | Emotion classification model       |
| `EMOTION_USE_TRANSLATION` | Enable translation fallback        |
| `ENABLE_AUDIO_EMOTION`    | Enable audio/tone emotion model    |
| `ENABLE_BOT_REPLY`        | Enable conversational response     |
| `ANTHROPIC_API_KEY`       | Optional Claude integration        |
| `PORT`                    | FastAPI server port                |
| `HOST`                    | Server host                        |

For the current implementation, `ENABLE_AUDIO_EMOTION=false` is recommended because the text/LLM emotion pipeline is more reliable for the current microphone workflow.

---

# 🔄 End-to-End Processing

```text
1. User speaks
       ↓
2. Browser captures microphone audio
       ↓
3. Voice activity detection identifies speech segments
       ↓
4. Sarvam converts speech → text
       ↓
5. Sarvam LLM analyzes context → emotion
       ↓
6. Local HuggingFace model provides fallback emotion
       ↓
7. Dashboard receives emotion + transcript
       ↓
8. LLM generates contextual response
       ↓
9. Browser converts response → speech
```

---

# 💡 Why This Project?

Traditional sentiment analysis often looks only at text.

Voice conversations contain additional information:

* What the person says
* The context of the statement
* Changes in emotional state over time
* How the conversation progresses

This project explores how **speech recognition + LLM-based reasoning + emotion classification + conversational AI** can be combined into a real-time voice intelligence system.

---

# 🎯 Current Status

### ✅ Implemented

* [x] Browser microphone input
* [x] Real-time speech processing
* [x] Hindi support
* [x] Hinglish support
* [x] English support
* [x] Sarvam speech-to-text
* [x] Context-aware emotion detection
* [x] Local emotion fallback
* [x] Live emotion dashboard
* [x] Emotion confidence
* [x] Emotion distribution
* [x] Emotion timeline
* [x] Conversational AI response
* [x] Browser voice response
* [x] Offline emotion testing
* [x] Unit testing
* [x] Optional Twilio integration

### 🔬 Future Improvements

* [ ] Production deployment
* [ ] Persistent conversation analytics
* [ ] User authentication
* [ ] Multi-user sessions
* [ ] Better multilingual emotion models
* [ ] Improved audio/tone emotion detection
* [ ] Call-level analytics
* [ ] Emotion-aware conversation summaries
* [ ] Production-grade telephony scaling

---

# ⚠️ Limitations

Emotion detection is **probabilistic** and should not be treated as ground truth.

The current system primarily relies on the semantic information in the transcript. Audio/tone emotion detection is disabled by default because the current tone model can introduce noisy results.

The local fallback emotion models are also more suitable for English than multilingual emotion understanding.

---

# 🔒 Security

Never commit:

```text
.env
API keys
access tokens
private credentials
```

The repository should contain only:

```text
.env.example
```

with placeholder values.

---

# 📌 Project Information

**Project:** Voice Emotion Intelligence

**Repository:**
https://github.com/Anshika113/voice-emotion-intelligence

**Developer:** Anshika Mishra

**GitHub:**
https://github.com/Anshika113

---

## ⭐ If You Find This Project Interesting

Star the repository and explore the implementation.

This project demonstrates practical integration of:

**Speech AI → NLP → LLMs → Emotion Detection → WebSockets → Conversational AI → Voice Interfaces**
