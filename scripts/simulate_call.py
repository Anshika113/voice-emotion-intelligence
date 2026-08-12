"""Simulate a Twilio phone call against the local server — no Twilio, no phone.

Connects to the /media WebSocket and speaks Twilio's Media Streams protocol
(connected -> start -> media... -> stop), sending 8 kHz mono G.711 μ-law audio in
20 ms frames, exactly like Twilio does. The server can't tell the difference.

Modes:
    python scripts/simulate_call.py --mic                 # talk into your mic (default)
    python scripts/simulate_call.py --mic --seconds 20    # mic, auto-stop after 20s
    python scripts/simulate_call.py --wav sample.wav      # stream a WAV file

While it runs, open the dashboard (http://localhost:8080/) to watch emotion +
transcript update live. The bot's spoken reply is saved to bot_reply.wav.

Requirements:
    pip install sounddevice        # only needed for --mic
    (websockets, numpy, soundfile already installed)
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys

import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.audio import convert  # noqa: E402

SR = 8000
FRAME_MS = 20
PCM_BYTES_PER_FRAME = SR * 2 * FRAME_MS // 1000  # 320 bytes PCM16 per 20 ms
STREAM_SID = "MZsimulated0001"
CALL_SID = "CAsimulated0001"

# ---------- outbound (bot) audio receiver ----------

async def _receive_bot_audio(ws, out_buffer: bytearray) -> None:
    try:
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("event") == "media":
                payload = msg.get("media", {}).get("payload")
                if payload:
                    out_buffer.extend(base64.b64decode(payload))
    except Exception:  # noqa: BLE001
        pass

def _save_bot_reply(mulaw: bytes, path: str) -> None:
    if not mulaw:
        print("(no bot audio received)")
        return
    import numpy as np
    import soundfile as sf

    pcm = convert.mulaw_to_pcm16(mulaw)
    samples = np.frombuffer(pcm, dtype=np.int16)
    sf.write(path, samples, SR)
    print(f"Bot reply audio saved to {path} ({len(samples) / SR:.1f}s)")

# ---------- Twilio frame helpers ----------

async def _send_start(ws) -> None:
    await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
    await ws.send(
        json.dumps(
            {
                "event": "start",
                "start": {
                    "streamSid": STREAM_SID,
                    "callSid": CALL_SID,
                    "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": SR, "channels": 1},
                },
            }
        )
    )

async def _send_frame(ws, mulaw_frame: bytes, seq: int) -> None:
    await ws.send(
        json.dumps(
            {
                "event": "media",
                "streamSid": STREAM_SID,
                "media": {
                    "track": "inbound",
                    "chunk": str(seq),
                    "timestamp": str(seq * FRAME_MS),
                    "payload": base64.b64encode(mulaw_frame).decode("ascii"),
                },
            }
        )
    )

async def _send_stop(ws) -> None:
    await ws.send(json.dumps({"event": "stop", "streamSid": STREAM_SID}))

async def _send_silence(ws, seconds: float, seq_start: int) -> int:
    """Send trailing μ-law silence so AssemblyAI finalizes the turn (0xFF = silence)."""
    seq = seq_start
    frame = b"\xff" * (SR * FRAME_MS // 1000)
    for _ in range(int(seconds * 1000 / FRAME_MS)):
        seq += 1
        await _send_frame(ws, frame, seq)
        await asyncio.sleep(FRAME_MS / 1000)
    return seq

async def _wait_for_reply(buf: bytearray, timeout: float = 25.0) -> None:
    """Wait until the bot reply audio arrives and stops growing (or timeout)."""
    print("Waiting for emotion + bot reply…")
    waited = 0.0
    last = 0
    stable = 0.0
    while waited < timeout:
        await asyncio.sleep(0.5)
        waited += 0.5
        if len(buf) > 0 and len(buf) == last:
            stable += 0.5
            if stable >= 1.5:  # no new audio for 1.5s -> reply finished
                return
        else:
            stable = 0.0
            last = len(buf)

# ---------- source: WAV file ----------

def _wav_frames(path: str):
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(path, dtype="int16", always_2d=True)
    mono = data[:, 0]  # first channel
    pcm8, _ = convert.resample_pcm16(mono.astype(np.int16).tobytes(), sr, SR)
    for i in range(0, len(pcm8), PCM_BYTES_PER_FRAME):
        frame = pcm8[i : i + PCM_BYTES_PER_FRAME]
        if len(frame) < PCM_BYTES_PER_FRAME:
            frame = frame + b"\x00" * (PCM_BYTES_PER_FRAME - len(frame))
        yield convert.pcm16_to_mulaw(frame)

async def _stream_wav(ws, path: str) -> None:
    print(f"Streaming {path} …")
    seq = 0
    for mulaw in _wav_frames(path):
        seq += 1
        await _send_frame(ws, mulaw, seq)
        await asyncio.sleep(FRAME_MS / 1000)  # real-time pacing
    await _send_silence(ws, 1.5, seq)  # trigger end-of-turn while the socket is open
    print("WAV finished.")

# ---------- source: microphone ----------

async def _stream_mic(ws, seconds: int) -> None:
    import queue

    import sounddevice as sd

    q: "queue.Queue[bytes]" = queue.Queue()

    def callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    print("🎤 Speak now! (Ctrl+C to stop)" + (f" — auto-stop in {seconds}s" if seconds else ""))
    seq = 0
    capture_sr = 16000  # capture hi-fi, then downsample to 8 kHz for cleaner audio
    state = None
    loop = asyncio.get_event_loop()
    start = loop.time()
    with sd.RawInputStream(
        samplerate=capture_sr,
        blocksize=capture_sr * FRAME_MS // 1000,
        channels=1,
        dtype="int16",
        callback=callback,
    ):
        try:
            while True:
                if seconds and (loop.time() - start) >= seconds:
                    break
                try:
                    pcm = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.005)
                    continue
                seq += 1
                pcm8, state = convert.resample_pcm16(pcm, capture_sr, SR, state)
                await _send_frame(ws, convert.pcm16_to_mulaw(pcm8), seq)
        except KeyboardInterrupt:
            pass
    print(f"\nStopping; sent {seq} audio frames (~{seq * FRAME_MS / 1000:.1f}s of audio).")
    if seq == 0:
        print("WARNING: 0 frames captured — the mic produced no audio (check input device).")
    await _send_silence(ws, 1.5, seq)  # trigger end-of-turn while the socket is open

# ---------- main ----------

async def main_async(args) -> None:
    ws_url = f"ws://{args.host}:{args.port}/media"
    print(f"Connecting to {ws_url}")
    async with websockets.connect(ws_url, max_size=None) as ws:
        bot_audio = bytearray()
        recv_task = asyncio.create_task(_receive_bot_audio(ws, bot_audio))

        await _send_start(ws)
        if args.wav:
            await _stream_wav(ws, args.wav)
        else:
            await _stream_mic(ws, args.seconds)

        # Keep the socket OPEN and wait for the bot reply before hanging up.
        await _wait_for_reply(bot_audio)
        await _send_stop(ws)
        await asyncio.sleep(0.3)

        recv_task.cancel()
        _save_bot_reply(bytes(bot_audio), args.out)

def main() -> int:
    ap = argparse.ArgumentParser(description="Simulate a Twilio call against the local server")
    ap.add_argument("--wav", help="stream a WAV file instead of the microphone")
    ap.add_argument("--mic", action="store_true", help="use the microphone (default)")
    ap.add_argument("--seconds", type=int, default=0, help="mic auto-stop after N seconds (0 = until Ctrl+C)")
    ap.add_argument("--host", default="127.0.0.1", help="server host (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8080, help="server port (default 8080)")
    ap.add_argument("--out", default="bot_reply.wav", help="where to save the bot's spoken reply")
    args = ap.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass
    return 0

if __name__ == "__main__":
    raise SystemExit(main())