"""Hi-fi local mic -> emotion, no telephone downgrade.

Sends clean 16 kHz PCM16 straight to the server's /mic endpoint (no μ-law, no
8 kHz), so the STT gets full-quality audio and transcription is far more accurate
than the phone-simulating path. Watch the dashboard for live emotion + transcript.

Usage:
    python scripts/mic_emotion.py --seconds 20
    python scripts/mic_emotion.py                # until Ctrl+C

Requires: sounddevice (pip install sounddevice).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import websockets

SR = 16000
FRAME_MS = 20

async def main_async(args) -> None:
    ws_url = f"ws://{args.host}:{args.port}/mic"
    print(f"Connecting to {ws_url}")
    async with websockets.connect(ws_url, max_size=None) as ws:
        import queue

        import sounddevice as sd

        q: "queue.Queue[bytes]" = queue.Queue()

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                print(status, file=sys.stderr)
            q.put(bytes(indata))

        print("🎤 Speak now! (Ctrl+C to stop)" + (f" — auto-stop in {args.seconds}s" if args.seconds else ""))
        print("   Open the dashboard: http://127.0.0.1:%d/" % args.port)
        sent = 0
        loop = asyncio.get_event_loop()
        start = loop.time()
        with sd.RawInputStream(
            samplerate=SR, blocksize=SR * FRAME_MS // 1000, channels=1, dtype="int16", callback=callback
        ):
            try:
                while True:
                    if args.seconds and (loop.time() - start) >= args.seconds:
                        break
                    try:
                        pcm = q.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.005)
                        continue
                    await ws.send(pcm)  # raw PCM16 @ 16 kHz (binary)
                    sent += 1
            except KeyboardInterrupt:
                pass

        print(f"\nStopping; sent {sent} frames (~{sent * FRAME_MS / 1000:.1f}s). "
              "Finishing transcription…")
        await ws.send(json.dumps({"event": "stop"}))
        await asyncio.sleep(3)  # let the last utterance transcribe + show on dashboard

def main() -> int:
    ap = argparse.ArgumentParser(description="Hi-fi local mic -> emotion")
    ap.add_argument("--seconds", type=int, default=0, help="auto-stop after N seconds (0 = Ctrl+C)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass
    return 0

if __name__ == "__main__":
    raise SystemExit(main())