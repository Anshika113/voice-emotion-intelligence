"""Run the emotion engine on a WAV file and/or a sentence — no phone needed.

Usage (from the project root):
    python tests/offline_pipeline.py sample.wav
    python tests/offline_pipeline.py sample.wav --text "I am so happy right now"
    python tests/offline_pipeline.py --text "This is completely unacceptable"

First run downloads the HuggingFace models (a few hundred MB) and caches them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as a script from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.emotion.engine import analyze_utterance  # noqa: E402

def _load_wav(path: str):
    import soundfile as sf

    waveform, sr = sf.read(path, dtype="float32", always_2d=False)
    return waveform, sr

def main() -> int:
    ap = argparse.ArgumentParser(description="Offline emotion engine runner")
    ap.add_argument("wav", nargs="?", help="path to a WAV file (optional)")
    ap.add_argument("--text", default=None, help="transcript text (optional)")
    args = ap.parse_args()

    if not args.wav and not args.text:
        ap.error("provide a WAV file, --text, or both")

    waveform, sr = (None, 16000)
    if args.wav:
        if not os.path.exists(args.wav):
            print(f"WAV not found: {args.wav}", file=sys.stderr)
            return 2
        waveform, sr = _load_wav(args.wav)
        print(f"Loaded {args.wav}: {len(waveform)} samples @ {sr} Hz")

    result = analyze_utterance(text=args.text, waveform=waveform, sample_rate=sr)

    print("\n=== Emotion result ===")
    print(json.dumps(result, indent=2))
    fused = result.get("fused")
    if fused:
        print(f"\n>>> FINAL: {fused['label']}  ({fused['score']:.2f})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())