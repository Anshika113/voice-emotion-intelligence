"""Diagnostic: find a Sarvam model + token budget that returns real `content`.

    python tests/check_sarvam_llm.py
"""
from __future__ import annotations

import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402

URL = "https://api.sarvam.ai/v1/chat/completions"
key = settings.sarvam_api_key
print("Sarvam key present:", bool(key), "len:", len(key))
print("=" * 64)

MSG = [{"role": "user", "content": "Reply with exactly one word: hello"}]

for model in ["sarvam-105b-conversations", "sarvam-105b"]:
    for mt in [100, 600]:
        print(f"\n### model={model}  max_tokens={mt}")
        try:
            r = requests.post(
                URL,
                headers={"api-subscription-key": key, "Content-Type": "application/json"},
                json={"model": model, "messages": MSG, "max_tokens": mt, "temperature": 0},
                timeout=40,
            )
            print("status:", r.status_code)
            if r.status_code == 200:
                ch = r.json()["choices"][0]
                msg = ch.get("message", {})
                print("  finish_reason:", ch.get("finish_reason"))
                print("  content:", repr(msg.get("content")))
                rc = msg.get("reasoning_content")
                if rc:
                    print("  reasoning (first 100):", repr(rc[:100]))
            else:
                print("  body:", r.text[:300])
        except Exception as exc:  # noqa: BLE001
            print("  ERROR:", exc)
        print("-" * 64)