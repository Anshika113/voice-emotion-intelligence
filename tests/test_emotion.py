"""Unit tests for the emotion engine.

The fusion + taxonomy logic is tested WITHOUT downloading any model (fast, offline).
Model-backed tests are opt-in via RUN_MODEL_TESTS=1 so CI stays light.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.emotion.fusion import fuse  # noqa: E402
from app.emotion.types import (  # noqa: E402
    CANONICAL_EMOTIONS,
    EmotionResult,
    empty_scores,
    top_label,
)

def _result(source: str, **label_scores) -> EmotionResult:
    scores = empty_scores()
    scores.update(label_scores)
    label = top_label(scores)
    return EmotionResult(label=label, score=scores[label], scores=scores, source=source)

# ---- taxonomy ----

def test_empty_scores_covers_all_labels():
    scores = empty_scores()
    assert set(scores) == set(CANONICAL_EMOTIONS)
    assert all(v == 0.0 for v in scores.values())

def test_top_label_picks_max():
    scores = empty_scores()
    scores["anger"] = 0.7
    scores["joy"] = 0.2
    assert top_label(scores) == "anger"

# ---- fusion ----

def test_fuse_none_none_returns_none():
    assert fuse(None, None) is None

def test_fuse_text_only_passes_through():
    text = _result("text", joy=0.9, neutral=0.1)
    fused = fuse(text, None)
    assert fused is not None
    assert fused.source == "fused"
    assert fused.label == "joy"

def test_fuse_audio_only_passes_through():
    audio = _result("audio", anger=0.8, neutral=0.2)
    fused = fuse(None, audio)
    assert fused.label == "anger"
    assert fused.source == "fused"

def test_fuse_weighted_average():
    # text says joy, audio says anger; default weights 0.6/0.4 -> joy wins
    text = _result("text", joy=1.0)
    audio = _result("audio", anger=1.0)
    fused = fuse(text, audio)
    assert fused.scores["joy"] == pytest.approx(0.6)
    assert fused.scores["anger"] == pytest.approx(0.4)
    assert fused.label == "joy"

def test_fuse_audio_can_override_with_high_weight():
    text = _result("text", joy=1.0)
    audio = _result("audio", anger=1.0)
    fused = fuse(text, audio, text_weight=0.3, audio_weight=0.7)
    assert fused.label == "anger"
    assert fused.scores["anger"] == pytest.approx(0.7)

def test_fuse_scores_are_a_distribution_over_canonical():
    text = _result("text", joy=0.5, sadness=0.5)
    audio = _result("audio", sadness=0.6, neutral=0.4)
    fused = fuse(text, audio)
    assert set(fused.scores) == set(CANONICAL_EMOTIONS)

# ---- optional model-backed smoke tests ----

@pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1",
    reason="set RUN_MODEL_TESTS=1 to run model download tests",
)
def test_text_model_detects_joy():
    from app.emotion.text_emotion import analyze_text

    res = analyze_text("I am so happy, this is wonderful!")
    assert res is not None
    assert res.label in {"joy", "surprise"}