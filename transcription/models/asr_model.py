# -*- coding: utf-8 -*-
"""
asr_model.py

Cherokee ASR model encapsulation and procedural inference pipeline.
Re-exports CherokeeASRModel from transcription.cherokee.models for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from transcription.cherokee.models import (
    DEFAULT_CHEROKEE_FALLBACK_REPO,
    CherokeeASRModel,
    load_cherokee_asr_model,
)

TARGET_SAMPLE_RATE = 16000
FRAME_DURATION_SEC = 0.02


@dataclass
class WordConfidence:
    word: str
    confidence: float
    start_time: float
    end_time: float
    chars: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ASRResult:
    text: str
    transcription: str
    confidence: float
    words: List[WordConfidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "transcription": self.transcription,
            "confidence": self.confidence,
            "words": [
                w.to_dict() if isinstance(w, WordConfidence) else w for w in self.words
            ],
        }


__all__ = [
    "CherokeeASRModel",
    "load_cherokee_asr_model",
    "ASRResult",
    "WordConfidence",
    "DEFAULT_CHEROKEE_FALLBACK_REPO",
    "TARGET_SAMPLE_RATE",
    "FRAME_DURATION_SEC",
]
