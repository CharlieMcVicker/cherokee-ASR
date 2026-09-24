# -*- coding: utf-8 -*-
"""
digohwelisgi.core.codeswitching.types

Algebraic domain models, tokens, loss records, and pure protocols for
cross-lingual phonetic mapping, grapheme-to-phoneme extraction, and acoustic
confusion matrix alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
    runtime_checkable,
)

# Special token used to represent epsilon / null transitions in alignment
EPSILON_TOKEN: str = "<eps>"

# 39 standard CMU / LibriSpeech ARPAbet phonemes (stress stripped)
STANDARD_ARPABET_VOWELS: Tuple[str, ...] = (
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
)

STANDARD_ARPABET_CONSONANTS: Tuple[str, ...] = (
    "B",
    "CH",
    "D",
    "DH",
    "F",
    "G",
    "HH",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
)

STANDARD_ARPABET_PHONEMES: Tuple[str, ...] = (
    STANDARD_ARPABET_VOWELS + STANDARD_ARPABET_CONSONANTS
)


class StressPattern(int, Enum):
    """Stress pattern for ARPAbet vowel phonemes."""

    NO_STRESS = 0
    PRIMARY = 1
    SECONDARY = 2


@dataclass(frozen=True)
class ARPAbetPhone:
    """
    Immutable representation of an ARPAbet phoneme.

    Normalizes phone tokens to uppercase. Strips trailing stress digits if present
    in the phone string and records them in the optional stress field.
    """

    phone: str
    stress: Optional[int] = None

    def __post_init__(self) -> None:
        raw_phone = self.phone.strip()
        if raw_phone in ("", "<eps>", "<EPS>", "EPS", "eps"):
            object.__setattr__(self, "phone", EPSILON_TOKEN)
            object.__setattr__(self, "stress", None)
            return

        m = re.match(r"^([A-Za-z]+)(\d)?$", raw_phone)
        if m:
            phone_part = m.group(1).upper()
            stress_part = int(m.group(2)) if m.group(2) is not None else self.stress
            object.__setattr__(self, "phone", phone_part)
            object.__setattr__(self, "stress", stress_part)
        else:
            object.__setattr__(self, "phone", raw_phone.upper())

    @property
    def is_epsilon(self) -> bool:
        return self.phone == EPSILON_TOKEN

    @property
    def is_vowel(self) -> bool:
        return self.phone in STANDARD_ARPABET_VOWELS

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"phone": self.phone}
        if self.stress is not None:
            d["stress"] = self.stress
        return d

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], str, ARPAbetPhone]) -> ARPAbetPhone:
        if isinstance(data, ARPAbetPhone):
            return data
        if isinstance(data, str):
            return cls(phone=data)
        if isinstance(data, dict):
            return cls(phone=str(data["phone"]), stress=data.get("stress"))
        if hasattr(data, "phone"):
            return cls(
                phone=str(getattr(data, "phone")), stress=getattr(data, "stress", None)
            )
        raise ValueError(f"Cannot construct ARPAbetPhone from {data}")

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> ARPAbetPhone:
        return cls.from_dict(json.loads(s))

    def __str__(self) -> str:
        return f"{self.phone}{self.stress}" if self.stress is not None else self.phone


# Alias for backward compatibility / naming preference
ArpabetToken = ARPAbetPhone


@dataclass(frozen=True)
class WordManifestEntry:
    """
    Immutable manifest record for a single-word audio clip (LibriSpeech or similar).
    """

    clip_id: str
    audio_path: str
    word: str
    duration: float
    arpabet: Tuple[ARPAbetPhone, ...]
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    speaker_id: Optional[str] = None

    @property
    def arpabet_phones(self) -> Tuple[str, ...]:
        return tuple(tok.phone for tok in self.arpabet)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "clip_id": self.clip_id,
            "audio_path": self.audio_path,
            "word": self.word,
            "duration": self.duration,
            "arpabet": [t.to_dict() for t in self.arpabet],
        }
        if self.start_sec is not None:
            d["start_sec"] = self.start_sec
        if self.end_sec is not None:
            d["end_sec"] = self.end_sec
        if self.speaker_id is not None:
            d["speaker_id"] = self.speaker_id
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WordManifestEntry:
        raw_arpabet = data.get("arpabet", [])
        arpabet_tokens: List[ARPAbetPhone] = []
        for item in raw_arpabet:
            if isinstance(item, ARPAbetPhone):
                arpabet_tokens.append(item)
            else:
                arpabet_tokens.append(ARPAbetPhone.from_dict(item))
        return cls(
            clip_id=str(data["clip_id"]),
            audio_path=str(data["audio_path"]),
            word=str(data["word"]),
            duration=float(data["duration"]),
            arpabet=tuple(arpabet_tokens),
            start_sec=(
                float(data["start_sec"])
                if "start_sec" in data and data["start_sec"] is not None
                else None
            ),
            end_sec=(
                float(data["end_sec"])
                if "end_sec" in data and data["end_sec"] is not None
                else None
            ),
            speaker_id=(
                str(data["speaker_id"])
                if "speaker_id" in data and data["speaker_id"] is not None
                else None
            ),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> WordManifestEntry:
        return cls.from_dict(json.loads(s))


@dataclass(frozen=True)
class TargetPhone:
    """
    Immutable representation of a generic target language acoustic/phonetic token.
    """

    phone: str

    def __post_init__(self) -> None:
        raw_phone = self.phone.strip()
        if raw_phone in ("", "<eps>", "<EPS>", "EPS", "eps"):
            object.__setattr__(self, "phone", EPSILON_TOKEN)
        else:
            object.__setattr__(self, "phone", raw_phone)

    @property
    def is_epsilon(self) -> bool:
        return self.phone == EPSILON_TOKEN

    def to_dict(self) -> Dict[str, Any]:
        return {"phone": self.phone}

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], str]) -> TargetPhone:
        if isinstance(data, str):
            return cls(phone=data)
        return cls(phone=str(data["phone"]))

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> TargetPhone:
        return cls.from_dict(json.loads(s))

    def __str__(self) -> str:
        return self.phone


@dataclass(frozen=True)
class ConfusionEntry:
    """A single conditional substitution entry in an acoustic confusion matrix."""

    source_phone: str
    target_phone: str
    probability: float
    log_cost: float


@dataclass(frozen=True)
class SubstitutionMapping:
    """Mapping of a source phoneme/ngram to candidate target phonemes/ngrams."""

    source_key: str
    candidates: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)

    @property
    def best_target(self) -> Tuple[str, float]:
        if not self.candidates:
            return ("", 0.0)
        return self.candidates[0]


@dataclass(frozen=True)
class JointNgram:
    """Joint N-gram transition representing multi-token source to target mapping."""

    source_tokens: Tuple[str, ...]
    target_tokens: Tuple[str, ...]
    probability: float = 1.0
    cost: float = 0.0

    @property
    def source_key(self) -> str:
        return " ".join(self.source_tokens)

    @property
    def target_key(self) -> str:
        return "".join(self.target_tokens)


@dataclass(frozen=True)
class CrossEntropyLoss:
    """Cross entropy loss value and alignment metrics for confusion matrix optimization."""

    loss: float
    mean_cost: float
    num_samples: int
    iteration: int = 0


@dataclass(frozen=True)
class AlignmentPath:
    """A single alignment step in DP traceback."""

    source_tokens: Tuple[str, ...] = field(default_factory=tuple)
    target_tokens: Tuple[str, ...] = field(default_factory=tuple)
    cost: float = 0.0
    confidence: float = 1.0

    @property
    def is_substitution(self) -> bool:
        return len(self.source_tokens) > 0 and len(self.target_tokens) > 0

    @property
    def is_insertion(self) -> bool:
        return len(self.source_tokens) == 0 and len(self.target_tokens) > 0

    @property
    def is_deletion(self) -> bool:
        return len(self.source_tokens) > 0 and len(self.target_tokens) == 0

    @property
    def source_key(self) -> str:
        return " ".join(self.source_tokens)

    @property
    def target_key(self) -> str:
        return "".join(self.target_tokens)


@dataclass(frozen=True)
class BeamHypothesis:
    """Beam search hypothesis for emitted phonetic tokens."""

    rank: int
    text: str
    score: float
    tokens: Tuple[str, ...] = field(default_factory=tuple)
    token_confidences: Tuple[float, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "text": self.text,
            "score": self.score,
            "tokens": list(self.tokens),
            "token_confidences": list(self.token_confidences),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BeamHypothesis:
        return cls(
            rank=int(data.get("rank", 1)),
            text=str(data["text"]),
            score=float(data["score"]),
            tokens=tuple(str(t) for t in data.get("tokens", [])),
            token_confidences=tuple(
                float(c) for c in data.get("token_confidences", [])
            ),
        )


@dataclass(frozen=True)
class ProjectedTarget:
    """
    Projected synthetic phonetic target derived from an English word
    via ARPAbet conversion and calibrated confusion matrix mapping.
    """

    source_word: str
    arpabet_tokens: Tuple[ARPAbetPhone, ...]
    target_tokens: Tuple[str, ...]
    projected_text: str
    confidence_score: float = 1.0
    per_token_probabilities: Tuple[float, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def projected_tth(self) -> str:
        """Alias for projected_text for Cherokee / alignment consumers."""
        return self.projected_text

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "source_word": self.source_word,
            "arpabet_tokens": [t.to_dict() for t in self.arpabet_tokens],
            "target_tokens": list(self.target_tokens),
            "projected_text": self.projected_text,
            "projected_tth": self.projected_text,
            "confidence_score": self.confidence_score,
            "per_token_probabilities": list(self.per_token_probabilities),
            "metadata": dict(self.metadata),
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProjectedTarget:
        raw_arpabet = data.get("arpabet_tokens", [])
        arp_tokens = tuple(
            t if isinstance(t, ARPAbetPhone) else ARPAbetPhone.from_dict(t)
            for t in raw_arpabet
        )

        raw_targets = data.get("target_tokens")
        if raw_targets is None:
            raw_cherokee = data.get("cherokee_tokens", [])
            target_tokens = tuple(
                (
                    str(t["phone"])
                    if isinstance(t, dict) and "phone" in t
                    else (getattr(t, "phone", str(t)))
                )
                for t in raw_cherokee
            )
        else:
            target_tokens = tuple(str(t) for t in raw_targets)

        projected = str(data.get("projected_text", data.get("projected_tth", "")))

        return cls(
            source_word=str(data["source_word"]),
            arpabet_tokens=arp_tokens,
            target_tokens=target_tokens,
            projected_text=projected,
            confidence_score=float(data.get("confidence_score", 1.0)),
            per_token_probabilities=tuple(
                float(p) for p in data.get("per_token_probabilities", [])
            ),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> ProjectedTarget:
        return cls.from_dict(json.loads(s))


@dataclass(frozen=True)
class TracebackResult:
    """Result of DP traceback alignment between source and target phonemes."""

    paths: Tuple[AlignmentPath, ...]
    total_cost: float
    normalized_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": [
                {
                    "source_tokens": list(p.source_tokens),
                    "target_tokens": list(p.target_tokens),
                    "cost": p.cost,
                    "confidence": p.confidence,
                }
                for p in self.paths
            ],
            "total_cost": self.total_cost,
            "normalized_cost": self.normalized_cost,
        }


# ============================================================================
# Pure Protocols
# ============================================================================


@runtime_checkable
class EnglishToArpabetProtocol(Protocol):
    """Protocol for converting English text to ARPAbet phoneme sequence."""

    def __call__(
        self, text: str, strip_stress: bool = True
    ) -> Tuple[ARPAbetPhone, ...]: ...

    def extract(
        self, text: str, strip_stress: bool = True
    ) -> Tuple[ARPAbetPhone, ...]: ...


@runtime_checkable
class GenericConfusionMatrixProtocol(Protocol):
    """Protocol representing an acoustic confusion matrix."""

    def get_substitution_cost(
        self, source: Union[str, Sequence[str]], target: Union[str, Sequence[str]]
    ) -> float: ...

    def get_insertion_cost(self, target: Union[str, Sequence[str]]) -> float: ...

    def get_deletion_cost(self, source: Union[str, Sequence[str]]) -> float: ...

    def get_deletion_probability(self, source: Union[str, Sequence[str]]) -> float: ...

    def get_insertion_probability(self, target: Union[str, Sequence[str]]) -> float: ...

    def get_probability(
        self, source: Union[str, Sequence[str]], target: Union[str, Sequence[str]]
    ) -> float: ...

    def best_target_for(
        self, source: Union[str, Sequence[str]]
    ) -> Tuple[str, float]: ...

    def has_transition(self, source: Union[str, Sequence[str]]) -> bool: ...


@runtime_checkable
class GenericSyntheticTargetProjectorProtocol(Protocol):
    """Protocol for projecting words or ARPAbet phonemes to target language phonetics."""

    def project_word(
        self,
        word: str,
        matrix: Optional[GenericConfusionMatrixProtocol] = None,
    ) -> Any: ...

    def project_arpabet(
        self,
        arpabet_tokens: Sequence[ARPAbetPhone],
        matrix: Optional[GenericConfusionMatrixProtocol] = None,
        source_word: str = "",
    ) -> Any: ...

    def project_english_text(
        self,
        text: str,
        matrix: Optional[GenericConfusionMatrixProtocol] = None,
    ) -> str: ...


__all__ = [
    "EPSILON_TOKEN",
    "STANDARD_ARPABET_VOWELS",
    "STANDARD_ARPABET_CONSONANTS",
    "STANDARD_ARPABET_PHONEMES",
    "StressPattern",
    "ARPAbetPhone",
    "ArpabetToken",
    "WordManifestEntry",
    "TargetPhone",
    "ConfusionEntry",
    "SubstitutionMapping",
    "JointNgram",
    "CrossEntropyLoss",
    "AlignmentPath",
    "BeamHypothesis",
    "ProjectedTarget",
    "TracebackResult",
    "EnglishToArpabetProtocol",
    "GenericConfusionMatrixProtocol",
    "GenericSyntheticTargetProjectorProtocol",
]
