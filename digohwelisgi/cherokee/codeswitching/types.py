# -*- coding: utf-8 -*-
"""
digohwelisgi.cherokee.codeswitching.types

Cherokee-specific domain models, tokens, and target representations
for code-switched alignment and cross-lingual loanword projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from digohwelisgi.cherokee.orthography.orthography import Orthography
from digohwelisgi.core.codeswitching.types import (
    EPSILON_TOKEN,
    ARPAbetPhone,
    ArpabetToken,
    GenericConfusionMatrixProtocol,
    GenericSyntheticTargetProjectorProtocol,
    ProjectedTarget,
)

# Canonical Cherokee TTH acoustic phonemes (CherokeeASRModel emission vocabulary)
CANONICAL_CHEROKEE_VOWELS: Tuple[str, ...] = (
    "a",
    "e",
    "i",
    "o",
    "u",
    "v",
)

CANONICAL_CHEROKEE_CONSONANTS: Tuple[str, ...] = (
    "t",
    "th",
    "k",
    "kh",
    "kw",
    "kwh",
    "tl",
    "tlh",
    "lh",
    "ts",
    "tsh",
    "s",
    "hs",
    "nh",
    "wh",
    "yh",
    "w",
    "y",
    "h",
    "m",
    "n",
    "'",
)

CANONICAL_CHEROKEE_TTH_PHONEMES: Tuple[str, ...] = (
    CANONICAL_CHEROKEE_VOWELS + CANONICAL_CHEROKEE_CONSONANTS
)
CANONICAL_CHEROKEE_PHONEMES: Tuple[str, ...] = CANONICAL_CHEROKEE_TTH_PHONEMES


@dataclass(frozen=True)
class CherokeeToken:
    """
    Immutable representation of a Cherokee phonetic or syllabary token.
    Defaults to canonical T/TH acoustic phonetics (CherokeeASRModel emission vocabulary).
    """

    phone: str
    orthography: Orthography = Orthography.TTH

    def __post_init__(self) -> None:
        raw_phone = self.phone.strip()
        if raw_phone in ("", "<eps>", "<EPS>", "EPS", "eps"):
            object.__setattr__(self, "phone", EPSILON_TOKEN)
            return

        if self.orthography == Orthography.TTH:
            object.__setattr__(self, "phone", raw_phone.lower())
        else:
            object.__setattr__(self, "phone", raw_phone)

    @property
    def is_epsilon(self) -> bool:
        return self.phone == EPSILON_TOKEN

    @property
    def is_vowel(self) -> bool:
        return self.phone in CANONICAL_CHEROKEE_VOWELS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phone": self.phone,
            "orthography": self.orthography.value,
        }

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], str]) -> CherokeeToken:
        if isinstance(data, str):
            return cls(phone=data)
        ortho = Orthography(data.get("orthography", Orthography.TTH.value))
        return cls(phone=str(data["phone"]), orthography=ortho)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> CherokeeToken:
        return cls.from_dict(json.loads(s))

    def __str__(self) -> str:
        return self.phone


@dataclass(frozen=True)
class SyntheticCherokeeTarget(ProjectedTarget):
    """
    Projected synthetic Cherokee phonetic target derived from an English word
    via ARPAbet conversion and calibrated confusion matrix mapping.
    """

    source_word: str
    arpabet_tokens: Tuple[Union[ArpabetToken, ARPAbetPhone], ...]
    cherokee_tokens: Tuple[CherokeeToken, ...] = field(default_factory=tuple)
    target_tokens: Tuple[str, ...] = field(default_factory=tuple)
    projected_text: str = ""
    syllabary: Optional[str] = None
    confidence_score: float = 1.0
    per_token_probabilities: Tuple[float, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Harmonize target_tokens and cherokee_tokens
        if not self.target_tokens and self.cherokee_tokens:
            object.__setattr__(
                self,
                "target_tokens",
                tuple(t.phone for t in self.cherokee_tokens),
            )
        elif self.target_tokens and not self.cherokee_tokens:
            object.__setattr__(
                self,
                "cherokee_tokens",
                tuple(
                    CherokeeToken(phone=t, orthography=Orthography.TTH)
                    for t in self.target_tokens
                ),
            )

    @property
    def projected_tth(self) -> str:
        return self.projected_text

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "source_word": self.source_word,
            "arpabet_tokens": [t.to_dict() for t in self.arpabet_tokens],
            "cherokee_tokens": [t.to_dict() for t in self.cherokee_tokens],
            "target_tokens": list(self.target_tokens),
            "projected_tth": self.projected_text,
            "projected_text": self.projected_text,
            "confidence_score": self.confidence_score,
            "per_token_probabilities": list(self.per_token_probabilities),
            "metadata": dict(self.metadata),
        }
        if self.syllabary is not None:
            d["syllabary"] = self.syllabary
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SyntheticCherokeeTarget:
        raw_arpabet = data.get("arpabet_tokens", [])
        arp_tokens = tuple(
            t if isinstance(t, ArpabetToken) else ArpabetToken.from_dict(t)
            for t in raw_arpabet
        )
        raw_cherokee = data.get("cherokee_tokens", [])
        chr_tokens = tuple(
            t if isinstance(t, CherokeeToken) else CherokeeToken.from_dict(t)
            for t in raw_cherokee
        )
        raw_targets = data.get("target_tokens")
        if raw_targets is not None:
            target_tokens = tuple(str(t) for t in raw_targets)
        else:
            target_tokens = tuple(t.phone for t in chr_tokens)

        projected = str(data.get("projected_tth", data.get("projected_text", "")))

        return cls(
            source_word=str(data.get("source_word", "")),
            arpabet_tokens=arp_tokens,
            cherokee_tokens=chr_tokens,
            target_tokens=target_tokens,
            projected_text=projected,
            syllabary=(
                str(data["syllabary"])
                if "syllabary" in data and data["syllabary"] is not None
                else None
            ),
            confidence_score=float(data.get("confidence_score", 1.0)),
            per_token_probabilities=tuple(
                float(p) for p in data.get("per_token_probabilities", [])
            ),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> SyntheticCherokeeTarget:
        return cls.from_dict(json.loads(s))


@dataclass(frozen=True)
class AlignedTokenPair:
    """
    A single alignment step in DP traceback between ARPAbet and Cherokee tokens.
    """

    arpabet: Optional[ArpabetToken] = None
    cherokee: Optional[CherokeeToken] = None
    arpabet_tokens: Tuple[ArpabetToken, ...] = field(default_factory=tuple)
    cherokee_tokens: Tuple[CherokeeToken, ...] = field(default_factory=tuple)
    cost: float = 0.0
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.arpabet_tokens and self.arpabet is not None:
            object.__setattr__(self, "arpabet_tokens", (self.arpabet,))
        elif self.arpabet_tokens and self.arpabet is None:
            if len(self.arpabet_tokens) == 1:
                object.__setattr__(self, "arpabet", self.arpabet_tokens[0])
            else:
                object.__setattr__(
                    self,
                    "arpabet",
                    ArpabetToken(" ".join(t.phone for t in self.arpabet_tokens)),
                )

        if not self.cherokee_tokens and self.cherokee is not None:
            object.__setattr__(self, "cherokee_tokens", (self.cherokee,))
        elif self.cherokee_tokens and self.cherokee is None:
            if len(self.cherokee_tokens) == 1:
                object.__setattr__(self, "cherokee", self.cherokee_tokens[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arpabet": self.arpabet.to_dict() if self.arpabet else None,
            "cherokee": self.cherokee.to_dict() if self.cherokee else None,
            "arpabet_tokens": [t.to_dict() for t in self.arpabet_tokens],
            "cherokee_tokens": [t.to_dict() for t in self.cherokee_tokens],
            "cost": self.cost,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AlignedTokenPair:
        raw_arp = data.get("arpabet")
        arp = (
            (
                raw_arp
                if isinstance(raw_arp, ArpabetToken)
                else ArpabetToken.from_dict(raw_arp)
            )
            if raw_arp is not None
            else None
        )
        raw_chr = data.get("cherokee")
        chr_tok = (
            (
                raw_chr
                if isinstance(raw_chr, CherokeeToken)
                else CherokeeToken.from_dict(raw_chr)
            )
            if raw_chr is not None
            else None
        )
        raw_arp_seq = data.get("arpabet_tokens", [])
        arp_seq = tuple(
            t if isinstance(t, ArpabetToken) else ArpabetToken.from_dict(t)
            for t in raw_arp_seq
        )
        raw_chr_seq = data.get("cherokee_tokens", [])
        chr_seq = tuple(
            t if isinstance(t, CherokeeToken) else CherokeeToken.from_dict(t)
            for t in raw_chr_seq
        )
        return cls(
            arpabet=arp,
            cherokee=chr_tok,
            arpabet_tokens=arp_seq,
            cherokee_tokens=chr_seq,
            cost=float(data.get("cost", 0.0)),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class TracebackAlignmentResult:
    """Immutable result of a DP traceback word alignment."""

    pairs: Tuple[AlignedTokenPair, ...]
    total_cost: float
    normalized_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pairs": [p.to_dict() for p in self.pairs],
            "total_cost": self.total_cost,
            "normalized_cost": self.normalized_cost,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TracebackAlignmentResult:
        raw_pairs = data.get("pairs", [])
        pairs = tuple(
            p if isinstance(p, AlignedTokenPair) else AlignedTokenPair.from_dict(p)
            for p in raw_pairs
        )
        return cls(
            pairs=pairs,
            total_cost=float(data["total_cost"]),
            normalized_cost=float(data["normalized_cost"]),
        )


@runtime_checkable
class SyntheticTargetProjectorProtocol(Protocol):
    """
    Protocol projecting English words or ARPAbet sequences
    into synthetic Cherokee phonetic targets.
    """

    def project_word(
        self,
        word: str,
        matrix: Optional[GenericConfusionMatrixProtocol] = None,
    ) -> SyntheticCherokeeTarget: ...

    def project_arpabet(
        self,
        arpabet_tokens: Sequence[Union[ARPAbetPhone, ArpabetToken]],
        matrix: Optional[GenericConfusionMatrixProtocol] = None,
        source_word: str = "",
    ) -> SyntheticCherokeeTarget: ...

    def project_english_text(
        self,
        text: str,
        matrix: Optional[GenericConfusionMatrixProtocol] = None,
    ) -> str: ...


__all__ = [
    "CANONICAL_CHEROKEE_VOWELS",
    "CANONICAL_CHEROKEE_CONSONANTS",
    "CANONICAL_CHEROKEE_TTH_PHONEMES",
    "CANONICAL_CHEROKEE_PHONEMES",
    "CherokeeToken",
    "SyntheticCherokeeTarget",
    "AlignedTokenPair",
    "TracebackAlignmentResult",
    "SyntheticTargetProjectorProtocol",
]
