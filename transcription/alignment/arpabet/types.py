# -*- coding: utf-8 -*-
"""
transcription.alignment.arpabet.types

Domain models, pure transformation protocols, and serialization specifications
for the ARPAbet-to-Cherokee phonetic mapping and acoustic alignment system.

Follows Types and Maps architectural principles:
- Strict algebraic domain modeling using immutable dataclasses (@dataclass(frozen=True)).
- Pure functional transformation protocols decoupled from I/O boundaries.
- Lossless JSON serialization/deserialization across all cache and matrix models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import (
    Any,
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

from transcription.utils.orthography import Orthography

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


# ============================================================================
# 1. Phonetic Tokens
# ============================================================================


@dataclass(frozen=True)
class ArpabetToken:
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
    def from_dict(cls, data: Union[Dict[str, Any], str]) -> ArpabetToken:
        if isinstance(data, str):
            return cls(phone=data)
        return cls(phone=str(data["phone"]), stress=data.get("stress"))

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> ArpabetToken:
        return cls.from_dict(json.loads(s))

    def __str__(self) -> str:
        return f"{self.phone}{self.stress}" if self.stress is not None else self.phone


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


# ============================================================================
# 2. Word Manifest
# ============================================================================


@dataclass(frozen=True)
class WordManifestEntry:
    """
    Immutable manifest record for a single-word audio clip (LibriSpeech or similar).
    """

    clip_id: str
    audio_path: str
    word: str
    duration: float
    arpabet: Tuple[ArpabetToken, ...]
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
        arpabet_tokens: List[ArpabetToken] = []
        for item in raw_arpabet:
            if isinstance(item, ArpabetToken):
                arpabet_tokens.append(item)
            else:
                arpabet_tokens.append(ArpabetToken.from_dict(item))
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


# ============================================================================
# 3. Model Inference & Cache Models
# ============================================================================


@dataclass(frozen=True)
class TopKHypothesis:
    """
    Immutable beam search hypothesis for an emitted word audio clip.
    """

    rank: int
    text: str
    score: float
    tokens: Tuple[CherokeeToken, ...] = field(default_factory=tuple)
    token_confidences: Tuple[float, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "text": self.text,
            "score": self.score,
            "tokens": [t.to_dict() for t in self.tokens],
            "token_confidences": list(self.token_confidences),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TopKHypothesis:
        raw_tokens = data.get("tokens", [])
        tokens: List[CherokeeToken] = []
        for t in raw_tokens:
            if isinstance(t, CherokeeToken):
                tokens.append(t)
            else:
                tokens.append(CherokeeToken.from_dict(t))
        return cls(
            rank=int(data.get("rank", 1)),
            text=str(data["text"]),
            score=float(data["score"]),
            tokens=tuple(tokens),
            token_confidences=tuple(
                float(c) for c in data.get("token_confidences", [])
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> TopKHypothesis:
        return cls.from_dict(json.loads(s))


@dataclass(frozen=True)
class WordInferenceCacheEntry:
    """
    Cached inference output for a single word audio clip through CherokeeASRModel.
    """

    clip_id: str
    word: str
    greedy_tokens: Tuple[CherokeeToken, ...]
    greedy_text: str
    token_confidences: Tuple[float, ...] = field(default_factory=tuple)
    top_hypotheses: Tuple[TopKHypothesis, ...] = field(default_factory=tuple)
    mean_confidence: float = 0.0
    duration: float = 0.0

    def __post_init__(self) -> None:
        if self.mean_confidence == 0.0 and self.token_confidences:
            object.__setattr__(
                self,
                "mean_confidence",
                sum(self.token_confidences) / len(self.token_confidences),
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "word": self.word,
            "greedy_tokens": [t.to_dict() for t in self.greedy_tokens],
            "greedy_text": self.greedy_text,
            "token_confidences": list(self.token_confidences),
            "top_hypotheses": [h.to_dict() for h in self.top_hypotheses],
            "mean_confidence": self.mean_confidence,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WordInferenceCacheEntry:
        raw_tokens = data.get("greedy_tokens", [])
        greedy_tokens: List[CherokeeToken] = []
        for t in raw_tokens:
            if isinstance(t, CherokeeToken):
                greedy_tokens.append(t)
            else:
                greedy_tokens.append(CherokeeToken.from_dict(t))

        raw_hyps = data.get("top_hypotheses", [])
        top_hypotheses: List[TopKHypothesis] = []
        for h in raw_hyps:
            if isinstance(h, TopKHypothesis):
                top_hypotheses.append(h)
            else:
                top_hypotheses.append(TopKHypothesis.from_dict(h))

        token_confidences = tuple(float(c) for c in data.get("token_confidences", []))
        mean_conf = float(data.get("mean_confidence", 0.0))

        return cls(
            clip_id=str(data["clip_id"]),
            word=str(data["word"]),
            greedy_tokens=tuple(greedy_tokens),
            greedy_text=str(data.get("greedy_text", "")),
            token_confidences=token_confidences,
            top_hypotheses=tuple(top_hypotheses),
            mean_confidence=mean_conf,
            duration=float(data.get("duration", 0.0)),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> WordInferenceCacheEntry:
        return cls.from_dict(json.loads(s))


@dataclass(frozen=True)
class InferenceCacheManifest:
    """
    Durable emissions cache manifest storing forward pass results for an ASR model checkpoint.
    """

    model_id: str
    created_at: str
    entries: Tuple[WordInferenceCacheEntry, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def get(self, clip_id: str) -> Optional[WordInferenceCacheEntry]:
        for e in self.entries:
            if e.clip_id == clip_id:
                return e
        return None

    @property
    def by_clip_id(self) -> Dict[str, WordInferenceCacheEntry]:
        return {e.clip_id: e for e in self.entries}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "created_at": self.created_at,
            "entries": [e.to_dict() for e in self.entries],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InferenceCacheManifest:
        raw_entries = data.get("entries", [])
        entries_list: List[WordInferenceCacheEntry] = []
        if isinstance(raw_entries, dict):
            for k, v in raw_entries.items():
                if isinstance(v, WordInferenceCacheEntry):
                    entries_list.append(v)
                else:
                    d = dict(v)
                    if "clip_id" not in d:
                        d["clip_id"] = k
                    entries_list.append(WordInferenceCacheEntry.from_dict(d))
        elif isinstance(raw_entries, (list, tuple)):
            for item in raw_entries:
                if isinstance(item, WordInferenceCacheEntry):
                    entries_list.append(item)
                else:
                    entries_list.append(WordInferenceCacheEntry.from_dict(item))

        return cls(
            model_id=str(data["model_id"]),
            created_at=str(data.get("created_at", "")),
            entries=tuple(entries_list),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> InferenceCacheManifest:
        return cls.from_dict(json.loads(s))

    def save(self, path: Union[Path, str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[Path, str]) -> InferenceCacheManifest:
        p = Path(path)
        return cls.from_json(p.read_text(encoding="utf-8"))


# ============================================================================
# 4. Acoustic Confusion Matrix
# ============================================================================


def _normalize_arpabet_key(
    arpabet: Union[str, ArpabetToken, Sequence[Union[str, ArpabetToken]]],
) -> str:
    """Normalizes an ARPAbet phoneme or sequence of phonemes into a lookup key."""
    if isinstance(arpabet, str):
        return arpabet.strip()
    elif isinstance(arpabet, ArpabetToken):
        return arpabet.phone
    elif isinstance(arpabet, (list, tuple)):
        return " ".join(
            t.phone if isinstance(t, ArpabetToken) else str(t).strip()
            for t in arpabet
            if (t.phone if isinstance(t, ArpabetToken) else str(t).strip())
            not in ("", "<eps>", "<EPS>", "eps", "EPS", EPSILON_TOKEN)
        )
    return str(arpabet)


def _normalize_cherokee_key(
    cherokee: Union[str, CherokeeToken, Sequence[Union[str, CherokeeToken]]],
) -> str:
    """Normalizes a Cherokee phoneme or sequence of phonemes into a target key."""
    if isinstance(cherokee, str):
        return cherokee.strip()
    elif isinstance(cherokee, CherokeeToken):
        return cherokee.phone
    elif isinstance(cherokee, (list, tuple)):
        return "".join(
            t.phone if isinstance(t, CherokeeToken) else str(t).strip()
            for t in cherokee
            if (t.phone if isinstance(t, CherokeeToken) else str(t).strip())
            not in ("", "<eps>", "<EPS>", "eps", "EPS", EPSILON_TOKEN)
        )
    return str(cherokee)


@dataclass(frozen=True)
class AcousticConfusionMatrix:
    """
    Empirical statistical mapping between ARPAbet tokens and Cherokee phonemes.
    Supports both 1-gram and multi-gram (e.g. 1-to-2, 2-to-1, 2-to-2) transitions.

    Stores:
    - Conditional substitution probabilities: P(cherokee | arpabet)
    - Negative log substitution costs: -log(P(cherokee | arpabet))
    - Insertion (epenthetic) probabilities and costs: P(cherokee | eps)
    - Deletion (coda loss) probabilities and costs: P(eps | arpabet)
    """

    model_id: str
    arpabet_vocab: Tuple[str, ...]
    cherokee_vocab: Tuple[str, ...]
    probabilities: Dict[str, Dict[str, float]]
    log_costs: Dict[str, Dict[str, float]]
    insertion_probabilities: Dict[str, float] = field(default_factory=dict)
    insertion_costs: Dict[str, float] = field(default_factory=dict)
    deletion_probabilities: Dict[str, float] = field(default_factory=dict)
    deletion_costs: Dict[str, float] = field(default_factory=dict)
    default_substitution_cost: float = 10.0
    default_insertion_cost: float = 5.0
    default_deletion_cost: float = 5.0
    prune_threshold: float = 0.05
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        model_id: str,
        probabilities: Dict[str, Dict[str, float]],
        insertion_probabilities: Optional[Dict[str, float]] = None,
        deletion_probabilities: Optional[Dict[str, float]] = None,
        arpabet_vocab: Optional[Sequence[str]] = None,
        cherokee_vocab: Optional[Sequence[str]] = None,
        prune_threshold: float = 0.05,
        iteration: int = 0,
        default_substitution_cost: float = 10.0,
        default_insertion_cost: float = 5.0,
        default_deletion_cost: float = 5.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AcousticConfusionMatrix:
        """
        Pure factory creating an AcousticConfusionMatrix with computed log-costs.
        """
        ins_probs = dict(insertion_probabilities or {})
        del_probs = dict(deletion_probabilities or {})

        # Compute substitution log-costs
        log_costs: Dict[str, Dict[str, float]] = {}
        all_cherokee: Set[str] = set(ins_probs.keys())

        for a_phone, c_map in probabilities.items():
            log_costs[a_phone] = {}
            for c_phone, prob in c_map.items():
                all_cherokee.add(c_phone)
                if prob > 0.0:
                    log_costs[a_phone][c_phone] = max(0.0, -math.log(max(prob, 1e-12)))
                else:
                    log_costs[a_phone][c_phone] = default_substitution_cost

        # Compute insertion log-costs
        ins_costs: Dict[str, float] = {}
        for c_phone, prob in ins_probs.items():
            if prob > 0.0:
                ins_costs[c_phone] = max(0.0, -math.log(max(prob, 1e-12)))
            else:
                ins_costs[c_phone] = default_insertion_cost

        # Compute deletion log-costs
        del_costs: Dict[str, float] = {}
        for a_phone, prob in del_probs.items():
            if prob > 0.0:
                del_costs[a_phone] = max(0.0, -math.log(max(prob, 1e-12)))
            else:
                del_costs[a_phone] = default_deletion_cost

        arp_v = (
            tuple(arpabet_vocab)
            if arpabet_vocab is not None
            else tuple(sorted(probabilities.keys()))
        )
        chr_v = (
            tuple(cherokee_vocab)
            if cherokee_vocab is not None
            else tuple(sorted(all_cherokee))
        )

        return cls(
            model_id=model_id,
            arpabet_vocab=arp_v,
            cherokee_vocab=chr_v,
            probabilities={k: dict(v) for k, v in probabilities.items()},
            log_costs=log_costs,
            insertion_probabilities=ins_probs,
            insertion_costs=ins_costs,
            deletion_probabilities=del_probs,
            deletion_costs=del_costs,
            default_substitution_cost=default_substitution_cost,
            default_insertion_cost=default_insertion_cost,
            default_deletion_cost=default_deletion_cost,
            prune_threshold=prune_threshold,
            iteration=iteration,
            metadata=dict(metadata or {}),
        )

    def get_substitution_cost(
        self,
        arpabet: Union[str, ArpabetToken, Sequence[Union[str, ArpabetToken]]],
        cherokee: Union[str, CherokeeToken, Sequence[Union[str, CherokeeToken]]],
    ) -> float:
        """Returns substitution cost -log P(cherokee | arpabet)."""
        a = _normalize_arpabet_key(arpabet)
        c = _normalize_cherokee_key(cherokee)
        if a in self.log_costs and c in self.log_costs[a]:
            return self.log_costs[a][c]
        # Fallback for unobserved 2-gram decomposition if present
        tokens = a.split()
        if len(tokens) == 2 and len(c) >= 2:
            c1, c2 = c[:1], c[1:]
            if tokens[0] in self.log_costs and tokens[1] in self.log_costs:
                return self.get_substitution_cost(
                    tokens[0], c1
                ) + self.get_substitution_cost(tokens[1], c2)
        return self.default_substitution_cost

    def get_probability(
        self,
        arpabet: Union[str, ArpabetToken, Sequence[Union[str, ArpabetToken]]],
        cherokee: Union[str, CherokeeToken, Sequence[Union[str, CherokeeToken]]],
    ) -> float:
        """Returns conditional probability P(cherokee | arpabet)."""
        a = _normalize_arpabet_key(arpabet)
        c = _normalize_cherokee_key(cherokee)
        if a in self.probabilities and c in self.probabilities[a]:
            return self.probabilities[a][c]
        return 0.0

    def has_transition(
        self,
        arpabet: Union[str, ArpabetToken, Sequence[Union[str, ArpabetToken]]],
    ) -> bool:
        """Returns True if the matrix contains transitions for the given phone or multi-gram key."""
        a = _normalize_arpabet_key(arpabet)
        return a in self.probabilities and len(self.probabilities[a]) > 0

    def get_insertion_cost(
        self,
        cherokee: Union[str, CherokeeToken, Sequence[Union[str, CherokeeToken]]],
    ) -> float:
        """Returns insertion cost -log P(cherokee | eps)."""
        c = _normalize_cherokee_key(cherokee)
        return self.insertion_costs.get(c, self.default_insertion_cost)

    def get_insertion_probability(
        self,
        cherokee: Union[str, CherokeeToken, Sequence[Union[str, CherokeeToken]]],
    ) -> float:
        """Returns insertion probability P(cherokee | eps)."""
        c = _normalize_cherokee_key(cherokee)
        return self.insertion_probabilities.get(c, 0.0)

    def get_deletion_cost(
        self,
        arpabet: Union[str, ArpabetToken, Sequence[Union[str, ArpabetToken]]],
    ) -> float:
        """Returns deletion cost -log P(eps | arpabet)."""
        a = _normalize_arpabet_key(arpabet)
        return self.deletion_costs.get(a, self.default_deletion_cost)

    def get_deletion_probability(
        self,
        arpabet: Union[str, ArpabetToken, Sequence[Union[str, ArpabetToken]]],
    ) -> float:
        """Returns deletion probability P(eps | arpabet)."""
        a = _normalize_arpabet_key(arpabet)
        return self.deletion_probabilities.get(a, 0.0)

    def best_cherokee_for(
        self,
        arpabet: Union[str, ArpabetToken, Sequence[Union[str, ArpabetToken]]],
    ) -> Tuple[str, float]:
        """
        Returns the argmax Cherokee token sequence and its probability for the given ARPAbet phoneme(s).
        """
        a = _normalize_arpabet_key(arpabet)
        targets = self.probabilities.get(a, {})
        if targets:
            best_c, best_p = max(targets.items(), key=lambda kv: kv[1])
            return (best_c, best_p)
        tokens = a.split()
        if len(tokens) > 1:
            parts = [self.best_cherokee_for(t) for t in tokens]
            best_c = "".join(p[0] for p in parts)
            prob = float(math.prod([p[1] for p in parts])) if parts else 0.0
            return (best_c, prob)
        return ("", 0.0)

    def top_cherokee_candidates(
        self,
        arpabet: Union[str, ArpabetToken, Sequence[Union[str, ArpabetToken]]],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Returns top-k candidate Cherokee tokens and probabilities, sorted descending.
        """
        a = _normalize_arpabet_key(arpabet)
        targets = self.probabilities.get(a, {})
        sorted_items = sorted(targets.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_items[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "arpabet_vocab": list(self.arpabet_vocab),
            "cherokee_vocab": list(self.cherokee_vocab),
            "probabilities": {k: dict(v) for k, v in self.probabilities.items()},
            "log_costs": {k: dict(v) for k, v in self.log_costs.items()},
            "insertion_probabilities": dict(self.insertion_probabilities),
            "insertion_costs": dict(self.insertion_costs),
            "deletion_probabilities": dict(self.deletion_probabilities),
            "deletion_costs": dict(self.deletion_costs),
            "default_substitution_cost": self.default_substitution_cost,
            "default_insertion_cost": self.default_insertion_cost,
            "default_deletion_cost": self.default_deletion_cost,
            "prune_threshold": self.prune_threshold,
            "iteration": self.iteration,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AcousticConfusionMatrix:
        raw_probs = data.get("probabilities", {})
        probabilities: Dict[str, Dict[str, float]] = {
            k: {inner_k: float(inner_v) for inner_k, inner_v in v.items()}
            for k, v in raw_probs.items()
        }

        raw_costs = data.get("log_costs", {})
        log_costs: Dict[str, Dict[str, float]] = {
            k: {inner_k: float(inner_v) for inner_k, inner_v in v.items()}
            for k, v in raw_costs.items()
        }

        raw_ins_p = data.get("insertion_probabilities", {})
        insertion_probabilities = {k: float(v) for k, v in raw_ins_p.items()}

        raw_ins_c = data.get("insertion_costs", {})
        insertion_costs = {k: float(v) for k, v in raw_ins_c.items()}

        raw_del_p = data.get("deletion_probabilities", {})
        deletion_probabilities = {k: float(v) for k, v in raw_del_p.items()}

        raw_del_c = data.get("deletion_costs", {})
        deletion_costs = {k: float(v) for k, v in raw_del_c.items()}

        return cls(
            model_id=str(data["model_id"]),
            arpabet_vocab=tuple(str(x) for x in data.get("arpabet_vocab", [])),
            cherokee_vocab=tuple(str(x) for x in data.get("cherokee_vocab", [])),
            probabilities=probabilities,
            log_costs=log_costs,
            insertion_probabilities=insertion_probabilities,
            insertion_costs=insertion_costs,
            deletion_probabilities=deletion_probabilities,
            deletion_costs=deletion_costs,
            default_substitution_cost=float(
                data.get("default_substitution_cost", 10.0)
            ),
            default_insertion_cost=float(data.get("default_insertion_cost", 5.0)),
            default_deletion_cost=float(data.get("default_deletion_cost", 5.0)),
            prune_threshold=float(data.get("prune_threshold", 0.05)),
            iteration=int(data.get("iteration", 0)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> AcousticConfusionMatrix:
        return cls.from_dict(json.loads(s))

    def save(self, path: Union[Path, str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[Path, str]) -> AcousticConfusionMatrix:
        p = Path(path)
        return cls.from_json(p.read_text(encoding="utf-8"))


# ============================================================================
# 5. Synthetic Cherokee Target
# ============================================================================


@dataclass(frozen=True)
class SyntheticCherokeeTarget:
    """
    Projected synthetic Cherokee phonetic target derived from an English word
    via ARPAbet conversion and calibrated confusion matrix mapping.
    """

    source_word: str
    arpabet_tokens: Tuple[ArpabetToken, ...]
    cherokee_tokens: Tuple[CherokeeToken, ...]
    projected_tth: str
    syllabary: Optional[str] = None
    confidence_score: float = 1.0
    per_token_probabilities: Tuple[float, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "source_word": self.source_word,
            "arpabet_tokens": [t.to_dict() for t in self.arpabet_tokens],
            "cherokee_tokens": [t.to_dict() for t in self.cherokee_tokens],
            "projected_tth": self.projected_tth,
            "confidence_score": self.confidence_score,
            "per_token_probabilities": list(self.per_token_probabilities),
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
        return cls(
            source_word=str(data["source_word"]),
            arpabet_tokens=arp_tokens,
            cherokee_tokens=chr_tokens,
            projected_tth=str(data["projected_tth"]),
            syllabary=(
                str(data["syllabary"])
                if "syllabary" in data and data["syllabary"] is not None
                else None
            ),
            confidence_score=float(data.get("confidence_score", 1.0)),
            per_token_probabilities=tuple(
                float(p) for p in data.get("per_token_probabilities", [])
            ),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> SyntheticCherokeeTarget:
        return cls.from_dict(json.loads(s))


# ============================================================================
# 6. DP Alignment Traceback Models
# ============================================================================


@dataclass(frozen=True)
class AlignedTokenPair:
    """
    A single alignment step in DP traceback (substitution, insertion, deletion,
    or multi-gram joint transition).
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
            else:
                object.__setattr__(
                    self,
                    "cherokee",
                    CherokeeToken("".join(t.phone for t in self.cherokee_tokens)),
                )

    @property
    def is_substitution(self) -> bool:
        return len(self.arpabet_tokens) > 0 and len(self.cherokee_tokens) > 0

    @property
    def is_insertion(self) -> bool:
        """Epenthetic Cherokee token emitted with no corresponding ARPAbet token."""
        return len(self.arpabet_tokens) == 0 and len(self.cherokee_tokens) > 0

    @property
    def is_deletion(self) -> bool:
        """Dropped ARPAbet phoneme with no corresponding Cherokee emission."""
        return len(self.arpabet_tokens) > 0 and len(self.cherokee_tokens) == 0

    @property
    def arpabet_key(self) -> str:
        return " ".join(t.phone for t in self.arpabet_tokens)

    @property
    def cherokee_key(self) -> str:
        return "".join(t.phone for t in self.cherokee_tokens)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "arpabet": self.arpabet.to_dict() if self.arpabet is not None else None,
            "cherokee": self.cherokee.to_dict() if self.cherokee is not None else None,
            "cost": self.cost,
            "confidence": self.confidence,
        }
        if len(self.arpabet_tokens) > 1:
            d["arpabet_tokens"] = [t.to_dict() for t in self.arpabet_tokens]
        if len(self.cherokee_tokens) > 1:
            d["cherokee_tokens"] = [t.to_dict() for t in self.cherokee_tokens]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AlignedTokenPair:
        raw_arps = data.get("arpabet_tokens")
        if raw_arps:
            arp_tokens = tuple(
                t if isinstance(t, ArpabetToken) else ArpabetToken.from_dict(t)
                for t in raw_arps
            )
            arp = None
        else:
            raw_arp = data.get("arpabet")
            arp = ArpabetToken.from_dict(raw_arp) if raw_arp is not None else None
            arp_tokens = (arp,) if arp is not None else ()

        raw_chrs = data.get("cherokee_tokens")
        if raw_chrs:
            chr_tokens = tuple(
                t if isinstance(t, CherokeeToken) else CherokeeToken.from_dict(t)
                for t in raw_chrs
            )
            chr_tok = None
        else:
            raw_chr = data.get("cherokee")
            chr_tok = CherokeeToken.from_dict(raw_chr) if raw_chr is not None else None
            chr_tokens = (chr_tok,) if chr_tok is not None else ()

        return cls(
            arpabet=arp,
            cherokee=chr_tok,
            arpabet_tokens=arp_tokens,
            cherokee_tokens=chr_tokens,
            cost=float(data.get("cost", 0.0)),
            confidence=float(data.get("confidence", 1.0)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> AlignedTokenPair:
        return cls.from_dict(json.loads(s))


@dataclass(frozen=True)
class TracebackAlignmentResult:
    """
    Result of aligning an ARPAbet token sequence with an emitted Cherokee token sequence.
    """

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

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> TracebackAlignmentResult:
        return cls.from_dict(json.loads(s))


# ============================================================================
# 7. Pure Mapping Protocols
# ============================================================================


@runtime_checkable
class EnglishToArpabetProtocol(Protocol):
    """
    Pure functional protocol mapping English text to standardized ARPAbet tokens.
    """

    def __call__(
        self, text: str, strip_stress: bool = True
    ) -> Tuple[ArpabetToken, ...]: ...

    def extract(
        self, text: str, strip_stress: bool = True
    ) -> Tuple[ArpabetToken, ...]: ...


# Type alias for G2P extraction
G2PExtractorProtocol = EnglishToArpabetProtocol


@runtime_checkable
class TracebackAlignerProtocol(Protocol):
    """
    Pure functional protocol executing dynamic programming traceback between
    ARPAbet phoneme sequences and emitted Cherokee tokens using an acoustic confusion matrix.
    """

    def align(
        self,
        arpabet_tokens: Sequence[ArpabetToken],
        cherokee_tokens: Sequence[CherokeeToken],
        matrix: AcousticConfusionMatrix,
    ) -> TracebackAlignmentResult: ...


@runtime_checkable
class DPTracebackAccumulatorProtocol(Protocol):
    """
    Protocol for accumulating alignment statistics across aligned pairs.
    """

    def accumulate(
        self,
        alignment: TracebackAlignmentResult,
    ) -> None: ...


@runtime_checkable
class SyntheticTargetProjectorProtocol(Protocol):
    """
    Pure functional protocol projecting English words or ARPAbet sequences
    into synthetic Cherokee phonetic targets using an acoustic confusion matrix.
    """

    def project_word(
        self,
        word: str,
        matrix: Optional[AcousticConfusionMatrix] = None,
    ) -> SyntheticCherokeeTarget: ...

    def project_arpabet(
        self,
        arpabet_tokens: Sequence[ArpabetToken],
        matrix: Optional[AcousticConfusionMatrix] = None,
        source_word: str = "",
    ) -> SyntheticCherokeeTarget: ...

    def project_english_text(
        self,
        text: str,
        matrix: Optional[AcousticConfusionMatrix] = None,
    ) -> str: ...
