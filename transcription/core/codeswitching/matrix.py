# -*- coding: utf-8 -*-
"""
transcription.core.codeswitching.matrix

Language-agnostic AcousticConfusionMatrix: empirical statistical mapping between
source (e.g. ARPAbet) tokens/ngrams and target language phonetic tokens/ngrams.
Supports generic serialization (JSON, NPZ), conditional probability calculations,
marginalization, and log-cost evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import numpy as np

from transcription.core.codeswitching.types import (
    EPSILON_TOKEN,
    ARPAbetPhone,
    GenericConfusionMatrixProtocol,
)


def normalize_source_key(
    source: Union[str, ARPAbetPhone, Sequence[Union[str, ARPAbetPhone]]],
) -> str:
    """Normalizes a source phone or sequence of phones into a lookup key (space-joined)."""
    if isinstance(source, str):
        return source.strip()
    elif isinstance(source, ARPAbetPhone):
        return source.phone
    elif isinstance(source, (list, tuple)):
        return " ".join(
            t.phone if isinstance(t, ARPAbetPhone) else str(t).strip()
            for t in source
            if (t.phone if isinstance(t, ARPAbetPhone) else str(t).strip())
            not in ("", "<eps>", "<EPS>", "eps", "EPS", EPSILON_TOKEN)
        )
    return str(source)


def normalize_target_key(
    target: Union[str, Sequence[str]],
) -> str:
    """Normalizes a target phone or sequence of phones into a target key (concatenated)."""
    if isinstance(target, str):
        return target.strip()
    elif isinstance(target, (list, tuple)):
        return "".join(
            str(t).strip()
            for t in target
            if str(t).strip() not in ("", "<eps>", "<EPS>", "eps", "EPS", EPSILON_TOKEN)
        )
    return str(target)


@dataclass(frozen=True)
class AcousticConfusionMatrix:
    """
    Empirical statistical mapping between source tokens (e.g. ARPAbet) and target phonemes.
    Language-agnostic: target phonemes and vocabulary are dynamically configured.
    Supports both 1-gram and multi-gram (e.g. 1-to-2, 2-to-1, 2-to-2) transitions.

    Stores:
    - Conditional substitution probabilities: P(target | source)
    - Negative log substitution costs: -log(P(target | source))
    - Insertion (epenthetic) probabilities and costs: P(target | eps)
    - Deletion (coda loss) probabilities and costs: P(eps | source)
    """

    model_id: str
    source_vocab: Tuple[str, ...]
    target_vocab: Tuple[str, ...]
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

    # Aliases for Cherokee backward compatibility
    @property
    def arpabet_vocab(self) -> Tuple[str, ...]:
        return self.source_vocab

    @property
    def cherokee_vocab(self) -> Tuple[str, ...]:
        return self.target_vocab

    @classmethod
    def create(
        cls,
        model_id: str,
        probabilities: Dict[str, Dict[str, float]],
        insertion_probabilities: Optional[Dict[str, float]] = None,
        deletion_probabilities: Optional[Dict[str, float]] = None,
        source_vocab: Optional[Sequence[str]] = None,
        target_vocab: Optional[Sequence[str]] = None,
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
        all_targets: Set[str] = set(ins_probs.keys())

        for src_phone, tgt_map in probabilities.items():
            log_costs[src_phone] = {}
            for tgt_phone, prob in tgt_map.items():
                all_targets.add(tgt_phone)
                if prob > 0.0:
                    log_costs[src_phone][tgt_phone] = max(
                        0.0, -math.log(max(prob, 1e-12))
                    )
                else:
                    log_costs[src_phone][tgt_phone] = default_substitution_cost

        # Compute insertion log-costs
        ins_costs: Dict[str, float] = {}
        for tgt_phone, prob in ins_probs.items():
            if prob > 0.0:
                ins_costs[tgt_phone] = max(0.0, -math.log(max(prob, 1e-12)))
            else:
                ins_costs[tgt_phone] = default_insertion_cost

        # Compute deletion log-costs
        del_costs: Dict[str, float] = {}
        for src_phone, prob in del_probs.items():
            if prob > 0.0:
                del_costs[src_phone] = max(0.0, -math.log(max(prob, 1e-12)))
            else:
                del_costs[src_phone] = default_deletion_cost

        resolved_src = source_vocab if source_vocab is not None else arpabet_vocab
        resolved_tgt = target_vocab if target_vocab is not None else cherokee_vocab

        src_v = (
            tuple(resolved_src)
            if resolved_src is not None
            else tuple(sorted(probabilities.keys()))
        )
        tgt_v = (
            tuple(resolved_tgt)
            if resolved_tgt is not None
            else tuple(sorted(all_targets))
        )

        return cls(
            model_id=model_id,
            source_vocab=src_v,
            target_vocab=tgt_v,
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
        source: Union[str, ARPAbetPhone, Sequence[Union[str, ARPAbetPhone]]],
        target: Union[str, Sequence[str]],
    ) -> float:
        """Returns substitution cost -log P(target | source)."""
        a = normalize_source_key(source)
        c = normalize_target_key(target)
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
        source: Union[str, ARPAbetPhone, Sequence[Union[str, ARPAbetPhone]]],
        target: Union[str, Sequence[str]],
    ) -> float:
        """Returns conditional probability P(target | source)."""
        a = normalize_source_key(source)
        c = normalize_target_key(target)
        if a in self.probabilities and c in self.probabilities[a]:
            return self.probabilities[a][c]
        return 0.0

    def has_transition(
        self,
        source: Union[str, ARPAbetPhone, Sequence[Union[str, ARPAbetPhone]]],
    ) -> bool:
        """Returns True if the matrix contains transitions for the given source phone/ngram key."""
        a = normalize_source_key(source)
        return a in self.probabilities and len(self.probabilities[a]) > 0

    def get_insertion_cost(
        self,
        target: Union[str, Sequence[str]],
    ) -> float:
        """Returns insertion cost -log P(target | eps)."""
        c = normalize_target_key(target)
        return self.insertion_costs.get(c, self.default_insertion_cost)

    def get_insertion_probability(
        self,
        target: Union[str, Sequence[str]],
    ) -> float:
        """Returns insertion probability P(target | eps)."""
        c = normalize_target_key(target)
        return self.insertion_probabilities.get(c, 0.0)

    def get_deletion_cost(
        self,
        source: Union[str, ARPAbetPhone, Sequence[Union[str, ARPAbetPhone]]],
    ) -> float:
        """Returns deletion cost -log P(eps | source)."""
        a = normalize_source_key(source)
        return self.deletion_costs.get(a, self.default_deletion_cost)

    def get_deletion_probability(
        self,
        source: Union[str, ARPAbetPhone, Sequence[Union[str, ARPAbetPhone]]],
    ) -> float:
        """Returns deletion probability P(eps | source)."""
        a = normalize_source_key(source)
        return self.deletion_probabilities.get(a, 0.0)

    def best_target_for(
        self,
        source: Union[str, ARPAbetPhone, Sequence[Union[str, ARPAbetPhone]]],
    ) -> Tuple[str, float]:
        """
        Returns the argmax target token sequence and its probability for the given source phoneme(s).
        """
        a = normalize_source_key(source)
        targets = self.probabilities.get(a, {})
        if targets:
            best_c, best_p = max(targets.items(), key=lambda kv: kv[1])
            return (best_c, best_p)
        tokens = a.split()
        if len(tokens) > 1:
            parts = [self.best_target_for(t) for t in tokens]
            best_c = "".join(p[0] for p in parts)
            prob = float(math.prod([p[1] for p in parts])) if parts else 0.0
            return (best_c, prob)
        return ("", 0.0)

    # Alias for Cherokee backward compatibility
    def best_cherokee_for(
        self,
        arpabet: Union[str, ARPAbetPhone, Sequence[Union[str, ARPAbetPhone]]],
    ) -> Tuple[str, float]:
        return self.best_target_for(arpabet)

    def top_target_candidates(
        self,
        source: Union[str, ARPAbetPhone, Sequence[Union[str, ARPAbetPhone]]],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Returns top-k candidate target tokens and probabilities, sorted descending.
        """
        a = normalize_source_key(source)
        targets = self.probabilities.get(a, {})
        sorted_items = sorted(targets.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_items[:top_k]

    def top_cherokee_candidates(
        self,
        arpabet: Union[str, ARPAbetPhone, Sequence[Union[str, ARPAbetPhone]]],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        return self.top_target_candidates(arpabet, top_k=top_k)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_vocab": list(self.source_vocab),
            "target_vocab": list(self.target_vocab),
            "arpabet_vocab": list(self.source_vocab),
            "cherokee_vocab": list(self.target_vocab),
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

        src_vocab = data.get("source_vocab") or data.get("arpabet_vocab") or []
        tgt_vocab = data.get("target_vocab") or data.get("cherokee_vocab") or []

        return cls(
            model_id=str(data["model_id"]),
            source_vocab=tuple(str(x) for x in src_vocab),
            target_vocab=tuple(str(x) for x in tgt_vocab),
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
        if p.suffix == ".npz":
            self.save_npz(p)
        else:
            p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[Path, str]) -> AcousticConfusionMatrix:
        p = Path(path)
        if p.suffix == ".npz":
            return cls.load_npz(p)
        return cls.from_json(p.read_text(encoding="utf-8"))

    def save_npz(self, path: Union[Path, str]) -> None:
        """Saves matrix in compressed NumPy format."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Serialize metadata and probabilities to JSON string for inclusion in npz
        json_str = self.to_json()
        np.savez_compressed(str(p), json_data=np.array([json_str], dtype=object))

    @classmethod
    def load_npz(cls, path: Union[Path, str]) -> AcousticConfusionMatrix:
        """Loads matrix from compressed NumPy format."""
        p = Path(path)
        data = np.load(str(p), allow_pickle=True)
        json_str = str(data["json_data"][0])
        return cls.from_json(json_str)


__all__ = [
    "AcousticConfusionMatrix",
    "normalize_source_key",
    "normalize_target_key",
]
