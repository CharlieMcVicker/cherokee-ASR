# -*- coding: utf-8 -*-
"""
digohwelisgi.core.codeswitching.projector

Generic, language-agnostic SyntheticTargetProjector translating source text (e.g. English)
via G2P -> ARPAbet -> target language phonetics using calibrated AcousticConfusionMatrix
and static memoized dictionaries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

import numpy as np

from digohwelisgi.core.codeswitching.g2p import G2PEngine, get_default_g2p
from digohwelisgi.core.codeswitching.matrix import AcousticConfusionMatrix
from digohwelisgi.core.codeswitching.types import (
    EPSILON_TOKEN,
    ARPAbetPhone,
    EnglishToArpabetProtocol,
    GenericConfusionMatrixProtocol,
    GenericSyntheticTargetProjectorProtocol,
    ProjectedTarget,
)

logger = logging.getLogger(__name__)

_WORD_BOUNDARY_PUNCTUATION_RE = re.compile(r"^[^a-zA-Z0-9']+|[^a-zA-Z0-9']+$")


class SyntheticTargetProjector:
    """
    Generic runtime projector translating words and ARPAbet sequences into
    target language phonetic targets.

    Implements GenericSyntheticTargetProjectorProtocol.
    Uses O(1) static dictionary lookup when available, falling back to
    G2P extraction + Viterbi / argmax AcousticConfusionMatrix mapping.
    """

    def __init__(
        self,
        matrix: Optional[AcousticConfusionMatrix] = None,
        dictionary: Optional[Union[str, Path, Dict[str, Any]]] = None,
        g2p: Optional[EnglishToArpabetProtocol] = None,
    ) -> None:
        self._matrix = matrix
        self._g2p: EnglishToArpabetProtocol = (
            g2p if g2p is not None else get_default_g2p()
        )
        self._static_dict: Dict[str, Any] = {}
        self._runtime_cache: Dict[str, ProjectedTarget] = {}

        if dictionary is not None:
            self._load_dictionary(dictionary)

    @property
    def matrix(self) -> Optional[AcousticConfusionMatrix]:
        return self._matrix

    @property
    def dictionary(self) -> Dict[str, Any]:
        return self._static_dict

    def _load_dictionary(self, source: Union[str, Path, Dict[str, Any]]) -> None:
        """Loads static dictionary from file or in-memory dictionary."""
        if isinstance(source, dict):
            self._static_dict = dict(source)
            return

        path = Path(source)
        if not path.exists():
            logger.warning(f"Static dictionary not found at {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._static_dict = data
        except Exception as e:
            logger.warning(f"Failed to load static dictionary from {path}: {e}")

    def project_word(
        self,
        word: str,
        matrix: Optional[GenericConfusionMatrixProtocol] = None,
    ) -> ProjectedTarget:
        """
        Projects a single word to a ProjectedTarget.

        Lookup precedence:
        1. Static precomputed dictionary (O(1)).
        2. Dynamic runtime cache (O(1)).
        3. Dynamic G2P extraction + Viterbi confusion matrix substitution.
        """
        raw_word = word.strip()
        cleaned = _WORD_BOUNDARY_PUNCTUATION_RE.sub("", raw_word).lower()
        if not cleaned:
            return ProjectedTarget(
                source_word=raw_word,
                arpabet_tokens=(),
                target_tokens=(),
                projected_text="",
            )

        # 1. Static memoized dictionary lookup (O(1))
        if cleaned in self._static_dict:
            entry = self._static_dict[cleaned]
            if isinstance(entry, ProjectedTarget):
                return entry
            elif isinstance(entry, dict):
                target = ProjectedTarget.from_dict(entry)
                self._runtime_cache[cleaned] = target
                return target
            elif isinstance(entry, str):
                target = ProjectedTarget(
                    source_word=raw_word,
                    arpabet_tokens=(),
                    target_tokens=(entry,),
                    projected_text=entry,
                )
                self._runtime_cache[cleaned] = target
                return target

        # 2. Dynamic runtime cache lookup
        if cleaned in self._runtime_cache:
            return self._runtime_cache[cleaned]

        # 3. Dynamic G2P extraction + Confusion Matrix mapping
        active_matrix = matrix if matrix is not None else self._matrix
        if active_matrix is None:
            raise ValueError(
                "Cannot project word: No AcousticConfusionMatrix provided or configured."
            )

        arpabet_tokens = self._g2p.extract(cleaned, strip_stress=True)
        if not arpabet_tokens:
            target = ProjectedTarget(
                source_word=raw_word,
                arpabet_tokens=(),
                target_tokens=(),
                projected_text="",
            )
            self._runtime_cache[cleaned] = target
            return target

        target = self.project_arpabet(
            arpabet_tokens=arpabet_tokens,
            matrix=active_matrix,
            source_word=raw_word,
        )
        self._runtime_cache[cleaned] = target
        return target

    def project_arpabet(
        self,
        arpabet_tokens: Sequence[ARPAbetPhone],
        matrix: Optional[GenericConfusionMatrixProtocol] = None,
        source_word: str = "",
    ) -> ProjectedTarget:
        """
        Pure functional mapping from ARPAbet token sequence to ProjectedTarget
        using dynamic programming (Viterbi tiling) across 1-gram and 2-gram transitions
        from the AcousticConfusionMatrix.
        """
        active_matrix = matrix if matrix is not None else self._matrix
        if active_matrix is None:
            raise ValueError(
                "Cannot project ARPAbet: No AcousticConfusionMatrix provided or configured."
            )

        K = len(arpabet_tokens)
        if K == 0:
            return ProjectedTarget(
                source_word=source_word,
                arpabet_tokens=(),
                target_tokens=(),
                projected_text="",
                confidence_score=1.0,
                per_token_probabilities=(),
            )

        # Dynamic programming arrays for Viterbi tiling
        dp_cost = [0.0] + [1e9] * K
        dp_step = [0] * (K + 1)
        dp_target = [""] * (K + 1)
        dp_prob = [1.0] * (K + 1)

        for k in range(1, K + 1):
            # 1. Option 1: 1-gram step on arpabet_tokens[k - 1]
            t1 = arpabet_tokens[k - 1].phone
            best_c1, p1 = active_matrix.best_target_for(t1)
            if not best_c1 or best_c1 in (
                "",
                "<eps>",
                "<EPS>",
                "EPS",
                "eps",
                EPSILON_TOKEN,
            ):
                cost1 = active_matrix.get_deletion_cost(t1)
                target1 = ""
                prob1 = (
                    active_matrix.get_deletion_probability(t1)
                    if hasattr(active_matrix, "get_deletion_probability")
                    else 0.0
                )
            else:
                cost1 = active_matrix.get_substitution_cost(t1, best_c1)
                target1 = best_c1
                prob1 = p1

            best_cost = dp_cost[k - 1] + cost1
            best_step = 1
            best_t = target1
            best_p = prob1

            # 2. Option 2: 2-gram step on arpabet_tokens[k - 2 : k]
            if k >= 2:
                pair_key = (
                    f"{arpabet_tokens[k - 2].phone} {arpabet_tokens[k - 1].phone}"
                )
                if active_matrix.has_transition(pair_key):
                    best_c2, p2 = active_matrix.best_target_for(pair_key)
                    if best_c2 and best_c2 not in (
                        "",
                        "<eps>",
                        "<EPS>",
                        "EPS",
                        "eps",
                        EPSILON_TOKEN,
                    ):
                        cost2 = active_matrix.get_substitution_cost(pair_key, best_c2)
                        if dp_cost[k - 2] + cost2 < best_cost:
                            best_cost = dp_cost[k - 2] + cost2
                            best_step = 2
                            best_t = best_c2
                            best_p = p2

            dp_cost[k] = best_cost
            dp_step[k] = best_step
            dp_target[k] = best_t
            dp_prob[k] = best_p

        # Traceback from K down to 0
        target_tokens: List[str] = []
        token_probs: List[float] = []
        k = K
        while k > 0:
            step = dp_step[k]
            t_phone = dp_target[k]
            p_val = dp_prob[k]
            if t_phone:
                target_tokens.append(t_phone)
                token_probs.append(p_val)
            k -= step

        target_tokens.reverse()
        token_probs.reverse()

        projected_text = "".join(target_tokens)
        confidence = float(np.mean(token_probs)) if token_probs else 1.0

        return ProjectedTarget(
            source_word=source_word,
            arpabet_tokens=tuple(arpabet_tokens),
            target_tokens=tuple(target_tokens),
            projected_text=projected_text,
            confidence_score=confidence,
            per_token_probabilities=tuple(token_probs),
        )

    def project_english_text(
        self,
        text: str,
        matrix: Optional[GenericConfusionMatrixProtocol] = None,
    ) -> str:
        """
        Converts space-separated English text to space-separated projected target text.
        """
        tokens = text.strip().split()
        if not tokens:
            return ""

        projected_words: List[str] = []
        for tok in tokens:
            t = self.project_word(tok, matrix=matrix)
            if t.projected_text:
                projected_words.append(t.projected_text)

        return " ".join(projected_words)


def generate_static_dictionary(
    words: Sequence[str],
    matrix: AcousticConfusionMatrix,
    output_path: Union[str, Path],
    g2p: Optional[EnglishToArpabetProtocol] = None,
    full_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Precomputes static memoized dictionary from a list of English words and serializes to JSON.
    """
    projector = SyntheticTargetProjector(matrix=matrix, g2p=g2p)
    out_dict: Dict[str, Any] = {}

    for raw_word in words:
        cleaned = _WORD_BOUNDARY_PUNCTUATION_RE.sub("", raw_word.strip()).lower()
        if not cleaned or cleaned in out_dict:
            continue
        target = projector.project_word(cleaned, matrix=matrix)
        if full_metadata:
            out_dict[cleaned] = target.to_dict()
        else:
            out_dict[cleaned] = target.projected_text

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    logger.info(
        f"Generated static dictionary with {len(out_dict)} entries at {out_file}"
    )
    return out_dict


__all__ = [
    "SyntheticTargetProjector",
    "generate_static_dictionary",
]
