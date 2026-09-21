# -*- coding: utf-8 -*-
"""
transcription.alignment.arpabet.projector module.

Pure functional projector mapping English text -> ARPAbet -> synthetic Cherokee TTH target
using calibrated acoustic confusion matrices (AcousticConfusionMatrix) and O(1) static dictionaries.

Follows Types and Maps architectural principles:
- Strict protocol compliance: implements SyntheticTargetProjectorProtocol.
- Precomputed static memoized dictionary for O(1) runtime lookup.
- Fallback dynamic G2P phoneme extraction via g2p_en.
- Clean argmax mapping conforming to canonical Cherokee T/TH consonant inventory.
- Pure representation-aware code-switched text normalization.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from transcription.alignment.arpabet.g2p import G2pExtractor, get_default_g2p
from transcription.alignment.arpabet.types import (
    EPSILON_TOKEN,
    AcousticConfusionMatrix,
    ArpabetToken,
    CherokeeToken,
    Orthography,
    SyntheticCherokeeTarget,
    SyntheticTargetProjectorProtocol,
)
from transcription.alignment.normalizers import normalize_syllabary_for_alignment
from transcription.utils.orthography import convert_orthography

logger = logging.getLogger(__name__)

# Default repository paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT
    / "data"
    / "arpabet_alignment"
    / "matrices"
    / "charliemcvicker_length-only-20260704-155307-asr-cherokee-colon_76e62140955f4738abdab345ea34068b02d8d2a2_confusion_matrix.json"
)
DEFAULT_STATIC_DICTIONARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "arpabet_alignment"
    / "dictionaries"
    / "english_loanwords_tth.json"
)

# Regex to strip non-alphanumeric punctuation from word boundaries
_WORD_BOUNDARY_PUNCTUATION_RE = re.compile(r"^[^a-zA-Z0-9']+|[^a-zA-Z0-9']+$")


def is_english_word(token: str) -> bool:
    """
    Checks whether a token represents an English code-switched word rather than
    native Cherokee syllabary.

    Returns True if token contains Latin characters and zero Cherokee Unicode
    syllabary characters.
    """
    if not token:
        return False
    has_syllabary = any(
        0x13A0 <= ord(c) <= 0x13FF or 0xAB70 <= ord(c) <= 0xABBF for c in token
    )
    if has_syllabary:
        return False
    return any(c.isalpha() for c in token)


class SyntheticTargetProjector:
    """
    Runtime projector translating English words and ARPAbet sequences into
    synthetic Cherokee TTH phonetic targets.

    Implements SyntheticTargetProjectorProtocol.
    Uses O(1) static dictionary lookup when available, falling back to
    G2P extraction + argmax AcousticConfusionMatrix mapping.
    """

    def __init__(
        self,
        matrix: Optional[AcousticConfusionMatrix] = None,
        dictionary: Optional[Union[str, Path, Dict[str, Any]]] = None,
        g2p: Optional[G2pExtractor] = None,
    ) -> None:
        self._matrix = matrix
        self._g2p = g2p if g2p is not None else get_default_g2p()
        self._static_dict: Dict[str, Any] = {}
        self._runtime_cache: Dict[str, SyntheticCherokeeTarget] = {}

        if dictionary is not None:
            self._load_dictionary(dictionary)
        elif DEFAULT_STATIC_DICTIONARY_PATH.exists():
            self._load_dictionary(DEFAULT_STATIC_DICTIONARY_PATH)

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
            logger.warning(f"Static loanword dictionary not found at {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._static_dict = data
                logger.debug(
                    f"Loaded {len(self._static_dict)} static word mappings from {path}"
                )
        except Exception as e:
            logger.warning(
                f"Failed to load static loanword dictionary from {path}: {e}"
            )

    def project_word(
        self,
        word: str,
        matrix: Optional[AcousticConfusionMatrix] = None,
    ) -> SyntheticCherokeeTarget:
        """
        Projects a single English word to a SyntheticCherokeeTarget.

        Lookup precedence:
        1. Static precomputed dictionary (O(1)).
        2. Dynamic runtime cache (O(1)).
        3. Dynamic G2P extraction + argmax confusion matrix substitution.
        """
        raw_word = word.strip()
        cleaned = _WORD_BOUNDARY_PUNCTUATION_RE.sub("", raw_word).lower()
        if not cleaned:
            return SyntheticCherokeeTarget(
                source_word=raw_word,
                arpabet_tokens=(),
                cherokee_tokens=(),
                projected_tth="",
            )

        # 1. Static memoized dictionary lookup (O(1))
        if cleaned in self._static_dict:
            entry = self._static_dict[cleaned]
            if isinstance(entry, SyntheticCherokeeTarget):
                return entry
            elif isinstance(entry, dict):
                target = SyntheticCherokeeTarget.from_dict(entry)
                self._runtime_cache[cleaned] = target
                return target
            elif isinstance(entry, str):
                target = SyntheticCherokeeTarget(
                    source_word=raw_word,
                    arpabet_tokens=(),
                    cherokee_tokens=(),
                    projected_tth=entry,
                )
                self._runtime_cache[cleaned] = target
                return target

        # 2. Dynamic runtime cache lookup
        if cleaned in self._runtime_cache:
            return self._runtime_cache[cleaned]

        # 3. Dynamic G2P extraction + Confusion Matrix mapping
        active_matrix = matrix if matrix is not None else self._matrix
        if active_matrix is None:
            active_matrix = load_default_confusion_matrix()

        arpabet_tokens = self._g2p.extract(cleaned, strip_stress=True)
        if not arpabet_tokens:
            target = SyntheticCherokeeTarget(
                source_word=raw_word,
                arpabet_tokens=(),
                cherokee_tokens=(),
                projected_tth="",
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
        arpabet_tokens: Sequence[ArpabetToken],
        matrix: Optional[AcousticConfusionMatrix] = None,
        source_word: str = "",
    ) -> SyntheticCherokeeTarget:
        """
        Pure functional mapping from ARPAbet token sequence to SyntheticCherokeeTarget
        using dynamic programming (Viterbi tiling) across 1-gram and 2-gram transitions
        from the AcousticConfusionMatrix.
        """
        active_matrix = matrix if matrix is not None else self._matrix
        if active_matrix is None:
            active_matrix = load_default_confusion_matrix()

        K = len(arpabet_tokens)
        if K == 0:
            return SyntheticCherokeeTarget(
                source_word=source_word,
                arpabet_tokens=(),
                cherokee_tokens=(),
                projected_tth="",
                syllabary=None,
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
            best_c1, p1 = active_matrix.best_cherokee_for(t1)
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
                prob1 = active_matrix.get_deletion_probability(t1)
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
                    best_c2, p2 = active_matrix.best_cherokee_for(pair_key)
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
        cherokee_tokens: List[CherokeeToken] = []
        token_probs: List[float] = []
        k = K
        while k > 0:
            step = dp_step[k]
            t_phone = dp_target[k]
            p_val = dp_prob[k]
            if t_phone:
                cherokee_tokens.append(
                    CherokeeToken(phone=t_phone, orthography=Orthography.TTH)
                )
                token_probs.append(p_val)
            k -= step

        cherokee_tokens.reverse()
        token_probs.reverse()

        projected_tth = "".join(t.phone for t in cherokee_tokens)
        confidence = float(np.mean(token_probs)) if token_probs else 1.0

        # Attempt optional native syllabary transliteration
        syllabary: Optional[str] = None
        if projected_tth:
            try:
                syll_cand = convert_orthography(
                    projected_tth,
                    source=Orthography.TTH,
                    target=Orthography.SYLLABARY,
                )
                if any(
                    0x13A0 <= ord(c) <= 0x13FF or 0xAB70 <= ord(c) <= 0xABBF
                    for c in syll_cand
                ):
                    syllabary = syll_cand
            except Exception:
                syllabary = None

        return SyntheticCherokeeTarget(
            source_word=source_word,
            arpabet_tokens=tuple(arpabet_tokens),
            cherokee_tokens=tuple(cherokee_tokens),
            projected_tth=projected_tth,
            syllabary=syllabary,
            confidence_score=confidence,
            per_token_probabilities=tuple(token_probs),
        )

    def project_english_text(
        self,
        text: str,
        matrix: Optional[AcousticConfusionMatrix] = None,
    ) -> str:
        """
        Converts space-separated English text to space-separated synthetic Cherokee TTH.
        """
        tokens = text.strip().split()
        if not tokens:
            return ""

        projected_words: List[str] = []
        for tok in tokens:
            t = self.project_word(tok, matrix=matrix)
            if t.projected_tth:
                projected_words.append(t.projected_tth)

        return " ".join(projected_words)


# Global singleton instances
_DEFAULT_CONFUSION_MATRIX: Optional[AcousticConfusionMatrix] = None
_DEFAULT_PROJECTOR: Optional[SyntheticTargetProjector] = None


def load_default_confusion_matrix(
    path: Optional[Union[str, Path]] = None,
) -> AcousticConfusionMatrix:
    """
    Loads and caches the default production pre-Bible confusion matrix.
    """
    global _DEFAULT_CONFUSION_MATRIX
    if _DEFAULT_CONFUSION_MATRIX is not None and path is None:
        return _DEFAULT_CONFUSION_MATRIX

    matrix_path = Path(path) if path is not None else DEFAULT_CONFUSION_MATRIX_PATH
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Production confusion matrix not found at '{matrix_path}'."
        )

    with open(matrix_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matrix = AcousticConfusionMatrix.from_dict(data)
    if path is None:
        _DEFAULT_CONFUSION_MATRIX = matrix
    return matrix


def get_default_projector(
    matrix_path: Optional[Union[str, Path]] = None,
    dictionary_path: Optional[Union[str, Path]] = None,
) -> SyntheticTargetProjector:
    """
    Returns or lazily initializes the default global SyntheticTargetProjector instance.
    """
    global _DEFAULT_PROJECTOR
    if (
        _DEFAULT_PROJECTOR is not None
        and matrix_path is None
        and dictionary_path is None
    ):
        return _DEFAULT_PROJECTOR

    matrix = load_default_confusion_matrix(matrix_path)
    dict_path = (
        dictionary_path
        if dictionary_path is not None
        else DEFAULT_STATIC_DICTIONARY_PATH
    )
    dict_arg = dict_path if Path(dict_path).exists() else None

    projector = SyntheticTargetProjector(
        matrix=matrix,
        dictionary=dict_arg,
    )
    if matrix_path is None and dictionary_path is None:
        _DEFAULT_PROJECTOR = projector
    return projector


def project_english_word(
    word: str,
    matrix: Optional[AcousticConfusionMatrix] = None,
) -> SyntheticCherokeeTarget:
    """Convenience function projecting a single English word using the default projector."""
    return get_default_projector().project_word(word, matrix=matrix)


def project_english_text(
    text: str,
    matrix: Optional[AcousticConfusionMatrix] = None,
) -> str:
    """Convenience function projecting English text to Cherokee TTH using the default projector."""
    return get_default_projector().project_english_text(text, matrix=matrix)


def generate_static_dictionary(
    words: Sequence[str],
    matrix: AcousticConfusionMatrix,
    output_path: Union[str, Path],
    g2p: Optional[G2pExtractor] = None,
    full_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Precomputes static memoized dictionary from a list of English words and serializes to JSON.

    Args:
        words: Collection of English words to project and cache.
        matrix: Calibrated AcousticConfusionMatrix.
        output_path: Destination JSON file path.
        g2p: Optional G2pExtractor instance.
        full_metadata: If True, serializes full SyntheticCherokeeTarget dicts.
                       If False, serializes compact word -> projected_tth strings.

    Returns:
        The generated dictionary mapping words to their projections.
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
            out_dict[cleaned] = target.projected_tth

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    logger.info(
        f"Generated static dictionary with {len(out_dict)} entries at {out_file}"
    )
    return out_dict


def normalize_code_switched_text(
    text: str,
    normalizer: Callable[[str], str] = normalize_syllabary_for_alignment,
    projector: Optional[SyntheticTargetProjectorProtocol] = None,
) -> str:
    """
    Normalizes mixed Cherokee Syllabary and English code-switched text into
    canonical Cherokee TTH phonetics for ASR alignment matching.

    Cherokee Syllabary words are converted to TTH via normalizer.
    English words are projected into synthetic Cherokee TTH targets via projector.
    Surrounding punctuation is stripped or normalized.

    Args:
        text: Input mixed Cherokee/English string (e.g. 'ᎯᎠ coffee ᎠᎩᏚᎵ').
        normalizer: Function to normalize Cherokee Syllabary into TTH.
        projector: Projector instance. If None, uses get_default_projector().

    Returns:
        Space-separated string of canonical Cherokee TTH phonetics.
    """
    if not text:
        return ""

    active_projector: SyntheticTargetProjectorProtocol = (
        projector if projector is not None else get_default_projector()
    )

    tokens = text.split()
    output_tokens: List[str] = []

    for tok in tokens:
        has_syllabary = any(
            0x13A0 <= ord(c) <= 0x13FF or 0xAB70 <= ord(c) <= 0xABBF for c in tok
        )
        if has_syllabary:
            norm = normalize_syllabary_for_alignment(tok)
            if norm:
                output_tokens.append(norm)
        elif is_english_word(tok):
            # Project English code-switched word
            target = active_projector.project_word(tok)
            if target.projected_tth:
                output_tokens.append(target.projected_tth)
            else:
                norm = normalizer(tok)
                if norm:
                    output_tokens.append(norm)
        else:
            # Latin Cherokee or other token
            norm = normalizer(tok)
            if norm:
                output_tokens.append(norm)

    return " ".join(output_tokens)


__all__ = [
    "DEFAULT_CONFUSION_MATRIX_PATH",
    "DEFAULT_STATIC_DICTIONARY_PATH",
    "SyntheticTargetProjector",
    "generate_static_dictionary",
    "get_default_projector",
    "is_english_word",
    "load_default_confusion_matrix",
    "normalize_code_switched_text",
    "project_english_text",
    "project_english_word",
]
