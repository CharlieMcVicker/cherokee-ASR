# -*- coding: utf-8 -*-
"""
transcription.cherokee.codeswitching.projector module.

Tier 2 Cherokee factory for SyntheticTargetProjector and Cherokee-specific
static dictionary / acoustic matrix configuration.
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

from transcription.core.codeswitching.g2p import get_default_g2p
from transcription.core.codeswitching.matrix import (
    AcousticConfusionMatrix,
    normalize_source_key,
    normalize_target_key,
)
from transcription.core.codeswitching.projector import (
    SyntheticTargetProjector,
    generate_static_dictionary,
)
from transcription.core.codeswitching.types import (
    EPSILON_TOKEN,
    ARPAbetPhone,
    ArpabetToken,
    EnglishToArpabetProtocol,
    GenericConfusionMatrixProtocol,
    GenericSyntheticTargetProjectorProtocol,
    ProjectedTarget,
)
from transcription.cherokee.codeswitching.types import (
    CANONICAL_CHEROKEE_CONSONANTS,
    CANONICAL_CHEROKEE_TTH_PHONEMES,
    CANONICAL_CHEROKEE_VOWELS,
    CherokeeToken,
    SyntheticCherokeeTarget,
    SyntheticTargetProjectorProtocol,
)
from transcription.cherokee.orthography import Orthography, convert_orthography

logger = logging.getLogger(__name__)

# Default repository paths resolved relative to repository root
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

CHEROKEE_TTH_TARGET_PHONEMES = CANONICAL_CHEROKEE_TTH_PHONEMES

# Regex to strip non-alphanumeric punctuation from word boundaries
_WORD_BOUNDARY_PUNCTUATION_RE = re.compile(r"^[^a-zA-Z0-9']+|[^a-zA-Z0-9']+$")


def get_english_loanwords_tth_dict(
    dict_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Loads and returns the precomputed English loanwords static dictionary.
    Resolves relative to data/arpabet_alignment/dictionaries/english_loanwords_tth.json.
    """
    path = Path(dict_path) if dict_path is not None else DEFAULT_STATIC_DICTIONARY_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        logger.warning(f"Static loanword dictionary not found at {path}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning(f"Failed to load static loanword dictionary from {path}: {e}")
    return {}


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
    if not matrix_path.is_absolute():
        matrix_path = PROJECT_ROOT / matrix_path

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Production confusion matrix not found at '{matrix_path}'."
        )

    matrix = AcousticConfusionMatrix.load(matrix_path)
    if path is None:
        _DEFAULT_CONFUSION_MATRIX = matrix
    return matrix


class CherokeeSyntheticTargetProjector(SyntheticTargetProjector):
    """
    Cherokee-specialized SyntheticTargetProjector that produces SyntheticCherokeeTarget
    results (with syllabary conversion and CherokeeToken sequences) while delegating
    all core DP tiling and static lookup logic to Tier 1 SyntheticTargetProjector.
    """

    def project_word(
        self,
        word: str,
        matrix: Optional[GenericConfusionMatrixProtocol] = None,
    ) -> SyntheticCherokeeTarget:
        """
        Projects a single English word to a SyntheticCherokeeTarget.
        """
        raw_word = word.strip()
        cleaned = _WORD_BOUNDARY_PUNCTUATION_RE.sub("", raw_word).lower()
        if not cleaned:
            return SyntheticCherokeeTarget(
                source_word=raw_word,
                arpabet_tokens=(),
                cherokee_tokens=(),
                projected_text="",
            )

        # 1. Static memoized dictionary lookup
        if cleaned in self._static_dict:
            entry = self._static_dict[cleaned]
            if isinstance(entry, SyntheticCherokeeTarget):
                return entry
            elif isinstance(entry, dict):
                # Ensure it has cherokee_tokens or convert from target_tokens
                return SyntheticCherokeeTarget.from_dict(entry)
            elif isinstance(entry, str):
                syll = None
                try:
                    syll_cand = convert_orthography(
                        entry, source=Orthography.TTH, target=Orthography.SYLLABARY
                    )
                    if any(
                        0x13A0 <= ord(c) <= 0x13FF or 0xAB70 <= ord(c) <= 0xABBF
                        for c in syll_cand
                    ):
                        syll = syll_cand
                except Exception:
                    syll = None
                return SyntheticCherokeeTarget(
                    source_word=raw_word,
                    arpabet_tokens=(),
                    cherokee_tokens=(),
                    projected_text=entry,
                    syllabary=syll,
                )

        # 2. Dynamic runtime cache lookup
        if cleaned in self._runtime_cache:
            entry = self._runtime_cache[cleaned]
            if isinstance(entry, SyntheticCherokeeTarget):
                return entry

        active_matrix = matrix if matrix is not None else self._matrix
        if active_matrix is None:
            active_matrix = load_default_confusion_matrix()

        # Delegate base word projection to Tier 1 engine
        base_target = super().project_word(cleaned, matrix=active_matrix)

        if isinstance(base_target, SyntheticCherokeeTarget):
            self._runtime_cache[cleaned] = base_target
            return base_target

        # Convert target tokens to CherokeeToken instances
        target_tokens = getattr(base_target, "target_tokens", ())
        cherokee_tokens = tuple(
            CherokeeToken(phone=t, orthography=Orthography.TTH) for t in target_tokens
        )

        projected_tth = getattr(base_target, "projected_text", "")

        # Syllabary transliteration
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

        arp_toks = tuple(
            ArpabetToken.from_dict(t) if not isinstance(t, ArpabetToken) else t
            for t in getattr(base_target, "arpabet_tokens", ())
        )

        target = SyntheticCherokeeTarget(
            source_word=raw_word,
            arpabet_tokens=arp_toks,
            cherokee_tokens=cherokee_tokens,
            projected_text=projected_tth,
            syllabary=syllabary,
            confidence_score=getattr(base_target, "confidence_score", 1.0),
            per_token_probabilities=getattr(base_target, "per_token_probabilities", ()),
        )
        self._runtime_cache[cleaned] = target
        return target

    def project_arpabet(
        self,
        arpabet_tokens: Sequence[Union[ARPAbetPhone, ArpabetToken]],
        matrix: Optional[GenericConfusionMatrixProtocol] = None,
        source_word: str = "",
    ) -> SyntheticCherokeeTarget:
        """
        Projects an ARPAbet token sequence into a SyntheticCherokeeTarget.
        """
        active_matrix = matrix if matrix is not None else self._matrix
        if active_matrix is None:
            active_matrix = load_default_confusion_matrix()

        base_target = super().project_arpabet(
            arpabet_tokens=arpabet_tokens,
            matrix=active_matrix,
            source_word=source_word,
        )

        if isinstance(base_target, SyntheticCherokeeTarget):
            return base_target

        target_tokens = getattr(base_target, "target_tokens", ())
        cherokee_tokens = tuple(
            CherokeeToken(phone=t, orthography=Orthography.TTH) for t in target_tokens
        )

        projected_tth = getattr(base_target, "projected_text", "")

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

        arp_toks = tuple(
            ArpabetToken.from_dict(t) if not isinstance(t, ArpabetToken) else t
            for t in getattr(base_target, "arpabet_tokens", ())
        )

        return SyntheticCherokeeTarget(
            source_word=source_word,
            arpabet_tokens=arp_toks,
            cherokee_tokens=cherokee_tokens,
            projected_text=projected_tth,
            syllabary=syllabary,
            confidence_score=getattr(base_target, "confidence_score", 1.0),
            per_token_probabilities=getattr(base_target, "per_token_probabilities", ()),
        )


def make_cherokee_projector(
    matrix_path: Optional[Union[str, Path]] = None,
    dictionary_path: Optional[Union[str, Path]] = None,
    g2p: Optional[EnglishToArpabetProtocol] = None,
    matrix: Optional[AcousticConfusionMatrix] = None,
    target_phonemes: Optional[Sequence[str]] = None,
) -> CherokeeSyntheticTargetProjector:
    """
    Factory function returning a configured CherokeeSyntheticTargetProjector
    with Cherokee default paths, acoustic matrix, and static dictionary.
    """
    if matrix is not None:
        active_matrix = matrix
    else:
        active_matrix = load_default_confusion_matrix(matrix_path)

    dict_path = (
        dictionary_path
        if dictionary_path is not None
        else DEFAULT_STATIC_DICTIONARY_PATH
    )
    dict_resolved = Path(dict_path)
    if not dict_resolved.is_absolute():
        dict_resolved = PROJECT_ROOT / dict_resolved

    dict_arg: Optional[Path] = dict_resolved if dict_resolved.exists() else None

    return CherokeeSyntheticTargetProjector(
        matrix=active_matrix,
        dictionary=dict_arg,
        g2p=g2p,
    )


# Global singleton instances
_DEFAULT_CONFUSION_MATRIX: Optional[AcousticConfusionMatrix] = None
_DEFAULT_PROJECTOR: Optional[CherokeeSyntheticTargetProjector] = None


def get_default_projector(
    matrix_path: Optional[Union[str, Path]] = None,
    dictionary_path: Optional[Union[str, Path]] = None,
) -> CherokeeSyntheticTargetProjector:
    """
    Returns or lazily initializes the default global CherokeeSyntheticTargetProjector instance.
    """
    global _DEFAULT_PROJECTOR
    if (
        _DEFAULT_PROJECTOR is not None
        and matrix_path is None
        and dictionary_path is None
    ):
        return _DEFAULT_PROJECTOR

    projector = make_cherokee_projector(
        matrix_path=matrix_path,
        dictionary_path=dictionary_path,
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


def _default_syllabary_to_tth(text: str) -> str:
    return convert_orthography(
        text, source=Orthography.SYLLABARY, target=Orthography.TTH
    )


def normalize_code_switched_text(
    text: str,
    normalizer: Callable[[str], str] = _default_syllabary_to_tth,
    projector: Optional[SyntheticTargetProjectorProtocol] = None,
) -> str:
    """
    Normalizes mixed Cherokee Syllabary and English code-switched text into
    canonical Cherokee TTH phonetics for ASR alignment matching.
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
            norm = convert_orthography(
                tok, source=Orthography.SYLLABARY, target=Orthography.TTH
            )
            if norm:
                output_tokens.append(norm)
        elif is_english_word(tok):
            target = active_projector.project_word(tok)
            if target.projected_tth:
                output_tokens.append(target.projected_tth)
            else:
                norm = normalizer(tok)
                if norm:
                    output_tokens.append(norm)
        else:
            norm = normalizer(tok)
            if norm:
                output_tokens.append(norm)

    return " ".join(output_tokens)


__all__ = [
    "CHEROKEE_TTH_TARGET_PHONEMES",
    "CherokeeSyntheticTargetProjector",
    "DEFAULT_CONFUSION_MATRIX_PATH",
    "DEFAULT_STATIC_DICTIONARY_PATH",
    "SyntheticTargetProjector",
    "generate_static_dictionary",
    "get_default_projector",
    "get_english_loanwords_tth_dict",
    "is_english_word",
    "load_default_confusion_matrix",
    "make_cherokee_projector",
    "normalize_code_switched_text",
    "project_english_text",
    "project_english_word",
]
