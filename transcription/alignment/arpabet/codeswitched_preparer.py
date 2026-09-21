# -*- coding: utf-8 -*-
"""
transcription.alignment.arpabet.codeswitched_preparer module.

Ground-truth text preparation and token discrimination for code-switched transcripts
containing mixed Cherokee Syllabary, English loanwords/names, compound clitic tokens
(e.g., JayᎢ -> English 'Jay' + Cherokee Syllabary 'Ꭲ'), and speaker labels.

Follows Types and Maps architectural principles:
- Immutable domain models (TokenType, CodeSwitchedToken, CodeSwitchedLineResult).
- Pure parsing and projection transformations decoupled from I/O boundaries.
- Strict isolation of English tokens from Cherokee DG-to-TTH consonant mutation (ZERO double-conversion).
- Lossless dictionary serialization/deserialization for all token and line models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from transcription.alignment.arpabet.projector import get_default_projector
from transcription.alignment.arpabet.types import SyntheticTargetProjectorProtocol
from transcription.alignment.normalizers import normalize_syllabary_for_alignment
from transcription.alignment.phonotactics import (
    get_intrusion_site_mask,
    get_syncope_mask,
)

# Regex matching compound clitic: English Latin stem + Cherokee Syllabary clitic/suffix
# Cherokee Syllabary unicode range: U+13A0-U+13FF (main) and U+AB70-U+ABBF (supplement)
_COMPOUND_CLITIC_RE = re.compile(r"^([a-zA-Z]+)([\u13A0-\u13FF\uAB70-\uABBF]+)$")

# Regex matching boundary punctuation (excluding internal apostrophes/glottals)
_BOUNDARY_PUNCT_RE = re.compile(
    r"^[^a-zA-Z0-9\u13A0-\u13FF\uAB70-\uABBF']+|[^a-zA-Z0-9\u13A0-\u13FF\uAB70-\uABBF']+$"
)


class TokenType(str, Enum):
    """
    Discriminative token categories for code-switched transcript text.
    """

    CHEROKEE_SYLLABARY = "cherokee_syllabary"
    """Pure Cherokee Syllabary token (e.g. ᎯᎠ, ᏓᏩᏙ, ᎣᏏᏍ)."""

    ENGLISH = "english"
    """Pure English word or proper name (e.g. Soldier, Charley, McCoy, yeah)."""

    COMPOUND_CLITIC = "compound_clitic"
    """Compound token with English stem and Cherokee Syllabary clitic (e.g. JayᎢ, WellingᏛ)."""

    PUNCTUATION = "punctuation"
    """Punctuation-only token or boundary symbol (e.g. ?, ,, ..., :)."""


@dataclass(frozen=True)
class CodeSwitchedToken:
    """
    Immutable representation of an individual token within code-switched text.

    Preserves multi-tier word metadata:
    - Raw token and display representation.
    - Script token classification (TokenType).
    - Segmented English stem and Syllabary clitic for compound clitics.
    - Canonical Cherokee TTH phonetic target.
    - Phonotactic syncope and intrusion masks (zeroed for English stems/words).
    """

    raw_token: str
    token_type: TokenType
    english_stem: Optional[str] = None
    syllabary_clitic: Optional[str] = None
    canonical_tth: str = ""
    source_display: str = ""
    syncope_mask: Tuple[bool, ...] = field(default_factory=tuple)
    intrusion_mask: Tuple[bool, ...] = field(default_factory=tuple)

    @property
    def syllabary_tier(self) -> Optional[str]:
        """Syllabary representation for alignment display tiers."""
        if self.token_type == TokenType.CHEROKEE_SYLLABARY:
            return self.source_display
        elif self.token_type == TokenType.COMPOUND_CLITIC:
            return self.source_display
        return None

    @property
    def english_tier(self) -> Optional[str]:
        """English representation for alignment display tiers."""
        if self.token_type in (TokenType.ENGLISH, TokenType.COMPOUND_CLITIC):
            return self.english_stem
        return None

    @property
    def reconciled_tier(self) -> str:
        """Canonical TTH acoustic phonetic representation."""
        return self.canonical_tth

    def to_dict(self) -> Dict[str, Any]:
        """Serializes token to a clean dictionary."""
        return {
            "raw_token": self.raw_token,
            "token_type": self.token_type.value,
            "english_stem": self.english_stem,
            "syllabary_clitic": self.syllabary_clitic,
            "canonical_tth": self.canonical_tth,
            "source_display": self.source_display,
            "syncope_mask": list(self.syncope_mask),
            "intrusion_mask": list(self.intrusion_mask),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CodeSwitchedToken:
        """Deserializes token from a dictionary."""
        return cls(
            raw_token=str(data["raw_token"]),
            token_type=TokenType(data["token_type"]),
            english_stem=data.get("english_stem"),
            syllabary_clitic=data.get("syllabary_clitic"),
            canonical_tth=str(data.get("canonical_tth", "")),
            source_display=str(data.get("source_display", "")),
            syncope_mask=tuple(bool(x) for x in data.get("syncope_mask", [])),
            intrusion_mask=tuple(bool(x) for x in data.get("intrusion_mask", [])),
        )


@dataclass(frozen=True)
class CodeSwitchedLineResult:
    """
    Immutable result of preparing a code-switched line of text for alignment.

    Contains:
    - raw_text: Original input line.
    - tokens: Tuple of classified and projected CodeSwitchedToken instances.
    - unified_tth: Canonical Cherokee TTH phonetic target sequence ready for ASR alignment.
    - speaker: Optional extracted speaker prefix.
    """

    raw_text: str
    tokens: Tuple[CodeSwitchedToken, ...] = field(default_factory=tuple)
    unified_tth: str = ""
    speaker: Optional[str] = None

    @property
    def syllabary_tier_tokens(self) -> Tuple[str, ...]:
        """Syllabary display tokens."""
        return tuple(
            t.source_display
            for t in self.tokens
            if t.token_type in (TokenType.CHEROKEE_SYLLABARY, TokenType.COMPOUND_CLITIC)
        )

    @property
    def english_tier_tokens(self) -> Tuple[str, ...]:
        """English stem tokens."""
        return tuple(t.english_stem for t in self.tokens if t.english_stem is not None)

    @property
    def reconciled_tier_tokens(self) -> Tuple[str, ...]:
        """Non-empty canonical TTH tokens."""
        return tuple(t.canonical_tth for t in self.tokens if t.canonical_tth)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result to a dictionary."""
        return {
            "raw_text": self.raw_text,
            "tokens": [t.to_dict() for t in self.tokens],
            "unified_tth": self.unified_tth,
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CodeSwitchedLineResult:
        """Deserializes result from a dictionary."""
        tokens = tuple(CodeSwitchedToken.from_dict(t) for t in data.get("tokens", []))
        return cls(
            raw_text=str(data["raw_text"]),
            tokens=tokens,
            unified_tth=str(data.get("unified_tth", "")),
            speaker=data.get("speaker"),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serializes result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> CodeSwitchedLineResult:
        """Deserializes result from JSON string."""
        return cls.from_dict(json.loads(s))


def strip_boundary_punctuation(text: str) -> str:
    """
    Strips leading and trailing non-alphanumeric punctuation while preserving
    internal apostrophes/glottals and characters.
    """
    if not text:
        return ""
    return _BOUNDARY_PUNCT_RE.sub("", text.strip())


def split_compound_clitic(token: str) -> Optional[Tuple[str, str]]:
    """
    Checks if token matches ^([a-zA-Z]+)([\\u13A0-\\u13FF\\uAB70-\\uABBF]+)$
    (e.g. JayᎢ -> ("Jay", "Ꭲ"), WellingᏛ -> ("Welling", "Ꮫ")).

    Also matches if boundary punctuation surrounds the compound token (e.g. JayᎢ?).
    Returns (english_stem, syllabary_clitic) tuple, or None if not a compound clitic.
    """
    cleaned = token.strip()
    match = _COMPOUND_CLITIC_RE.match(cleaned)
    if match:
        return match.group(1), match.group(2)

    punct_stripped = strip_boundary_punctuation(cleaned)
    if punct_stripped and punct_stripped != cleaned:
        match_punct = _COMPOUND_CLITIC_RE.match(punct_stripped)
        if match_punct:
            return match_punct.group(1), match_punct.group(2)

    return None


def classify_token(token: str) -> TokenType:
    """
    Classifies an input token into TokenType:
    - COMPOUND_CLITIC: Latin stem with attached Syllabary clitic (e.g. JayᎢ).
    - CHEROKEE_SYLLABARY: Pure Cherokee Syllabary glyphs (e.g. ᎯᎠ, ᏓᏩᏙ).
    - ENGLISH: Pure Latin letters (e.g. Soldier, Charley, coffee).
    - PUNCTUATION: Punctuation, symbols, or whitespace.
    """
    stripped = token.strip()
    if not stripped:
        return TokenType.PUNCTUATION

    # 1. Compound clitic check
    if split_compound_clitic(stripped) is not None:
        return TokenType.COMPOUND_CLITIC

    # Strip boundary punctuation for script detection
    clean = strip_boundary_punctuation(stripped)
    if not clean:
        return TokenType.PUNCTUATION

    has_syllabary = any(
        0x13A0 <= ord(c) <= 0x13FF or 0xAB70 <= ord(c) <= 0xABBF for c in clean
    )
    has_latin = any(c.isalpha() and ord(c) < 0x13A0 for c in clean)

    if has_syllabary and not has_latin:
        return TokenType.CHEROKEE_SYLLABARY
    elif has_latin and not has_syllabary:
        return TokenType.ENGLISH
    elif not has_syllabary and not has_latin:
        return TokenType.PUNCTUATION
    else:
        # Fallback if mixed characters do not match the compound clitic regex
        if has_syllabary:
            return TokenType.CHEROKEE_SYLLABARY
        return TokenType.ENGLISH


def prepare_code_switched_token(
    token: str,
    projector: Optional[SyntheticTargetProjectorProtocol] = None,
    contextual_preaspiration: bool = True,
) -> CodeSwitchedToken:
    """
    Parses and projects an individual token into a CodeSwitchedToken with canonical TTH phonetics.

    Ensures ZERO double-conversion:
    - Pure Cherokee Syllabary is converted directly via normalize_syllabary_for_alignment.
    - Pure English words are projected strictly via projector.project_word (never Cherokee DG-to-TTH).
    - Compound tokens (English stem + Syllabary clitic) are cleanly segmented, with the English
      stem projected via projector and the clitic converted directly from Syllabary to TTH.
    - Punctuation emits empty canonical TTH.
    """
    active_projector = projector if projector is not None else get_default_projector()
    stripped = token.strip()
    tok_type = classify_token(stripped)

    if tok_type == TokenType.COMPOUND_CLITIC:
        split_res = split_compound_clitic(stripped)
        if split_res is None:
            split_res = (strip_boundary_punctuation(stripped), "")
        stem, clitic = split_res

        # Project English stem strictly via projector (zero DG-to-TTH mutation)
        stem_target = active_projector.project_word(stem)
        stem_tth = stem_target.projected_tth

        # Convert Cherokee Syllabary clitic directly to canonical TTH
        clitic_tth = (
            normalize_syllabary_for_alignment(
                clitic, contextual_preaspiration=contextual_preaspiration
            )
            if clitic
            else ""
        )
        canonical_tth = f"{stem_tth}{clitic_tth}"

        # English stem receives ZERO syncope/intrusion; Syllabary clitic receives Cherokee phonotactics
        stem_syncope = (False,) * len(stem_tth)
        stem_intrusion = (False,) * len(stem_tth)
        clitic_syncope = (
            tuple(get_syncope_mask(clitic_tth, return_char_mask=True))
            if clitic_tth
            else ()
        )
        clitic_intrusion = (
            tuple(get_intrusion_site_mask(clitic_tth, return_char_mask=True))
            if clitic_tth
            else ()
        )

        return CodeSwitchedToken(
            raw_token=token,
            token_type=TokenType.COMPOUND_CLITIC,
            english_stem=stem,
            syllabary_clitic=clitic,
            canonical_tth=canonical_tth,
            source_display=stripped,
            syncope_mask=stem_syncope + clitic_syncope,
            intrusion_mask=stem_intrusion + clitic_intrusion,
        )

    elif tok_type == TokenType.ENGLISH:
        clean_word = strip_boundary_punctuation(stripped)
        # Project English word strictly via projector (zero DG-to-TTH mutation)
        target = active_projector.project_word(clean_word)
        # English words receive strictly ZERO Cherokee syncope and ZERO intrusion
        zero_syncope = (False,) * len(target.projected_tth)
        zero_intrusion = (False,) * len(target.projected_tth)

        return CodeSwitchedToken(
            raw_token=token,
            token_type=TokenType.ENGLISH,
            english_stem=clean_word,
            syllabary_clitic=None,
            canonical_tth=target.projected_tth,
            source_display=stripped,
            syncope_mask=zero_syncope,
            intrusion_mask=zero_intrusion,
        )

    elif tok_type == TokenType.CHEROKEE_SYLLABARY:
        clean_word = strip_boundary_punctuation(stripped)
        # Convert Cherokee Syllabary directly to canonical TTH phonetics
        norm_tth = normalize_syllabary_for_alignment(
            clean_word, contextual_preaspiration=contextual_preaspiration
        )
        # Native Cherokee Syllabary receives full Cherokee phonotactic analysis
        syll_syncope = tuple(get_syncope_mask(norm_tth, return_char_mask=True))
        syll_intrusion = tuple(get_intrusion_site_mask(norm_tth, return_char_mask=True))

        return CodeSwitchedToken(
            raw_token=token,
            token_type=TokenType.CHEROKEE_SYLLABARY,
            english_stem=None,
            syllabary_clitic=None,
            canonical_tth=norm_tth,
            source_display=stripped,
            syncope_mask=syll_syncope,
            intrusion_mask=syll_intrusion,
        )

    else:  # TokenType.PUNCTUATION
        return CodeSwitchedToken(
            raw_token=token,
            token_type=TokenType.PUNCTUATION,
            english_stem=None,
            syllabary_clitic=None,
            canonical_tth="",
            source_display=stripped,
            syncope_mask=(),
            intrusion_mask=(),
        )


def extract_speaker_prefix(text: str) -> Tuple[Optional[str], str]:
    """
    Extracts speaker prefix if present in text (e.g. 'Guy Soldier: ᎯᏅ ...' -> ('Guy Soldier', 'ᎯᏅ ...')).
    Returns (speaker, spoken_text). If no speaker prefix is detected, returns (None, text).
    """
    stripped = text.strip()
    if ":" in stripped:
        parts = stripped.split(":", 1)
        candidate = parts[0].strip()
        body = parts[1].strip()
        if candidate and body and len(candidate) <= 40 and "\n" not in candidate:
            return candidate, body
    return None, stripped


def create_groundtruth_for_code_switched_syllabary(
    text: str,
    projector: Optional[SyntheticTargetProjectorProtocol] = None,
    strip_speaker: bool = False,
    contextual_preaspiration: bool = True,
) -> CodeSwitchedLineResult:
    """
    Prepares mixed Cherokee Syllabary and English code-switched text for acoustic alignment
    with zero double conversion.

    Discriminates tokens into pure Cherokee Syllabary, pure English, compound clitics
    (e.g. JayᎢ -> Jay + Ꭲ), and punctuation.

    Converts Cherokee Syllabary directly to canonical TTH phonetics.
    Projects English tokens strictly through SyntheticTargetProjector, preventing
    Cherokee DG-to-TTH consonant mutation from corrupting English words.

    Args:
        text: Input code-switched text line (e.g. 'Guy Soldier: ᎯᏅ ...' or 'JayᎢ ᏂᏛᎩᎶᏒ').
        projector: Optional projector implementing SyntheticTargetProjectorProtocol.
                   Defaults to get_default_projector().
        strip_speaker: If True, strips leading speaker prefix (e.g. 'Guy Soldier:') and
                       records speaker on the result. If False, preserves all tokens.
        contextual_preaspiration: Whether to apply contextual pre-aspiration (suppressing
            leading 'h' before word-initial 's' and affricates) or unconditional 'hs' conversion.

    Returns:
        CodeSwitchedLineResult containing token breakdown, speaker, and unified canonical TTH string.
    """
    if not text or not text.strip():
        return CodeSwitchedLineResult(
            raw_text=text,
            tokens=(),
            unified_tth="",
            speaker=None,
        )

    active_projector: SyntheticTargetProjectorProtocol = (
        projector if projector is not None else get_default_projector()
    )

    speaker: Optional[str] = None
    text_to_process = text.strip()

    if strip_speaker:
        speaker, text_to_process = extract_speaker_prefix(text_to_process)

    # Tokenize on whitespace
    raw_tokens = text_to_process.split()
    processed_tokens: List[CodeSwitchedToken] = []

    for tok in raw_tokens:
        tok_obj = prepare_code_switched_token(
            tok,
            projector=active_projector,
            contextual_preaspiration=contextual_preaspiration,
        )
        processed_tokens.append(tok_obj)

    tth_parts = [t.canonical_tth for t in processed_tokens if t.canonical_tth]
    unified_tth = " ".join(tth_parts)

    return CodeSwitchedLineResult(
        raw_text=text,
        tokens=tuple(processed_tokens),
        unified_tth=unified_tth,
        speaker=speaker,
    )


__all__ = [
    "TokenType",
    "CodeSwitchedToken",
    "CodeSwitchedLineResult",
    "strip_boundary_punctuation",
    "split_compound_clitic",
    "classify_token",
    "prepare_code_switched_token",
    "extract_speaker_prefix",
    "create_groundtruth_for_code_switched_syllabary",
]
