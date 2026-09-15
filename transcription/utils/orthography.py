# -*- coding: utf-8 -*-
"""
orthography.py

Domain models and pure transformation maps for Cherokee text orthographies.
Explicitly defines Orthography enums (SYLLABARY, DG, TTH), completely decouples tone
processing from consonant systems, and eliminates heuristic detection.
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Optional, Set

from transcription.utils.syllabary_map import (
    cherokee_to_bad_phonetics,
    phonetics_to_syllabary,
)
from transcription.utils.tone_normalization import respell_consonants

PUNCTUATION_REGEX = r"[\,\?\.\!\-\;\:\"\“\%\”\(\)\[\]\{\}«»…\´\‛]"
GLOTTAL_VARIANTS_REGEX = r"[\’\‘\ʼ\ʻ\`]"
VOWELS: Set[str] = set("aeiouvAEIOUV")


class Orthography(str, Enum):
    """
    Explicit enumeration of Cherokee consonant orthographies.
    Tone and vowel length are treated as an orthogonal dimension.
    """

    SYLLABARY = "syllabary"
    """Cherokee Unicode syllabary glyphs (e.g. ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ)."""

    DG = "dg"
    """Latin transliteration in base d/g consonant system (e.g. tsalagi gawonihisdi, otla)."""

    TTH = "tth"
    """Latin phonetics in aspirated t/th consonant system (standard training data format, e.g. tsalakhi khawonihisthi, olha)."""


def clean_punctuation_and_whitespace(text: str) -> str:
    """Removes punctuation and normalizes whitespace."""
    if not text:
        return ""
    t = text.lower()
    t = re.sub(GLOTTAL_VARIANTS_REGEX, "'", t)
    t = re.sub(PUNCTUATION_REGEX, "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def strip_tones_and_colons(text: str) -> str:
    """Strips numeric tone digits (0-9), combining diacritics, and vowel length colons."""
    if not text:
        return ""
    t = text.replace(":", "")
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"[\u0300-\u036f]", "", t)
    return t


def convert_orthography(
    text: str,
    source: Orthography,
    target: Orthography,
    strip_tones: bool = True,
) -> str:
    """
    Pure and deterministic transformation map between explicit orthographies.

    Short-circuits immediately with zero consonant transformation when source == target.

    Args:
        text: Input Cherokee string.
        source: Explicit source Orthography.
        target: Explicit target Orthography.
        strip_tones: Whether to strip tone digits and vowel-length colons.

    Returns:
        Converted Cherokee string in target orthography.
    """
    if not text:
        return ""

    # Short-circuit: when source == target, never run consonant mutation functions
    if source == target:
        t = strip_tones_and_colons(text) if strip_tones else text
        return (
            clean_punctuation_and_whitespace(t)
            if source != Orthography.SYLLABARY
            else text.strip()
        )

    # Step 1: Handle tone stripping and hyphen removal
    t = strip_tones_and_colons(text) if strip_tones else text
    t = t.replace("-", "")

    # Step 2: Source-specific conversion
    if source == Orthography.SYLLABARY:
        has_cherokee = any(
            0x13A0 <= ord(c) <= 0x13FF or 0xAB70 <= ord(c) <= 0xABBF for c in t
        )
        if has_cherokee:
            out = cherokee_to_bad_phonetics(t)
            return clean_punctuation_and_whitespace(out)
        else:
            # Latin transliteration passed with SYLLABARY source
            if target == Orthography.SYLLABARY:
                return phonetics_to_syllabary(t)
            if target == Orthography.TTH:
                t_low = t.lower()
                out = t_low.replace("qu", "gw").replace("tl", "dl").replace("TL", "dl")
                out = respell_consonants(out)
                out = re.sub(r"([aeiouvAEIOUV])(?=[aeiouvAEIOUV])", r"\1'", out)
                return clean_punctuation_and_whitespace(out)
            if target == Orthography.DG:
                out = re.sub(r"([aeiouvAEIOUV])(?=[aeiouvAEIOUV])", r"\1'", t)
                return clean_punctuation_and_whitespace(out)

    elif source == Orthography.DG:
        if target == Orthography.SYLLABARY:
            return phonetics_to_syllabary(t)

        if target == Orthography.TTH:
            t_low = t.lower()
            out = t_low.replace("qu", "gw").replace("tl", "dl").replace("TL", "dl")
            out = respell_consonants(out)
            out = re.sub(r"([aeiouvAEIOUV])(?=[aeiouvAEIOUV])", r"\1'", out)
            return clean_punctuation_and_whitespace(out)

    elif source == Orthography.TTH:
        if target == Orthography.SYLLABARY:
            return phonetics_to_syllabary(t)

        if target == Orthography.DG:
            # TTH -> DG is already in Latin phonetics; clean punctuation
            return clean_punctuation_and_whitespace(t)

    return clean_punctuation_and_whitespace(t)


__all__ = [
    "Orthography",
    "convert_orthography",
    "clean_punctuation_and_whitespace",
    "strip_tones_and_colons",
]
