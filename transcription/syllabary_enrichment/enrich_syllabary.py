# -*- coding: utf-8 -*-
"""
enrich_syllabary.py

Phonetic Rule Merger Engine for enriching immutable Cherokee Syllabary base transliteration
with ASR acoustic features (vowel deletion/syncopation, aspiration/glottal activity, laryngeal toggles).
"""

from typing import List, Tuple, Union, Any
from transcription.utils.syllabary_map import CHEROKEE_SYLLABARY_MAP

VOWELS = set("aeiouvAEIOUV")


def _get_base_syllable(char: str) -> str:
    """Return default transliteration for a Cherokee syllabary character or character itself."""
    return CHEROKEE_SYLLABARY_MAP.get(char, char)


def reconcile_phonetics(
    syllabary_text: str,
    base_transliteration: str,
    emitted_text: str,
    aligned_pairs: List[Union[Tuple[str, str], Any]],
) -> str:
    """
    Reconcile base transliteration of Cherokee syllabary with emitted ASR text using aligned pairs,
    enforcing phonetic enrichment rules while maintaining Cherokee Syllabary as the immutable structural anchor.

    Rules:
    - Rule 1: Vowel deletion / syncopation based on aligned ASR window (e.g. dropping vowels when ASR indicates vocalic deletion).
    - Rule 2: Aspiration and glottal activity transfer from ASR (pre-aspiration 'h', glottal stop ''',
              and laryngeal toggles such as 't' -> 'th', 'k' -> 'kh').

    Args:
        syllabary_text: Immutable Cherokee syllabary string (e.g., "ᎠᏓᎴᏂᏍᎬ")
        base_transliteration: Un-enriched base transliteration (e.g., "adalenisgv")
        emitted_text: Emitted ASR phonetic text (e.g., "athaleniskv")
        aligned_pairs: List of tuples (syllabary_char, aligned_emitted_slice) or SyllableAlignment objects.

    Returns:
        Enriched phonetic text string.
    """
    if not aligned_pairs:
        return base_transliteration if base_transliteration else ""

    result_parts = []

    for item in aligned_pairs:
        # Extract syllabary_char and emitted_slice whether item is tuple or SyllableAlignment
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            char = item[0]
            emitted_slice = item[1]
        elif hasattr(item, "syllabary_char") and hasattr(item, "emitted_text"):
            char = getattr(item, "syllabary_char")
            emitted_slice = getattr(item, "emitted_text")
        else:
            continue

        base_phon = _get_base_syllable(char)

        # Non-syllabary elements (spaces, control tags, punctuation) pass through as emitted or base
        if char not in CHEROKEE_SYLLABARY_MAP:
            result_parts.append(emitted_slice if emitted_slice != "" else char)
            continue

        # Clean emitted slice (strip whitespace for phonetic checking inside a syllable unit)
        emitted_clean = emitted_slice.strip()

        if not emitted_clean:
            # Empty emitted slice for a syllabary character => Check Rule 1 (syncopation)
            # If base syllable ends in vowel and ASR omitted it, keep consonant base or drop vowel
            if len(base_phon) > 1 and base_phon[-1] in VOWELS:
                # Omit final vowel (syncopation)
                result_parts.append(base_phon[:-1])
            else:
                result_parts.append(base_phon)
            continue

        # Analyze base_phon vs emitted_clean to perform enrichment
        enriched = _enrich_single_syllable(base_phon, emitted_clean)
        result_parts.append(enriched)

    return "".join(result_parts)


def _enrich_single_syllable(base: str, emitted: str) -> str:
    """
    Enrich a single syllable base phonetic representation with ASR emitted acoustic features.

    Two Core Rules:
    1. Aspiration Transfer (Rule 2): Pre-aspiration 'h', post-consonantal/laryngeal aspiration (th, kh, lh, nh, wh, yh, sh, ch),
       and post-vocalic aspiration 'h' emitted by ASR are systematically preserved.
    2. Vowel Deletion / Syncopation (Rule 1): If ASR omitted the vowel for a vocalic base syllable, drop the vowel
       while strictly preserving all ASR-indicated aspiration and glottal stop activity.
    """
    curr = base
    has_glottal_stop = "'" in emitted
    has_pre_h = emitted.startswith("h") and not base.startswith("h")

    # A. Lateral / Laryngeal / Digraph Aspiration Shift across all consonant series
    # Map (consonant_prefix, emitted_indicator, new_consonant_prefix)
    aspiration_shifts = [
        ("tl", "lh", "lh"),
        ("l", "lh", "lh"),
        ("n", "nh", "nh"),
        ("hn", "nh", "nh"),
        ("w", "wh", "wh"),
        ("hw", "wh", "wh"),
        ("y", "yh", "yh"),
        ("hy", "yh", "yh"),
        ("r", "rh", "rh"),
        ("hr", "rh", "rh"),
        ("t", "th", "th"),
        ("d", "th", "th"),
        ("k", "kh", "kh"),
        ("g", "kh", "kh"),
        ("s", "sh", "sh"),
        ("c", "ch", "ch"),
        ("ts", "tsh", "tsh"),
    ]

    for orig_pfx, trigger, new_pfx in aspiration_shifts:
        if trigger in emitted or emitted.startswith(trigger):
            if curr.startswith(orig_pfx) and not curr.startswith(new_pfx):
                curr = new_pfx + curr[len(orig_pfx) :]
                break

    # B. Pre-aspiration 'h'
    if has_pre_h and not curr.startswith("h"):
        curr = "h" + curr

    # C. Glottal stop handling:
    # Drop standalone onset glottal stop from ASR unless accompanied by another onset consonant
    # (e.g., ASR "'a" for base "ya" drops onset glottal to output "ya", assuming "'" was a mistranscription of "y").
    if has_glottal_stop and "'" not in curr:
        if emitted.startswith("'"):
            # Check if there is an explicit onset consonant in emitted after "'" and before any vowel
            emitted_after_glottal = emitted[1:]
            has_onset_consonant = False
            for ch in emitted_after_glottal:
                if ch in VOWELS:
                    break
                if ch.isalpha() and ch.lower() not in VOWELS:
                    has_onset_consonant = True
                    break
            if has_onset_consonant:
                curr = "'" + curr
        elif emitted.endswith("'"):
            curr = curr + "'"
        else:
            curr = curr + "'"

    # D. Post-vocalic / vocalic aspiration 'h' transfer
    if emitted.endswith("h") and not curr.endswith("h") and base[-1] in VOWELS:
        if (
            len(emitted) >= 2
            and emitted[-2] in VOWELS
            and emitted[-2:]
            not in ("th", "kh", "lh", "nh", "wh", "yh", "rh", "sh", "ch")
        ) or any(c in VOWELS for c in emitted[:-1]):
            curr = curr + "h"

    # E. Rule 1: Syncopation / Vowel Deletion
    base_vowel = base[-1] if (len(base) > 0 and base[-1] in VOWELS) else None
    emitted_has_vowel = any(c in VOWELS for c in emitted)

    if base_vowel and not emitted_has_vowel:
        # Final vowel was syncopated / dropped in ASR window!
        if len(curr) > 1 and curr[-1] in VOWELS:
            curr = curr[:-1]
        elif len(curr) == 1 and curr in VOWELS:
            curr = ""
        elif curr.endswith("h") and len(curr) > 1 and curr[-2] in VOWELS:
            # If vowel was syncopated, strip trailing post-vocalic h if vowel drops
            curr = curr[:-1]

    return curr
