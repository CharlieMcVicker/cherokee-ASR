# -*- coding: utf-8 -*-
"""
enrich_syllabary.py

Phonetic Rule Merger Engine for enriching immutable Cherokee Syllabary base transliteration
with ASR acoustic features (vowel deletion/syncopation, aspiration/glottal activity, laryngeal toggles).
"""

from typing import List, Tuple, Union, Any
from transcription.syllabary_enrichment.alignment_engine import CHEROKEE_SYLLABARY_MAP

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
        if hasattr(item, "syllabary_char") and hasattr(item, "emitted_text"):
            char = item.syllabary_char
            emitted_slice = item.emitted_text
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            char = item[0]
            emitted_slice = item[1]
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
    """
    # 1. Glottal stop injection: If emitted contains glottal stop '\'', preserve glottal stop position/activity
    has_glottal_stop = "'" in emitted

    # 2. Check pre-aspiration 'h' or internal 'h' in emitted
    has_pre_h = emitted.startswith("h") and not base.startswith("h")

    # 3. Laryngeal Toggles & Consonant modifications (t -> th, k -> kh, d -> th, g -> kh, d -> t, g -> k, etc.)
    curr = base

    # Toggle laryngeal stops:
    # d/t -> th if emitted has 'th'
    if ("th" in emitted or emitted.startswith("th")) and (
        curr.startswith("d") or curr.startswith("t")
    ):
        if curr.startswith("d"):
            curr = "th" + curr[1:]
        elif curr.startswith("t") and not curr.startswith("th"):
            curr = "th" + curr[1:]
    # g/k -> kh if emitted has 'kh'
    elif ("kh" in emitted or emitted.startswith("kh")) and (
        curr.startswith("g") or curr.startswith("k")
    ):
        if curr.startswith("g"):
            curr = "kh" + curr[1:]
        elif curr.startswith("k") and not curr.startswith("kh"):
            curr = "kh" + curr[1:]
    # d -> t if emitted starts with 't'
    elif emitted.startswith("t") and curr.startswith("d"):
        curr = "t" + curr[1:]
    # g -> k if emitted starts with 'k'
    elif emitted.startswith("k") and curr.startswith("g"):
        curr = "k" + curr[1:]

    # Pre-aspiration 'h'
    if has_pre_h:
        curr = "h" + curr

    # Glottal stop handling:
    if has_glottal_stop and "'" not in curr:
        if emitted.startswith("'"):
            curr = "'" + curr
        elif emitted.endswith("'"):
            curr = curr + "'"
        else:
            curr = curr + "'"

    # 4. Rule 1: Syncopation / Vowel Deletion
    base_vowel = base[-1] if (len(base) > 0 and base[-1] in VOWELS) else None
    emitted_has_vowel = any(c in VOWELS for c in emitted)

    if base_vowel and not emitted_has_vowel:
        # Final vowel was syncopated / dropped in ASR window!
        if len(curr) > 1 and curr[-1] in VOWELS:
            curr = curr[:-1]
        elif len(curr) == 1 and curr in VOWELS:
            curr = ""

    return curr
