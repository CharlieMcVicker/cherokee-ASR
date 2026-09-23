# -*- coding: utf-8 -*-
"""
transcription.cherokee.enrichment.syllable_alignment module.

Fine-grained character and syllable level alignment engine and phonetic reconciliation
for mapping ground-truth Cherokee Syllabary against ASR emitted phonetic text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from transcription.core.alignment.models import AlignmentOutput, WordInterval
from transcription.cherokee.orthography import (
    CHEROKEE_SYLLABARY_MAP,
    syllabary_to_phonetics,
)

VOWELS = set("aeiouvAEIOUV")


@dataclass(frozen=True)
class SyllableAlignment:
    """
    Data structure representing aligned character/syllable pair with sequence bounds.
    """

    syllabary_char: str
    base_phonetic: str
    emitted_text: str
    syl_start_idx: int
    syl_end_idx: int
    emitted_start_idx: int
    emitted_end_idx: int


def is_cherokee_syllable(char: str) -> bool:
    """Check if character is a Cherokee syllabary character."""
    code = ord(char) if char else 0
    return (
        (0x13A0 <= code <= 0x13F5)
        or (0xAB70 <= code <= 0xABBF)
        or (char in CHEROKEE_SYLLABARY_MAP)
    )


def get_base_transliteration(syllabary_text: str) -> str:
    """Convert Cherokee syllabary text into its base phonetic transliteration."""
    res = []
    for char in syllabary_text:
        if char in CHEROKEE_SYLLABARY_MAP:
            res.append(CHEROKEE_SYLLABARY_MAP[char])
        elif is_cherokee_syllable(char):
            res.append(CHEROKEE_SYLLABARY_MAP.get(char, syllabary_to_phonetics(char)))
        else:
            res.append(char)
    return "".join(res)


def _get_base_syllable(char: str) -> str:
    """Return default transliteration for a Cherokee syllabary character or character itself."""
    return CHEROKEE_SYLLABARY_MAP.get(char, char)


def _char_distance(c1: str, c2: str) -> float:
    """Compute distance between two phonetic characters."""
    c1_lower = c1.lower()
    c2_lower = c2.lower()
    if c1_lower == c2_lower:
        return 0.0

    # Soft matches for phonetically related sounds (e.g. t/d/th, k/g/kh, s/h, vowels v/u/o)
    vowels = {"a", "e", "i", "o", "u", "v"}
    if c1_lower in vowels and c2_lower in vowels:
        return 0.5

    dt_set = {"d", "t", "th"}
    if c1_lower in dt_set and c2_lower in dt_set:
        return 0.3

    gk_set = {"g", "k", "kh"}
    if c1_lower in gk_set and c2_lower in gk_set:
        return 0.3

    return 1.0


def _segment_distance(phon: str, emitted: str) -> float:
    """Edit distance between unit base phonetic and emitted phonetic substring."""
    len_p = len(phon)
    len_e = len(emitted)

    if phon == emitted:
        return 0.0

    d = [[0.0] * (len_e + 1) for _ in range(len_p + 1)]
    for i in range(len_p + 1):
        d[i][0] = i * 1.0
    for j in range(len_e + 1):
        d[0][j] = j * 1.0

    for i in range(1, len_p + 1):
        for j in range(1, len_e + 1):
            cost = _char_distance(phon[i - 1], emitted[j - 1])
            d[i][j] = min(
                d[i - 1][j] + 0.8,  # deletion
                d[i][j - 1] + 0.8,  # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )

    return d[len_p][len_e]


def _proportional_fallback(
    units: List[Tuple[str, str, int, int]], emitted_text: str
) -> List[SyllableAlignment]:
    """Fallback linear partition when DTW table has no path."""
    N = len(emitted_text)
    M = len(units)
    alignments = []
    for i, (syl_char, base_phon, s_start, s_end) in enumerate(units):
        e_start = int(round(i * N / M))
        e_end = int(round((i + 1) * N / M))
        alignments.append(
            SyllableAlignment(
                syllabary_char=syl_char,
                base_phonetic=base_phon,
                emitted_text=emitted_text[e_start:e_end],
                syl_start_idx=s_start,
                syl_end_idx=s_end,
                emitted_start_idx=e_start,
                emitted_end_idx=e_end,
            )
        )
    return alignments


def align_character_syllable_detailed(
    syllabary_text: str, emitted_text: str
) -> List[SyllableAlignment]:
    """
    Detailed character/syllable alignment returning sequence bounds and metadata.

    Args:
        syllabary_text: Ground truth Cherokee syllabary string.
        emitted_text: Emitted ASR phonetic string.

    Returns:
        List of SyllableAlignment objects.
    """
    if not syllabary_text:
        return []

    # Segment syllabary into tokens (characters + spaces/punctuation)
    units: List[Tuple[str, str, int, int]] = []
    idx = 0
    for char in syllabary_text:
        base_phon = CHEROKEE_SYLLABARY_MAP.get(
            char,
            syllabary_to_phonetics(char) if is_cherokee_syllable(char) else char,
        )
        units.append((char, base_phon, idx, idx + len(char)))
        idx += len(char)

    emitted_chars = list(emitted_text)
    M = len(units)
    N = len(emitted_chars)

    if N == 0:
        return [
            SyllableAlignment(
                syllabary_char=u[0],
                base_phonetic=u[1],
                emitted_text="",
                syl_start_idx=u[2],
                syl_end_idx=u[3],
                emitted_start_idx=0,
                emitted_end_idx=0,
            )
            for u in units
        ]

    # Dynamic Programming Alignment between syllabary units and emitted_text
    dp = [[float("inf")] * (N + 1) for _ in range(M + 1)]
    ptr = [[(0, 0)] * (N + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0

    for i in range(1, M + 1):
        unit_char, unit_phon = units[i - 1][0], units[i - 1][1]
        is_space_or_punc = not is_cherokee_syllable(unit_char)

        for j in range(N + 1):
            if dp[i - 1][j] == float("inf"):
                continue

            del_cost = 0.5 if is_space_or_punc else 1.5
            if dp[i - 1][j] + del_cost < dp[i][j]:
                dp[i][j] = dp[i - 1][j] + del_cost
                ptr[i][j] = (i - 1, j)

            max_k = max(len(unit_phon) + 3, 4) if not is_space_or_punc else 2
            for k in range(1, max_k + 1):
                if j + k > N:
                    break
                emitted_sub = "".join(emitted_chars[j : j + k])

                if is_space_or_punc:
                    if unit_char == emitted_sub:
                        match_cost = 0.0
                    elif unit_char.isspace() and emitted_sub.isspace():
                        match_cost = 0.0
                    else:
                        match_cost = 2.0
                else:
                    match_cost = _segment_distance(unit_phon, emitted_sub)

                if dp[i - 1][j] + match_cost < dp[i][j + k]:
                    dp[i][j + k] = dp[i - 1][j] + match_cost
                    ptr[i][j + k] = (i - 1, j)

    curr_i, curr_j = M, N
    if dp[M][N] == float("inf"):
        return _proportional_fallback(units, emitted_text)

    boundaries: Dict[int, Tuple[int, int]] = {}
    while curr_i > 0:
        prev_i, prev_j = ptr[curr_i][curr_j]
        boundaries[curr_i - 1] = (prev_j, curr_j)
        curr_i, curr_j = prev_i, prev_j

    alignments = []
    for idx_u, (syl_char, base_phon, s_start, s_end) in enumerate(units):
        e_start, e_end = boundaries.get(idx_u, (0, 0))
        emitted_sub = emitted_text[e_start:e_end]
        alignments.append(
            SyllableAlignment(
                syllabary_char=syl_char,
                base_phonetic=base_phon,
                emitted_text=emitted_sub,
                syl_start_idx=s_start,
                syl_end_idx=s_end,
                emitted_start_idx=e_start,
                emitted_end_idx=e_end,
            )
        )

    return alignments


def align_character_syllable(
    syllabary_text: str, emitted_text: str
) -> List[Tuple[str, str]]:
    """
    Fine-grained character and syllable level alignment function mapping
    ground-truth Cherokee Syllabary characters directly against ASR emitted phonetic text.
    """
    alignments = align_character_syllable_detailed(syllabary_text, emitted_text)
    return [(item.syllabary_char, item.emitted_text) for item in alignments]


def _enrich_single_syllable(base: str, emitted: str) -> str:
    """
    Enrich a single syllable base phonetic representation with ASR emitted acoustic features.
    """
    curr = base
    has_glottal_stop = "'" in emitted
    has_pre_h = emitted.startswith("h") and not base.startswith("h")

    # A. Lateral / Laryngeal / Digraph Aspiration Shift
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

    # C. Glottal stop handling
    if has_glottal_stop and "'" not in curr:
        if emitted.startswith("'"):
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
    if emitted.endswith("h") and not curr.endswith("h") and base and base[-1] in VOWELS:
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
        if len(curr) > 1 and curr[-1] in VOWELS:
            curr = curr[:-1]
        elif len(curr) == 1 and curr in VOWELS:
            curr = ""
        elif curr.endswith("h") and len(curr) > 1 and curr[-2] in VOWELS:
            curr = curr[:-1]

    return curr


def reconcile_phonetics(
    syllabary_text: str,
    base_transliteration: str,
    emitted_text: str,
    aligned_pairs: List[Union[Tuple[str, str], Any]],
) -> str:
    """
    Reconcile base transliteration of Cherokee syllabary with emitted ASR text using aligned pairs,
    enforcing phonetic enrichment rules while maintaining Cherokee Syllabary as the immutable structural anchor.
    """
    if not aligned_pairs:
        return base_transliteration if base_transliteration else ""

    result_parts = []

    for item in aligned_pairs:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            char = item[0]
            emitted_slice = item[1]
        elif hasattr(item, "syllabary_char") and hasattr(item, "emitted_text"):
            char = getattr(item, "syllabary_char")
            emitted_slice = getattr(item, "emitted_text")
        else:
            continue

        base_phon = _get_base_syllable(char)

        if char not in CHEROKEE_SYLLABARY_MAP:
            result_parts.append(emitted_slice if emitted_slice != "" else char)
            continue

        emitted_clean = emitted_slice.strip()

        if not emitted_clean:
            if len(base_phon) > 1 and base_phon[-1] in VOWELS:
                result_parts.append(base_phon[:-1])
            else:
                result_parts.append(base_phon)
            continue

        enriched = _enrich_single_syllable(base_phon, emitted_clean)
        result_parts.append(enriched)

    return "".join(result_parts)


def reconcile_word_intervals(
    words: Sequence[WordInterval],
    syllabary_text: str,
) -> List[WordInterval]:
    """
    Pure mapping: takes word intervals and syllabary reference text,
    returning a new list of WordIntervals where `word` contains the reconciled phonetics.
    """
    syll_text = syllabary_text.strip()
    if not syll_text or not words:
        return [
            WordInterval(
                word=w.word,
                start_sec=w.start_sec,
                end_sec=w.end_sec,
                confidence=w.confidence,
                flagged=w.flagged,
                emitted_word=w.emitted_word,
            )
            for w in words
        ]

    syll_words = [sw for sw in syll_text.split() if sw]
    reconciled: List[WordInterval] = []
    syll_idx = 0

    for w in words:
        k = max(1, len(w.word.split())) if w.word else 1
        if syll_words:
            if syll_idx < len(syll_words):
                target_syll_words = syll_words[syll_idx : syll_idx + k]
                syll_idx += k
                target_syll = " ".join(target_syll_words)
            else:
                target_syll = syll_words[-1]
        else:
            target_syll = syll_text

        target_emitted = w.emitted_word or w.word
        try:
            align_pairs = align_character_syllable(target_syll, target_emitted)
            base_trans = "".join([pair[0] for pair in align_pairs])
            rec_word = reconcile_phonetics(
                syllabary_text=target_syll,
                base_transliteration=base_trans,
                emitted_text=target_emitted,
                aligned_pairs=align_pairs,
            )
        except Exception:
            rec_word = w.word

        reconciled.append(
            WordInterval(
                word=rec_word,
                start_sec=w.start_sec,
                end_sec=w.end_sec,
                confidence=w.confidence,
                flagged=w.flagged,
                emitted_word=w.emitted_word,
            )
        )

    return reconciled


def reconcile_alignment_words(
    alignment: AlignmentOutput,
    syllabary_lookup: Dict[str, str],
) -> List[WordInterval]:
    """
    Reconciles all words across chunks in an AlignmentOutput,
    returning a single flattened list of reconciled WordIntervals.
    """
    all_reconciled: List[WordInterval] = []
    for chunk in alignment.aligned_chunks:
        syll_text = syllabary_lookup.get(chunk.chunk_id, "")
        all_reconciled.extend(reconcile_word_intervals(chunk.words, syll_text))
    return all_reconciled


def reconcile_alignment_by_chunk(
    alignment: AlignmentOutput,
    syllabary_lookup: Dict[str, str],
) -> Dict[str, List[WordInterval]]:
    """
    Reconciles words grouped by chunk_id.
    """
    result: Dict[str, List[WordInterval]] = {}
    for chunk in alignment.aligned_chunks:
        syll_text = syllabary_lookup.get(chunk.chunk_id, "")
        result[chunk.chunk_id] = reconcile_word_intervals(chunk.words, syll_text)
    return result


class SyllableAlignmentEngine:
    """
    Engine class encapsulating character/syllable alignment and phonetic reconciliation.
    """

    def __init__(self) -> None:
        pass

    def align(self, syllabary_text: str, emitted_text: str) -> List[Tuple[str, str]]:
        return align_character_syllable(syllabary_text, emitted_text)

    def align_detailed(
        self, syllabary_text: str, emitted_text: str
    ) -> List[SyllableAlignment]:
        return align_character_syllable_detailed(syllabary_text, emitted_text)

    def reconcile(
        self,
        syllabary_text: str,
        emitted_text: str,
        base_transliteration: Optional[str] = None,
    ) -> str:
        """
        Convenience reconciliation wrapper.
        """
        aligned_pairs = self.align(syllabary_text, emitted_text)
        base_trans = (
            base_transliteration
            if base_transliteration is not None
            else get_base_transliteration(syllabary_text)
        )
        return reconcile_phonetics(
            syllabary_text=syllabary_text,
            base_transliteration=base_trans,
            emitted_text=emitted_text,
            aligned_pairs=aligned_pairs,
        )

    def reconcile_words(
        self,
        alignment: AlignmentOutput,
        syllabary_lookup: Dict[str, str],
    ) -> List[WordInterval]:
        return reconcile_alignment_words(alignment, syllabary_lookup)

    def reconcile_by_chunk(
        self,
        alignment: AlignmentOutput,
        syllabary_lookup: Dict[str, str],
    ) -> Dict[str, List[WordInterval]]:
        return reconcile_alignment_by_chunk(alignment, syllabary_lookup)


__all__ = [
    "SyllableAlignment",
    "SyllableAlignmentEngine",
    "align_character_syllable",
    "align_character_syllable_detailed",
    "get_base_transliteration",
    "is_cherokee_syllable",
    "reconcile_alignment_by_chunk",
    "reconcile_alignment_words",
    "reconcile_phonetics",
    "reconcile_word_intervals",
]
