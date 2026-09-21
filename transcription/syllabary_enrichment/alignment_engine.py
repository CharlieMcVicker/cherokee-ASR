# -*- coding: utf-8 -*-
"""
alignment_engine.py

Fine-grained character and syllable level alignment engine for mapping ground-truth
Cherokee Syllabary characters (or syllable tokens) directly against ASR emitted phonetic text.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from transcription.utils.syllabary_map import (
    CHEROKEE_SYLLABARY_MAP,
    syllabary_to_phonetics,
)
from transcription.utils.tone_normalization import respell_consonants


@dataclass
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


def align_character_syllable(
    syllabary_text: str, emitted_text: str
) -> List[Tuple[str, str]]:
    """
    Fine-grained character and syllable level alignment function mapping
    ground-truth Cherokee Syllabary characters (or syllable tokens) directly against
    ASR emitted phonetic text.

    Args:
        syllabary_text: Ground truth Cherokee syllabary string (e.g. "<ctrl42>Ꭳ ᎠᏓᎴᏂᏍᎬ")
        emitted_text: Emitted ASR phonetic string (e.g. "no ataleniskv")

    Returns:
        List of (syllabary_char, aligned_emitted_text) tuples.
    """
    alignments = align_character_syllable_detailed(syllabary_text, emitted_text)
    return [(item.syllabary_char, item.emitted_text) for item in alignments]


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
    units: List[Tuple[str, str, int, int]] = (
        []
    )  # (char, base_phonetic, start_idx, end_idx)
    idx = 0
    for char in syllabary_text:
        base_phon = CHEROKEE_SYLLABARY_MAP.get(
            char,
            syllabary_to_phonetics(char) if is_cherokee_syllable(char) else char,
        )
        units.append((char, base_phon, idx, idx + len(char)))
        idx += len(char)

    # Dynamic Programming Alignment between syllabary units (expanded to phonetic length) and emitted_text
    emitted_chars = list(emitted_text)
    M = len(units)
    N = len(emitted_chars)

    # If emitted_text is empty, return empty mappings for each syllabary char
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

    # Pre-build phonetic string representation per unit for alignment cost matrix
    unit_phonetics = [u[1] for u in units]

    # We perform DTW or Needleman-Wunsch block alignment mapping units to emitted_text slices
    # Cost matrix: dp[i][j] is min cost to align first i units with first j emitted chars
    dp = [[float("inf")] * (N + 1) for _ in range(M + 1)]
    ptr = [[(0, 0)] * (N + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0

    # Fill DP
    for i in range(1, M + 1):
        unit_char, unit_phon = units[i - 1][0], units[i - 1][1]
        is_space_or_punc = not is_cherokee_syllable(unit_char)

        for j in range(N + 1):
            if dp[i - 1][j] == float("inf"):
                continue

            # Option 1: Map unit i to empty emitted slice (deletion/vowel drop)
            del_cost = 0.5 if is_space_or_punc else 1.5
            if dp[i - 1][j] + del_cost < dp[i][j]:
                dp[i][j] = dp[i - 1][j] + del_cost
                ptr[i][j] = (i - 1, j)

            # Option 2: Map unit i to k emitted characters (k >= 1)
            # Limit k based on length of unit_phon
            max_k = max(len(unit_phon) + 3, 4) if not is_space_or_punc else 2
            for k in range(1, max_k + 1):
                if j + k > N:
                    break
                emitted_sub = "".join(emitted_chars[j : j + k])

                # Calculate match cost between unit_phon and emitted_sub
                if is_space_or_punc:
                    if unit_char == emitted_sub:
                        match_cost = 0.0
                    elif unit_char.isspace() and emitted_sub.isspace():
                        match_cost = 0.0
                    else:
                        match_cost = 2.0
                else:
                    # Levenshtein-like edit distance between base phonetic and emitted sub
                    match_cost = _segment_distance(unit_phon, emitted_sub)

                if dp[i - 1][j] + match_cost < dp[i][j + k]:
                    dp[i][j + k] = dp[i - 1][j] + match_cost
                    ptr[i][j + k] = (i - 1, j)

    # Backtrack to find optimal boundaries
    curr_i, curr_j = M, N
    if dp[M][N] == float("inf"):
        # Fallback to simple linear proportional partition if DP fails
        return _proportional_fallback(units, emitted_text)

    boundaries = {}  # i -> (emitted_start, emitted_end)
    while curr_i > 0:
        prev_i, prev_j = ptr[curr_i][curr_j]
        boundaries[curr_i - 1] = (prev_j, curr_j)
        curr_i, curr_j = prev_i, prev_j

    alignments = []
    for idx, (syl_char, base_phon, s_start, s_end) in enumerate(units):
        e_start, e_end = boundaries.get(idx, (0, 0))
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


def _segment_distance(phon: str, emitted: str) -> float:
    """Edit distance between unit base phonetic and emitted phonetic substring."""
    len_p = len(phon)
    len_e = len(emitted)

    # Fast paths
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
