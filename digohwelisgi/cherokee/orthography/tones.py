# -*- coding: utf-8 -*-
"""
tones.py

Tone stripping, colon/vowel length normalization, and consonant respelling for Cherokee orthography.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

VOWELS: Set[str] = set("aeiouvAEIOUV")
GLOTTAL_STOP_CHARS_REGEX = r"[’‘ʼʻ`ʔ‚\?']"
CONSONANTS_CHARS = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"

# Dropped marks (occurred only once in the dataset)
DROPPED_MARKS = ["\u0302\u003a\u0301", "\u003a\u003a", "\u0307"]  # ̂:́  # ::  # ̇

# Tone replacement dictionary
TONE_DICT = {
    # Colons first
    "\u0300\u003a": "21",  # ̀: (Grave with colon)
    "\u030b\u003a": "34",  # ̋: (Double acute with colon)
    "\u0302\u003a": "32",  # ̂: (Circumflex with colon)
    "\u0301\u003a": "33",  # ́: (Acute with colon)
    "\u030c\u003a": "23",  # ̌: (Caron with colon)
    "\u003a": "22",  # : (Colon alone)
    # Without colons (mapped to same numbers where specified)
    "\u0300": "21",  # ̀ (Grave without colon)
    "\u030b": "34",  # ̋ (Double acute without colon)
    "\u0302": "32",  # ̂ (Circumflex without colon)
    "\u0301": "3",  # ́ (Acute without colon)
    "\u030c": "23",  # ̌ (Caron without colon)
}

# Non-word-final vowel with no mark gets '2'
UNMARKED_VAL = "2"

# Mapping from numeric tone strings to space-separated moraic tone tokens
TONE_TO_MORA_MAP: Dict[str, str] = {
    "2": "2_1",
    "3": "3_1",
    "1": "2_1 1_2",
    "4": "3_1 4_2",
    "22": "2_1 2_2",
    "33": "3_1 3_2",
    "21": "2_1 1_2",
    "23": "2_1 3_2",
    "32": "3_1 2_2",
    "34": "3_1 4_2",
}

# Inverse mapping from moraic token tuples to canonical numeric tone strings
MORA_TO_TONE_MAP: Dict[Tuple[str, ...], str] = {
    ("2_1",): "2",
    ("3_1",): "3",
    ("1_1",): "1",
    ("4_1",): "4",
    ("2_1", "2_2"): "22",
    ("3_1", "3_2"): "33",
    ("2_1", "1_2"): "21",
    ("2_1", "3_2"): "23",
    ("3_1", "2_2"): "32",
    ("3_1", "4_2"): "34",
}

# Standard 10-token attested moraic tone vocabulary
MORAIC_TONE_VOCAB: Dict[str, int] = {
    "|": 0,
    "C": 1,
    "2_1": 2,
    "3_1": 3,
    "1_2": 4,
    "2_2": 5,
    "3_2": 6,
    "4_2": 7,
    "[UNK]": 8,
    "[PAD]": 9,
}

# Standard 19-token segmental CV vocabulary
SEGMENTAL_CV_VOCAB: Dict[str, int] = {
    "|": 0,
    "'": 1,
    "a": 2,
    "e": 3,
    "h": 4,
    "i": 5,
    "k": 6,
    "l": 7,
    "m": 8,
    "n": 9,
    "o": 10,
    "s": 11,
    "t": 12,
    "u": 13,
    "v": 14,
    "w": 15,
    "y": 16,
    "[UNK]": 17,
    "[PAD]": 18,
}


def standardize_glottal_stops(text: str) -> str:
    """
    Unifies all glottal stop character variants (', ’, ʼ, ʔ, ‚, ?, ‘, ʻ, `)
    to canonical ASCII apostrophe (U+0027) and strips phonotactically illegal
    post-consonantal glottal stops (*C') and word-initial quote marks before consonants (*'C).
    """
    if not text:
        return ""

    s = re.sub(GLOTTAL_STOP_CHARS_REGEX, "'", text)
    s = re.sub(r"'+", "'", s)
    # Strip illegal post-consonantal glottal stops (*C')
    s = re.sub(rf"(?<=[{CONSONANTS_CHARS}])'+", "", s)
    # Strip word-initial apostrophes immediately preceding a consonant (*'C quotation artifacts)
    s = re.sub(rf"(?<!\S)'+(?=[{CONSONANTS_CHARS}])", "", s)
    return s


def resolve_vowel_hiatus(text: str) -> str:
    """
    Inserts an epenthetic glottal stop (') between adjacent vowels across
    syllable or morpheme boundaries (V1 V2 -> V1'V2, e.g. inakei -> inake'i).
    Does not split plain identical doubled vowels (aa, ee, ii, oo, uu, vv) that
    represent long vowel moras within a single syllable nucleus unless separated
    by diacritics, colons, or tone digits.
    """
    if not text:
        return ""

    pattern = re.compile(r"([aeiouvAEIOUV])([\u0300-\u036f:ː\d]*)")
    out: list[str] = []
    last_end = 0
    prev_vowel: Optional[str] = None
    prev_marks: Optional[str] = None

    for m in pattern.finditer(text):
        start, end = m.span()
        v_char = m.group(1)
        v_marks = m.group(2)

        if start > last_end:
            out.append(text[last_end:start])
            prev_vowel = None
            prev_marks = None

        if prev_vowel is not None and prev_marks is not None:
            is_plain_doubled = (
                prev_vowel.lower() == v_char.lower()
                and prev_marks == ""
                and v_marks == ""
            )
            if not is_plain_doubled:
                out.append("'")

        out.append(m.group(0))
        prev_vowel = v_char
        prev_marks = v_marks
        last_end = end

    if last_end < len(text):
        out.append(text[last_end:])

    return "".join(out)


def collapse_double_vowels(text: str) -> str:
    """Collapses doubled identical plain vowels (aa -> a, ee -> e, etc.) to base vowel quality."""
    if not text:
        return ""
    return re.sub(r"([aeiouvAEIOUV])\1+", r"\1", text)


def respell_consonants(s: str) -> str:
    """
    Rewrite rules for Cherokee consonant respelling and aspiration marking (Standard 2 TTH).
    Priority order:
      1. Glottal stop canonicalization
      2. t(?!s) -> th (before d -> t)
      3. d -> t
      4. k -> kh (before g -> k)
      5. g -> k
      6. j -> ts
      7. ch -> tsh
      8. Voiceless sonorant inversions: hn -> nh, hl -> lh, hy -> yh, hw -> wh
      9. Sibilant-lateral cluster: sl(?=[aeiouv]) -> slh
      10. Contextual pre-aspiration: ([^ht\\s])s -> \\1hs (word-initial s remains bare s-)
    """
    if not s:
        return ""

    s = standardize_glottal_stops(s)

    # Replace 't' with 'th' only if not followed by 's' or 'h'
    s = re.sub(r"t(?!s|h)", "th", s)
    s = s.replace("d", "t")

    # Replace 'k' with 'kh' if not already 'kh', before 'g' -> 'k'
    s = re.sub(r"k(?!h)", "kh", s)
    s = s.replace("g", "k")

    # Affricates and voiceless sonorants
    rules = [
        ("j", "ts"),
        ("ch", "tsh"),
        ("hn", "nh"),
        ("hl", "lh"),
        ("hy", "yh"),
        ("hw", "wh"),
    ]
    for old, new in rules:
        s = s.replace(old, new)

    s = re.sub(r"sl(?=[aeiouvAEIOUV])", "slh", s)
    # Standard 2: contextual pre-aspiration on medial/postvocalic sibilants only
    s = re.sub(r"([^ht\s])s", r"\1hs", s)
    # Ensure word-initial sibilants are strictly bare s-
    s = re.sub(r"(?<!\S)hs", "s", s)

    return standardize_glottal_stops(s)


def strip_tones_and_colons(text: str) -> str:
    """Strips numeric tone digits (0-9), combining diacritics, and vowel length colons."""
    if not text:
        return ""
    t = text.replace(":", "").replace("ː", "")
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"[\u0300-\u036f]", "", t)
    return t


def replace_tones(text: str) -> Tuple[Optional[str], bool]:
    """
    Applies Unicode NFD normalization and replaces tone/diacritic sequences
    following vowels with their mapped numerical values.

    Returns:
        (normalized_text, should_drop)
        - normalized_text: str (or None if dropped)
        - should_drop: bool (True if text contains low-frequency dropped marks)
    """
    if not isinstance(text, str):
        return "", False

    nfd_text = unicodedata.normalize("NFD", text)

    # Check for dropped marks
    for mark in DROPPED_MARKS:
        if mark in nfd_text:
            return None, True

    # Strip any stray colons not attached to a vowel or combining diacritic (e.g. 'deh:yv' -> 'dehyv')
    nfd_text = re.sub(r"(?<![aeiouvAEIOUV\u0300-\u036f])[ː:]+", "", nfd_text)

    # Resolve vowel hiatus and respell consonants
    text_normaled_all = respell_consonants(resolve_vowel_hiatus(nfd_text))
    words = text_normaled_all.split(" ")
    new_words = []

    for word in words:
        new_word = []
        i = 0
        n = len(word)
        while i < n:
            char = word[i]
            if char in VOWELS:
                # Scan for following diacritics/colons or pre-existing tone digits
                j = i + 1
                seq = []
                while j < n:
                    next_char = word[j]
                    is_combining = unicodedata.category(next_char).startswith("M")
                    if is_combining or next_char in [":", "ː"]:
                        seq.append(next_char)
                        j += 1
                    else:
                        break

                seq_str = "".join(seq)

                if seq_str:
                    # Look up sequence in replacement dictionary
                    if seq_str in TONE_DICT:
                        replacement = TONE_DICT[seq_str]
                    else:
                        # Fallback for unexpected sequences
                        replacement = seq_str
                else:
                    # Check if already followed by numeric tone digits (e.g. 21, 22, 23, 32, 33, 34, 2, 3)
                    digit_match = re.match(r"^(21|22|23|32|33|34|[1234])", word[j:])
                    if digit_match:
                        replacement = digit_match.group(1)
                        j += len(replacement)
                    else:
                        # Both non-word-final and word-final vowels with no mark get UNMARKED_VAL
                        replacement = UNMARKED_VAL

                new_word.append(char + replacement)
                i = j
            else:
                new_word.append(char)
                i += 1
        new_words.append("".join(new_word))

    return " ".join(new_words), False


def clean_asterisks_with_syllabary(
    phonetic: str, syllabary: Optional[str] = None
) -> str:
    """
    Removes formatting asterisks (*) from Feeling dictionary phonetic transcriptions.
    Uses syllabary word count when available to disambiguate missing word-boundary
    spaces (*word*nextword) vs intra-word stem bolding (pre*stem*suffix).
    """
    if not phonetic or "*" not in phonetic:
        return phonetic or ""

    # Fix cases where '*' was placed between a vowel and its length colon (e.g. ì*:sgv)
    s = re.sub(r"\*+:", ":", phonetic)

    if syllabary:
        syl_clean = re.sub(r"[^\w\s]", "", syllabary.replace("*", ""))
        syl_word_count = len(syl_clean.split())
        strip_words = s.replace("*", "").split()
        space_words = s.replace("*", " ").split()
        if len(strip_words) == syl_word_count:
            return " ".join(strip_words)
        if len(space_words) == syl_word_count:
            return " ".join(space_words)

    # Heuristic fallback when syllabary is unavailable:
    # If *...* wraps a word (>=4 chars) immediately followed/preceded by a letter (not ' or :), pad with space
    s = re.sub(
        r"\*([^*\s]{4,})\*(?=[^\s\*\'’ʼ:;\.,\?!])",
        r" \1 ",
        s,
    )
    return " ".join(s.replace("*", "").split())


def _normalize_already_tth_tones(text: str) -> str:
    """
    Normalizes a string that is already in TTH consonants (with optional colons or tone digits)
    so every vowel is followed by its canonical numeric tone without re-mutating t->th or k->kh.
    """
    nfd_text = unicodedata.normalize("NFD", text)
    s = standardize_glottal_stops(resolve_vowel_hiatus(nfd_text))
    # Enforce contextual pre-aspiration and bare word-initial s-
    s = re.sub(r"([^ht\s])s", r"\1hs", s)
    s = re.sub(r"(?<!\S)hs", "s", s)

    words = s.split()
    out_words: List[str] = []
    for word in words:
        new_word: List[str] = []
        i = 0
        n = len(word)
        while i < n:
            ch = word[i]
            if ch in VOWELS:
                j = i + 1
                seq: List[str] = []
                while j < n:
                    nxt = word[j]
                    if unicodedata.category(nxt).startswith("M") or nxt in [":", "ː"]:
                        seq.append(nxt)
                        j += 1
                    else:
                        break
                seq_str = "".join(seq)
                if seq_str:
                    replacement = TONE_DICT.get(seq_str, UNMARKED_VAL)
                else:
                    digit_match = re.match(r"^(21|22|23|32|33|34|[1234])", word[j:])
                    if digit_match:
                        replacement = digit_match.group(1)
                        j += len(replacement)
                    else:
                        replacement = UNMARKED_VAL
                new_word.append(ch.lower() + replacement)
                i = j
            else:
                if ch not in [":", "ː"] and not unicodedata.category(ch).startswith(
                    "M"
                ):
                    new_word.append(ch.lower())
                i += 1
        out_words.append("".join(new_word))
    return " ".join(out_words)


def to_canonical_toned_tth(
    raw_phonetic: str, syllabary: Optional[str] = None
) -> Tuple[Optional[str], bool]:
    """
    Converts raw Feeling (1975) phonetic text (or toned Cherokee text) into canonical
    Standard 2 TTH with inline numeric tones (e.g. 'na3 a2ni22tshu34tsa2 skwa21lhe34hsti2').

    Returns:
        (canonical_toned_tth, should_drop)
    """
    if not isinstance(raw_phonetic, str) or not raw_phonetic.strip():
        return "", False

    nfd_check = unicodedata.normalize("NFD", raw_phonetic)
    for mark in DROPPED_MARKS:
        if mark in nfd_check:
            return None, True

    text = clean_asterisks_with_syllabary(raw_phonetic, syllabary)

    # Drop word-final ';' and replace word-medial ';' with ':'
    words = text.split()
    processed_words: List[str] = []
    for w in words:
        stripped = w.rstrip(";")
        processed_words.append(stripped.replace(";", ":"))
    text = " ".join(processed_words)

    # Strip formatting punctuation (preserving glottal stop variants for standardize_glottal_stops)
    for p in [
        "[",
        "]",
        '"',
        "(",
        ")",
        ".",
        "\u0f7b",
        "_",
        "|",
        "》",
        "!",
        "/",
        ",",
        "-",
        "<",
        "…",
        ">",
    ]:
        text = text.replace(p, " ")
    # Strip word-final low-9 comma or closing quote marks at word boundaries
    text = re.sub(r"[‚’ʼ']+(?=\s|$)", "", text)
    text = " ".join(text.split()).lower()
    if not text:
        return "", False

    norm_text, should_drop = replace_tones(text)
    if should_drop or norm_text is None:
        return None, True

    # Clean any stray colons that were misplaced on consonants
    norm_text = norm_text.replace(":", "").replace("ː", "")
    # Re-ensure vowel hiatus resolution if stray colons or punctuation removal exposed adjacent toned vowels
    norm_text = resolve_vowel_hiatus(norm_text)
    norm_text = standardize_glottal_stops(norm_text)
    norm_text = " ".join(norm_text.split()).lower()
    return norm_text, False


def _needs_dg_to_tth_conversion(text: str) -> bool:
    """
    Detects whether input text is raw Feeling/DG (containing combining diacritics,
    asterisks, semicolons, uppercase, or DG-specific consonants d, g, j, ch, hn, hl, hw, hy)
    versus already-canonical TTH.
    """
    nfd = unicodedata.normalize("NFD", text)
    if any(unicodedata.category(c).startswith("M") for c in nfd):
        return True
    if any(ch in nfd for ch in ("*", ";", "ʔ", "ʼ", "‚", "’")):
        return True
    low = nfd.lower()
    if re.search(r"[dgj]|ch|hn|hl|hw|hy", low):
        return True
    return False


def to_moraic_tone_tokens(
    raw_phonetic: str, syllabary: Optional[str] = None
) -> Tuple[str, str]:
    """
    Factors phonetic Cherokee with tones (raw Feeling diacritics or canonical numeric-tone TTH)
    into dual segmental streams:
      - cv_text: clean segmental stream with single phonemic vowels (a, e, i, o, u, v),
                 zero colons, and zero numeric tones.
      - tone_tokens: space-separated sequence of 'C' (contiguous consonant/glottal cluster),
                     moraic tone tokens ('2_1', '3_1', '1_2', '2_2', '3_2', '4_2'),
                     and word delimiters '|'. Omitted leading 'C' when a word starts with a vowel.
    """
    if not raw_phonetic or not raw_phonetic.strip():
        return "", ""

    if _needs_dg_to_tth_conversion(raw_phonetic):
        canon, should_drop = to_canonical_toned_tth(raw_phonetic, syllabary=syllabary)
        if should_drop or not canon:
            return "", ""
    else:
        canon = _normalize_already_tth_tones(raw_phonetic)

    words = canon.split()
    cv_words: List[str] = []
    tone_word_seqs: List[str] = []

    token_pattern = re.compile(
        r"([aeiouv])(21|22|23|32|33|34|[1234])?|([^aeiouv\d\s]+)"
    )

    for word in words:
        cv_chars: List[str] = []
        word_tones: List[str] = []
        for m in token_pattern.finditer(word):
            v_char, tone_str, c_cluster = m.group(1), m.group(2), m.group(3)
            if v_char is not None:
                cv_chars.append(v_char)
                t_key = tone_str if tone_str else UNMARKED_VAL
                mora_toks = TONE_TO_MORA_MAP.get(t_key, "2_1")
                word_tones.append(mora_toks)
            elif c_cluster is not None:
                cv_chars.append(c_cluster)
                word_tones.append("C")
        if cv_chars:
            cv_words.append("".join(cv_chars))
            tone_word_seqs.append(" ".join(word_tones))

    return " ".join(cv_words), " | ".join(tone_word_seqs)


def _parse_tone_tokens_to_segments(tone_tokens: str) -> List[Tuple[str, str]]:
    """
    Parses a space-separated moraic tone token sequence into structural segments:
      - ("|", "|") for word boundaries
      - ("C", "C") for consonant clusters
      - ("V", tone_digits) for 1-mora or 2-mora vowel tone nuclei (e.g. "2", "3", "21", "34")
    """
    raw_toks = [t for t in tone_tokens.strip().split() if t not in ("[PAD]", "[UNK]")]
    segments: List[Tuple[str, str]] = []
    i = 0
    n = len(raw_toks)
    while i < n:
        tok = raw_toks[i]
        if tok == "|":
            segments.append(("|", "|"))
            i += 1
        elif tok == "C":
            segments.append(("C", "C"))
            i += 1
        elif tok.endswith("_1"):
            d1 = tok.split("_")[0]
            if i + 1 < n and raw_toks[i + 1].endswith("_2"):
                d2 = raw_toks[i + 1].split("_")[0]
                mora_pair = (tok, raw_toks[i + 1])
                tone_digits = MORA_TO_TONE_MAP.get(mora_pair, d1 + d2)
                segments.append(("V", tone_digits))
                i += 2
            else:
                tone_digits = MORA_TO_TONE_MAP.get((tok,), d1)
                segments.append(("V", tone_digits))
                i += 1
        elif tok.endswith("_2"):
            # Orphaned Mora 2 token in noisy inference: reconstruct plausible 2-mora contour
            d2 = tok.split("_")[0]
            default_d1 = "3" if d2 == "4" else "2"
            segments.append(("V", default_d1 + d2))
            i += 1
        else:
            i += 1
    return segments


def _parse_cv_text_to_segments(cv_text: str) -> List[Tuple[str, str]]:
    """
    Parses cv_text into structural segments:
      - ("|", " ") for word boundaries
      - ("C", c_cluster) for contiguous consonant/glottal clusters
      - ("V", v_char) for single vowels
    """
    words = collapse_double_vowels(cv_text.strip().lower()).split()
    segments: List[Tuple[str, str]] = []
    seg_pattern = re.compile(r"([aeiouv])|([^aeiouv\s]+)")
    for w_idx, word in enumerate(words):
        if w_idx > 0:
            segments.append(("|", " "))
        for m in seg_pattern.finditer(word):
            v_char, c_cluster = m.group(1), m.group(2)
            if v_char is not None:
                segments.append(("V", v_char))
            elif c_cluster is not None:
                segments.append(("C", c_cluster))
    return segments


def reconcile_dual_stream(cv_text: str, tone_tokens: str, mode: str = "dp") -> str:
    """
    Reconciles segmental CV text (Head A) and moraic tone tokens (Head B) into
    canonical numeric-tone TTH (e.g. 'na3 a2ni22tshu34tsa2').

    Args:
        cv_text: Segmental CV string with base vowels.
        tone_tokens: Space-separated moraic tone token sequence ('C', '2_1', ..., '|').
        mode: 'fast' (deterministic slot projection) or 'dp' (Needleman-Wunsch sequence alignment
              anchored on 'C' and '|' boundaries).

    Returns:
        Reconciled canonical numeric-tone TTH string.
    """
    if not cv_text or not cv_text.strip():
        return ""

    cv_segments = _parse_cv_text_to_segments(cv_text)
    if not tone_tokens or not tone_tokens.strip():
        return "".join(
            val + UNMARKED_VAL if stype == "V" else val for stype, val in cv_segments
        )

    tone_segments = _parse_tone_tokens_to_segments(tone_tokens)

    if mode == "fast":
        # Group by word boundaries '|' when word counts match; otherwise slot-match globally
        cv_words = collapse_double_vowels(cv_text.strip().lower()).split()
        tone_word_chunks = [chunk.strip() for chunk in tone_tokens.split("|")]
        if len(cv_words) == len(tone_word_chunks):
            reconciled_words: List[str] = []
            for w_cv, w_tone_str in zip(cv_words, tone_word_chunks):
                w_tone_segs = [
                    val
                    for stype, val in _parse_tone_tokens_to_segments(w_tone_str)
                    if stype == "V"
                ]
                v_idx = 0
                chars_out: List[str] = []
                for ch in w_cv:
                    if ch in VOWELS:
                        t_val = (
                            w_tone_segs[v_idx]
                            if v_idx < len(w_tone_segs)
                            else UNMARKED_VAL
                        )
                        chars_out.append(ch + t_val)
                        v_idx += 1
                    else:
                        chars_out.append(ch)
                reconciled_words.append("".join(chars_out))
            return " ".join(reconciled_words)

        # Global vowel slot fallback
        v_tones = [val for stype, val in tone_segments if stype == "V"]
        v_idx = 0
        out_parts: List[str] = []
        for stype, val in cv_segments:
            if stype == "V":
                t_val = v_tones[v_idx] if v_idx < len(v_tones) else UNMARKED_VAL
                out_parts.append(val + t_val)
                v_idx += 1
            else:
                out_parts.append(val)
        return "".join(out_parts)

    if mode != "dp":
        raise ValueError(
            f"Unsupported reconciliation mode: {mode!r}. Expected 'dp' or 'fast'."
        )

    # Mode 'dp': Needleman-Wunsch sequence alignment between cv_segments and tone_segments
    n = len(cv_segments)
    m = len(tone_segments)

    gap_cost = 1.0
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    ptr = [
        [0] * (m + 1) for _ in range(n + 1)
    ]  # 0: diag, 1: up (delete tone), 2: left (insert tone)

    for i in range(1, n + 1):
        dp[i][0] = i * gap_cost
        ptr[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j * gap_cost
        ptr[0][j] = 2

    for i in range(1, n + 1):
        cv_type = cv_segments[i - 1][0]
        for j in range(1, m + 1):
            t_type = tone_segments[j - 1][0]
            if cv_type == t_type:
                sub_cost = 0.0
            elif cv_type == "|" or t_type == "|":
                sub_cost = 5.0
            else:
                sub_cost = 3.0

            cost_diag = dp[i - 1][j - 1] + sub_cost
            cost_up = dp[i - 1][j] + gap_cost
            cost_left = dp[i][j - 1] + gap_cost

            if cost_diag <= cost_up and cost_diag <= cost_left:
                dp[i][j] = cost_diag
                ptr[i][j] = 0
            elif cost_up <= cost_left:
                dp[i][j] = cost_up
                ptr[i][j] = 1
            else:
                dp[i][j] = cost_left
                ptr[i][j] = 2

    # Backtrack to find assigned tone for each cv_segment
    assigned_tones: List[Optional[str]] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ptr[i][j] == 0:
            if cv_segments[i - 1][0] == "V" and tone_segments[j - 1][0] == "V":
                assigned_tones[i - 1] = tone_segments[j - 1][1]
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or ptr[i][j] == 1):
            i -= 1
        else:
            j -= 1

    result: List[str] = []
    for idx, (stype, val) in enumerate(cv_segments):
        if stype == "V":
            t_digits = assigned_tones[idx] or UNMARKED_VAL
            result.append(val + t_digits)
        else:
            result.append(val)

    return "".join(result)


def moraic_tokens_to_pitch_contour(
    tone_tokens: str, step_sec: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Converts a space-separated moraic tone token sequence into a list of
    (time_sec, pitch_level) coordinate pairs for Praat and visual inspection.
    Pitch levels correspond to discrete Cherokee tone levels (1.0, 2.0, 3.0, 4.0).
    """
    if not tone_tokens or not tone_tokens.strip():
        return []

    contour: List[Tuple[float, float]] = []
    time_idx = 0
    for tok in tone_tokens.strip().split():
        if "_" in tok:
            level_str = tok.split("_")[0]
            if level_str.isdigit():
                contour.append((round(time_idx * step_sec, 4), float(int(level_str))))
                time_idx += 1
        elif tok in ("C", "|"):
            time_idx += 1
    return contour


def normalize_standard2_cv(text: str) -> str:
    """
    Normalizes an existing TTH segmental string to strict Standard 2 CV format:
      - Strips word-final quote/comma artifacts
      - Strips tones and colons
      - Standardizes glottal stops and strips post-consonantal violations (*C')
      - Resolves inter-vocalic hiatus (V1V2 -> V1'V2)
      - Enforces contextual pre-aspiration on medial /s/ and bare word-initial s-
      - Collapses doubled vowels (aa -> a)
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    s = strip_tones_and_colons(text.strip().lower())
    for p in [
        "[",
        "]",
        '"',
        "(",
        ")",
        ".",
        "\u0f7b",
        "_",
        "|",
        "》",
        "!",
        "/",
        ",",
        "-",
        "<",
        "…",
        ">",
        "*",
    ]:
        s = s.replace(p, " ")
    s = re.sub(r"[‚’ʼ']+(?=\s|$)", "", s)
    s = standardize_glottal_stops(resolve_vowel_hiatus(s))
    s = re.sub(r"([^ht\s])s", r"\1hs", s)
    s = re.sub(r"(?<!\S)hs", "s", s)
    s = collapse_double_vowels(standardize_glottal_stops(s))
    return " ".join(s.split())


def encode_cv_text(
    cv_text: str, vocab: Mapping[str, int] = SEGMENTAL_CV_VOCAB
) -> List[int]:
    """
    Encodes a Standard 2 segmental CV string into integer token IDs using vocab.json.
    Spaces are mapped to the word delimiter token '|'.
    """
    norm = " ".join(cv_text.strip().split())
    if not norm:
        return []
    unk_id = vocab.get("[UNK]", 17)
    delim_id = vocab.get("|", 0)
    ids: List[int] = []
    for ch in norm:
        if ch == " ":
            ids.append(delim_id)
        else:
            ids.append(vocab.get(ch, unk_id))
    return ids


def decode_cv_ids(
    ids: Sequence[int],
    vocab: Mapping[str, int] = SEGMENTAL_CV_VOCAB,
    collapse_repeats: bool = False,
    remove_pad: bool = True,
) -> str:
    """
    Decodes a sequence of segmental CV token IDs back into a space-separated string.
    Ignores -100, [PAD], and [UNK] tokens.
    """
    id_to_tok = {int(v): str(k) for k, v in vocab.items()}
    pad_id = vocab.get("[PAD]", 18)
    unk_id = vocab.get("[UNK]", 17)
    delim_id = vocab.get("|", 0)

    chars: List[str] = []
    prev_id: Optional[int] = None
    for raw_id in ids:
        cid = int(raw_id)
        if collapse_repeats and cid == prev_id:
            continue
        prev_id = cid
        if cid == -100 or cid == unk_id or (remove_pad and cid == pad_id):
            continue
        if cid == delim_id:
            chars.append(" ")
        else:
            tok = id_to_tok.get(cid, "")
            if tok and tok != "[UNK]" and (not remove_pad or tok != "[PAD]"):
                chars.append(tok)
    return " ".join("".join(chars).split())


def encode_tone_tokens(
    tone_tokens: Optional[str], vocab: Mapping[str, int] = MORAIC_TONE_VOCAB
) -> List[int]:
    """
    Encodes a space-separated moraic tone token sequence (e.g. 'C 2_1 | C 3_1 4_2')
    into integer token IDs using vocab_tone.json.
    Returns [-100] when tone_tokens is empty or None (e.g. CV-only Bible rows).
    """
    if not isinstance(tone_tokens, str) or not tone_tokens.strip():
        return [-100]
    toks = tone_tokens.strip().split()
    if not toks:
        return [-100]
    unk_id = vocab.get("[UNK]", 8)
    return [vocab.get(t, unk_id) for t in toks]


def decode_tone_ids(
    ids: Sequence[int],
    vocab: Mapping[str, int] = MORAIC_TONE_VOCAB,
    collapse_repeats: bool = False,
    remove_pad: bool = True,
) -> str:
    """
    Decodes a sequence of moraic tone token IDs back into a space-separated moraic token string.
    Ignores -100, [PAD], and [UNK] tokens.
    """
    id_to_tok = {int(v): str(k) for k, v in vocab.items()}
    pad_id = vocab.get("[PAD]", 9)
    unk_id = vocab.get("[UNK]", 8)

    out_toks: List[str] = []
    prev_id: Optional[int] = None
    for raw_id in ids:
        tid = int(raw_id)
        if collapse_repeats and tid == prev_id:
            continue
        prev_id = tid
        if tid == -100 or tid == unk_id or (remove_pad and tid == pad_id):
            continue
        tok = id_to_tok.get(tid, "")
        if tok and tok != "[UNK]" and (not remove_pad or tok != "[PAD]"):
            out_toks.append(tok)
    return " ".join(out_toks)


def save_dual_vocabularies(output_dir: Union[str, Path]) -> Tuple[Path, Path]:
    """
    Serializes both vocab.json (19 tokens) and vocab_tone.json (10 attested moraic tokens)
    into output_dir.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    vocab_cv_path = out_path / "vocab.json"
    vocab_tone_path = out_path / "vocab_tone.json"

    with open(vocab_cv_path, "w", encoding="utf-8") as f:
        json.dump(SEGMENTAL_CV_VOCAB, f, ensure_ascii=False, indent=2)
    with open(vocab_tone_path, "w", encoding="utf-8") as f:
        json.dump(MORAIC_TONE_VOCAB, f, ensure_ascii=False, indent=2)

    return vocab_cv_path, vocab_tone_path


def remove_tones_and_double_vowels(text: str) -> Tuple[Optional[str], bool]:
    """
    Normalizes Cherokee text:
    1. Unicode NFD normalization.
    2. Respell consonants.
    3. For vowels (aeiouvAEIOUV):
       - If followed by a colon (':' or 'ː'), double the vowel and remove the colon and all combining diacritics.
       - If not followed by a colon, keep the vowel single and remove all combining diacritics.
    """
    if not isinstance(text, str):
        return "", False

    nfd_text = unicodedata.normalize("NFD", text)

    # Check for dropped marks
    for mark in DROPPED_MARKS:
        if mark in nfd_text:
            return None, True

    # Respell consonants
    text_normaled_all = respell_consonants(nfd_text)
    words = text_normaled_all.split(" ")
    new_words = []

    for word in words:
        new_word = []
        i = 0
        n = len(word)
        while i < n:
            char = word[i]
            if char in VOWELS:
                # Scan for following diacritics/colons
                j = i + 1
                seq = []
                while j < n:
                    next_char = word[j]
                    is_combining = unicodedata.category(next_char).startswith("M")
                    if is_combining or next_char in [":", "ː"]:
                        seq.append(next_char)
                        j += 1
                    else:
                        break

                seq_str = "".join(seq)
                if ":" in seq_str or "ː" in seq_str:
                    new_word.append(char * 2)
                else:
                    new_word.append(char)
                i = j
            else:
                new_word.append(char)
                i += 1
        new_words.append("".join(new_word))

    return " ".join(new_words), False


__all__ = [
    "standardize_glottal_stops",
    "resolve_vowel_hiatus",
    "collapse_double_vowels",
    "respell_consonants",
    "strip_tones_and_colons",
    "replace_tones",
    "remove_tones_and_double_vowels",
    "clean_asterisks_with_syllabary",
    "to_canonical_toned_tth",
    "to_moraic_tone_tokens",
    "reconcile_dual_stream",
    "moraic_tokens_to_pitch_contour",
    "normalize_standard2_cv",
    "encode_cv_text",
    "decode_cv_ids",
    "encode_tone_tokens",
    "decode_tone_ids",
    "save_dual_vocabularies",
    "TONE_DICT",
    "TONE_TO_MORA_MAP",
    "MORA_TO_TONE_MAP",
    "MORAIC_TONE_VOCAB",
    "SEGMENTAL_CV_VOCAB",
    "DROPPED_MARKS",
    "UNMARKED_VAL",
    "VOWELS",
    "GLOTTAL_STOP_CHARS_REGEX",
    "CONSONANTS_CHARS",
]
