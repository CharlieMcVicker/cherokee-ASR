# -*- coding: utf-8 -*-
"""
tones.py

Tone stripping, colon/vowel length normalization, and consonant respelling for Cherokee orthography.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Set, Tuple

VOWELS: Set[str] = set("aeiouvAEIOUV")

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


def respell_consonants(s: str) -> str:
    """
    Rewrite rules for Cherokee consonant respelling and aspiration marking.
    Order matters: t->th before d->t, k->kh before g->k.
    Exception: ts stays ts (not ths).
    """
    if not s:
        return ""

    # Replace 't' with 'th' only if not followed by 's'
    s = re.sub(r"t(?!s)", "th", s)

    rules = [
        ("d", "t"),
        ("k", "kh"),
        ("g", "k"),
        ("j", "ts"),
        ("ch", "tsh"),
        ("hn", "nh"),
        ("hl", "lh"),
        ("hy", "yh"),
        ("hw", "wh"),
        ("?", "'"),
        ("’", "'"),
    ]
    for old, new in rules:
        s = s.replace(old, new)

    s = re.sub(r"sl(?=[aeiouv])", "slh", s)
    s = re.sub(r"([^ht])s", r"\1hs", s)

    return s


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

    # Split by spaces to preserve word structure
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

                if seq_str:
                    # Look up sequence in replacement dictionary
                    if seq_str in TONE_DICT:
                        replacement = TONE_DICT[seq_str]
                    else:
                        # Fallback for unexpected sequences
                        replacement = seq_str
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
    "respell_consonants",
    "strip_tones_and_colons",
    "replace_tones",
    "remove_tones_and_double_vowels",
    "TONE_DICT",
    "DROPPED_MARKS",
    "UNMARKED_VAL",
    "VOWELS",
]
