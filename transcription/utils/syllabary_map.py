# -*- coding: utf-8 -*-
"""
syllabary_map.py

Centralized Cherokee Syllabary to phonetic transliteration mapping module.
Reflects respell_consonants rules so all phonetic representations are unified
and consistent across the codebase (e.g. Ꮏ -> nha).
"""

from typing import Dict

# Base Cherokee Syllabary character mapping (un-respelled base transliteration)
# Same base phonetic definitions as alignment_engine
_BASE_CHEROKEE_SYLLABARY_MAP: Dict[str, str] = {
    "Ꭰ": "a",
    "Ꭱ": "e",
    "Ꭲ": "i",
    "Ꭳ": "o",
    "Ꭴ": "u",
    "Ꭵ": "v",
    "Ꭶ": "ka",
    "Ꭷ": "kha",
    "Ꭸ": "ke",
    "Ꭹ": "ki",
    "Ꭺ": "ko",
    "Ꭻ": "ku",
    "Ꭼ": "kv",
    "Ꭽ": "ha",
    "Ꭾ": "he",
    "Ꭿ": "hi",
    "Ꮀ": "ho",
    "Ꮁ": "hu",
    "Ꮂ": "hv",
    "Ꮃ": "la",
    "Ꮄ": "le",
    "Ꮅ": "li",
    "Ꮆ": "lo",
    "Ꮇ": "lu",
    "Ꮈ": "lv",
    "Ꮉ": "ma",
    "Ꮊ": "me",
    "Ꮋ": "mi",
    "Ꮌ": "mo",
    "Ꮍ": "mu",
    "Ꮎ": "na",
    "Ꮏ": "hna",
    "Ꮐ": "nah",
    "Ꮑ": "ne",
    "Ꮒ": "ni",
    "Ꮓ": "no",
    "Ꮔ": "nu",
    "Ꮕ": "nv",
    "Ꮖ": "kwa",
    "Ꮗ": "kwe",
    "Ꮘ": "kwi",
    "Ꮙ": "kwo",
    "Ꮚ": "kwu",
    "Ꮛ": "kwv",
    "Ꮜ": "sa",
    "Ꮝ": "s",
    "Ꮞ": "se",
    "Ꮟ": "si",
    "Ꮠ": "so",
    "Ꮡ": "su",
    "Ꮢ": "sv",
    "Ꮣ": "ta",
    "Ꮤ": "tha",
    "Ꮥ": "te",
    "Ꮦ": "the",
    "Ꮧ": "ti",
    "Ꮨ": "thi",
    "Ꮩ": "to",
    "Ꮪ": "tu",
    "Ꮫ": "tv",
    "Ꮭ": "tla",
    "Ꮬ": "tla",
    "Ꮮ": "tle",
    "Ꮯ": "tli",
    "Ꮰ": "tlo",
    "Ꮱ": "tlu",
    "Ꮲ": "tlv",
    "Ꮳ": "tsa",
    "Ꮴ": "tse",
    "Ꮵ": "tsi",
    "Ꮶ": "tso",
    "Ꮷ": "tsu",
    "Ꮸ": "tsv",
    "Ꮹ": "wa",
    "Ꮺ": "we",
    "Ꮻ": "wi",
    "Ꮼ": "wo",
    "Ꮽ": "wu",
    "Ꮾ": "wv",
    "Ꮿ": "ya",
    "Ᏸ": "ye",
    "Ᏹ": "yi",
    "Ᏺ": "yo",
    "Ᏻ": "yu",
    "Ᏼ": "yv",
}

# Authoritative centralized Cherokee Syllabary mapping
# Derived from base transliterations with respell_consonants rules (specifically Ꮏ -> nha instead of hna)
CHEROKEE_SYLLABARY_MAP: Dict[str, str] = dict(_BASE_CHEROKEE_SYLLABARY_MAP)
CHEROKEE_SYLLABARY_MAP["Ꮏ"] = "nha"


def cherokee_to_bad_phonetics(text: str) -> str:
    """
    Translates Cherokee syllabary into phonetic transliteration character by character.
    Preserves spaces, punctuation, and unknown non-syllabary characters.
    """
    return "".join(CHEROKEE_SYLLABARY_MAP.get(char, char) for char in text)
