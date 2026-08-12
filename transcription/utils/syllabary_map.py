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


import re

# Reverse map: phonetic transliteration -> Cherokee syllabary character
PHONETIC_TO_SYLLABARY_MAP: Dict[str, str] = {
    phon: char for char, phon in CHEROKEE_SYLLABARY_MAP.items()
}


# Add common voiced/unvoiced variants
PHONETIC_TO_SYLLABARY_MAP.update(
    {
        "g": "Ꭶ",
        "ga": "Ꭶ",
        "ge": "Ꭸ",
        "gi": "Ꭹ",
        "go": "Ꭺ",
        "gu": "Ꭻ",
        "gv": "Ꭼ",
        "d": "Ꮣ",
        "da": "Ꮣ",
        "de": "Ꮥ",
        "di": "Ꮧ",
        "do": "Ꮩ",
        "du": "Ꮪ",
        "dv": "Ꮫ",
        "hna": "Ꮏ",
        "lha": "Ꮭ",
        "lhe": "Ꮮ",
        "hli": "Ꮯ",
        "lho": "Ꮰ",
        "lhu": "Ꮱ",
        "lhv": "Ꮲ",
    }
)


def cherokee_to_bad_phonetics(text: str) -> str:
    """
    Translates Cherokee syllabary into phonetic transliteration character by character.
    Preserves spaces, punctuation, and unknown non-syllabary characters.
    """
    return "".join(CHEROKEE_SYLLABARY_MAP.get(char, char) for char in text)


def phonetics_to_syllabary(text: str) -> str:
    """
    Converts Cherokee phonetic transliteration into Cherokee Syllabary characters.
    Splits words on vowel boundaries (a, e, i, o, u, v). Handles glottal stops ('), pre-aspiration 'h',
    and aspirated fallbacks (e.g. thv -> tv -> Ꮫ, hska -> s + ka -> Ꮝ + Ꭶ).
    """
    if not text:
        return ""

    clean_text = text.replace("’", "'").replace("`", "'").replace("'", "")
    words = clean_text.split(" ")
    out_words = []

    for word in words:
        if not word:
            continue
        # Split into syllable tokens: optional glottal stop/consonants + vowel OR non-vowels
        tokens = re.findall(r"[^aeiouv]*[aeiouv]|[^aeiouv]+", word.lower())
        syllabary_chars = []
        for sub_tok in tokens:
            # Strip glottals at beginning of syllable token (e.g. 'v -> v)
            if not sub_tok:
                continue

            if sub_tok in PHONETIC_TO_SYLLABARY_MAP:
                syllabary_chars.append(PHONETIC_TO_SYLLABARY_MAP[sub_tok])
            elif sub_tok.startswith("h") and sub_tok[1:] in PHONETIC_TO_SYLLABARY_MAP:
                # Pre-aspiration 'h' on consonant or vowel (e.g., hsi -> si -> Ꮟ)
                syllabary_chars.append(PHONETIC_TO_SYLLABARY_MAP[sub_tok[1:]])
            elif sub_tok.startswith("hs") or sub_tok.startswith("s"):
                # Handle s-clusters (e.g., hska -> s + ka -> Ꮝ + Ꭶ, ska -> s + ka -> Ꮝ + Ꭶ)
                remainder = sub_tok.lstrip("h")
                if remainder.startswith("s") and len(remainder) > 1:
                    syllabary_chars.append(PHONETIC_TO_SYLLABARY_MAP.get("s", "Ꮝ"))
                    rest = remainder[1:]
                    if rest in PHONETIC_TO_SYLLABARY_MAP:
                        syllabary_chars.append(PHONETIC_TO_SYLLABARY_MAP[rest])
                    elif "h" in rest:
                        fallback = rest.replace("h", "")
                        if fallback in PHONETIC_TO_SYLLABARY_MAP:
                            syllabary_chars.append(PHONETIC_TO_SYLLABARY_MAP[fallback])
                        else:
                            syllabary_chars.append(rest)
                    else:
                        syllabary_chars.append(rest)
                else:
                    syllabary_chars.append(
                        PHONETIC_TO_SYLLABARY_MAP.get(sub_tok, sub_tok)
                    )
            elif "h" in sub_tok:
                # Drop 'h' fallback (e.g. thv -> tv -> Ꮫ, khv -> kv -> Ꭼ)
                fallback = sub_tok.replace("h", "")
                if fallback in PHONETIC_TO_SYLLABARY_MAP:
                    syllabary_chars.append(PHONETIC_TO_SYLLABARY_MAP[fallback])
                else:
                    syllabary_chars.append(sub_tok)
            else:
                syllabary_chars.append(sub_tok)

        out_words.append("".join(syllabary_chars))

    return " ".join(out_words)
