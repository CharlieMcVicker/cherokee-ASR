"""
Text normalization utilities for alignment.
"""

import re
from transcription.utils.tone_normalization import respell_consonants


def normalize_text_for_alignment(text: str) -> str:
    """
    Normalizes transliterated Cherokee text for ASR alignment matching:
    1. Lowercase text and strip hyphens (e.g., A-da-le-ni-s-gv -> adalenisgv).
    2. Convert 'qu' to 'gw'.
    3. Apply consonant & aspiration respelling (t->th, d->t, k->kh, g->k, etc.).
    4. Strip /h/ sound markers.
    5. Remove punctuation and collapse extra whitespace.
    """
    if not text:
        return ""

    text = text.lower()
    # Strip hyphens
    text = text.replace("-", "")

    # Convert qu -> gw
    text = text.replace("qu", "gw")

    # Respell consonants
    text = respell_consonants(text)

    # Drop all /h/ sound markers
    text = text.replace("h", "")

    # Strip punctuation
    punctuation_regex = r"[\,\?\.\!\-\;\:\"\'\“\%\”\(\)\[\]\{\}«»…\’\‘\ʼ\ʻ\`\´\‛]"
    text = re.sub(punctuation_regex, "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
