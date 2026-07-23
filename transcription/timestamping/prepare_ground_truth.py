# -*- coding: utf-8 -*-
"""
prepare_ground_truth.py

Module to ingest raw Bible metadata (e.g. mark_01_metadata.json) and normalize
the text for CER/edit-distance matching against ASR CTC emissions.
"""

import json
import re
import os
from typing import List, Dict, Any
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


def parse_bible_metadata(metadata_path: str) -> List[Dict[str, Any]]:
    """
    Parses mark_01_metadata.json into a list of normalized verse records.

    Returns:
        List of dicts:
        [
            {
                "line_id": "020101",
                "cherokee_syllabary": "...",
                "raw_phonetic": "...",
                "normalized_text": "...",
                "english": "..."
            }, ...
        ]
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    verses = []
    for verse_id, verse_info in data.items():
        raw_phonetic = verse_info.get("phonetic", "")
        cherokee = verse_info.get("cherokee", "")
        english = verse_info.get("english", "")

        normalized = normalize_text_for_alignment(raw_phonetic)

        verses.append(
            {
                "line_id": verse_id,
                "cherokee_syllabary": cherokee,
                "raw_phonetic": raw_phonetic,
                "normalized_text": normalized,
                "english": english,
            }
        )

    return verses
