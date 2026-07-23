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


def parse_chunk_list(chunk_list_path: str) -> List[Dict[str, Any]]:
    """
    Parses a simple JSON list of chunk dictionaries into normalized segment records.

    Expected JSON format:
    [
        {
            "line_id": "seg_01", (optional)
            "raw_phonetic": "...",
            "cherokee_syllabary": "...", (optional)
            "english": "..." (optional)
        }, ...
    ]
    """
    if not os.path.exists(chunk_list_path):
        raise FileNotFoundError(f"Chunk list file not found: {chunk_list_path}")

    with open(chunk_list_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Chunk list JSON must be a top-level list of dictionaries.")

    segments = []
    for idx, item in enumerate(data):
        line_id = str(item.get("line_id", f"chunk_{idx + 1:03d}"))
        raw_phonetic = item.get("raw_phonetic", item.get("phonetic", ""))
        cherokee = item.get("cherokee_syllabary", item.get("cherokee", ""))
        english = item.get("english", "")

        normalized = item.get("normalized_text", "")
        if not normalized:
            normalized = normalize_text_for_alignment(raw_phonetic)

        segments.append(
            {
                "line_id": line_id,
                "cherokee_syllabary": cherokee,
                "raw_phonetic": raw_phonetic,
                "normalized_text": normalized,
                "english": english,
            }
        )

    return segments


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
