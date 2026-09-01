"""
Ingestion utilities for loading text chunks from Bible metadata and generic JSON sources.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transcription.alignment.models import TextChunk
from transcription.alignment.normalizers import normalize_text_for_alignment


def load_bible_chunks(
    source: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    normalizer: Callable[[str], str] = normalize_text_for_alignment,
) -> Tuple[List[TextChunk], Dict[str, Dict[str, Any]]]:
    """
    Loads Bible verse metadata into TextChunks and a source lookup dictionary.

    Args:
        source: File path to JSON, or a pre-parsed dictionary / list of verse items.
        normalizer: Function to normalize raw phonetic text. Defaults to normalize_text_for_alignment.

    Returns:
        A tuple of (chunks, source_lookup) where:
            chunks: List[TextChunk] with chunk_id and normalized text.
            source_lookup: Dict[str, Dict[str, Any]] mapping chunk_id to source metadata.
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Bible metadata file not found: {source}")
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = source

    chunks: List[TextChunk] = []
    source_lookup: Dict[str, Dict[str, Any]] = {}

    if isinstance(data, dict):
        for verse_id, verse_info in data.items():
            verse_id_str = str(verse_id)
            if not isinstance(verse_info, dict):
                verse_info = {"text": str(verse_info)}
            raw_phonetic = verse_info.get(
                "phonetic", verse_info.get("raw_phonetic", verse_info.get("text", ""))
            )
            normalized = normalizer(raw_phonetic)
            chunks.append(TextChunk(chunk_id=verse_id_str, text=normalized))
            source_lookup[verse_id_str] = verse_info
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                item = {"text": str(item)}
            verse_id_str = str(
                item.get(
                    "line_id", item.get("verse_id", item.get("id", f"{idx + 1:06d}"))
                )
            )
            raw_phonetic = item.get(
                "raw_phonetic", item.get("phonetic", item.get("text", ""))
            )
            normalized = normalizer(raw_phonetic)
            chunks.append(TextChunk(chunk_id=verse_id_str, text=normalized))
            source_lookup[verse_id_str] = item
    else:
        raise ValueError(
            "Input data must be a dictionary, a list of dictionaries, or a file path."
        )

    return chunks, source_lookup


def load_generic_chunks(
    source: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    normalizer: Optional[Callable[[str], str]] = None,
) -> Tuple[List[TextChunk], Dict[str, Dict[str, Any]]]:
    """
    Loads generic text chunks into TextChunks and a source lookup dictionary.

    Args:
        source: File path to JSON, or a pre-parsed list/dict of chunk items.
        normalizer: Optional function to normalize text. If None, raw text is kept as-is.

    Returns:
        A tuple of (chunks, source_lookup) where:
            chunks: List[TextChunk] with chunk_id and text.
            source_lookup: Dict[str, Dict[str, Any]] mapping chunk_id to source item.
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Chunk list file not found: {source}")
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = source

    chunks: List[TextChunk] = []
    source_lookup: Dict[str, Dict[str, Any]] = {}

    if isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                item = {"text": str(item)}
            chunk_id = str(
                item.get(
                    "chunk_id",
                    item.get("line_id", item.get("id", f"chunk_{idx + 1:03d}")),
                )
            )
            raw_text = item.get(
                "raw_text",
                item.get("raw_phonetic", item.get("phonetic", item.get("text", ""))),
            )
            text = normalizer(raw_text) if normalizer is not None else raw_text
            chunks.append(TextChunk(chunk_id=chunk_id, text=text))
            source_lookup[chunk_id] = item
    elif isinstance(data, dict):
        for chunk_id, item in data.items():
            chunk_id_str = str(chunk_id)
            if isinstance(item, dict):
                raw_text = item.get(
                    "raw_text",
                    item.get(
                        "raw_phonetic", item.get("phonetic", item.get("text", ""))
                    ),
                )
                info = item
            else:
                raw_text = str(item)
                info = {"text": raw_text}
            text = normalizer(raw_text) if normalizer is not None else raw_text
            chunks.append(TextChunk(chunk_id=chunk_id_str, text=text))
            source_lookup[chunk_id_str] = info
    else:
        raise ValueError(
            "Input data must be a list of dictionaries, a dictionary, or a file path."
        )

    return chunks, source_lookup
