"""
Ingestion utilities for loading text chunks from Bible metadata and generic JSON sources.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transcription.alignment.models import TextChunk
from transcription.alignment.normalizers import (
    normalize_phonetics_for_alignment,
    normalize_syllabary_for_alignment,
    normalize_text_for_alignment,
)


def load_bible_chunks(
    source: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]],
    normalizer: Callable[[str], str] = normalize_syllabary_for_alignment,
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
    if isinstance(source, (str, Path)):
        if not os.path.exists(str(source)):
            raise FileNotFoundError(f"Bible metadata file not found: {source}")
        with open(str(source), "r", encoding="utf-8") as f:
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
    source: Union[str, Path, List[Dict[str, Any]], Dict[str, Any]],
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
    if isinstance(source, (str, Path)):
        if not os.path.exists(str(source)):
            raise FileNotFoundError(f"Chunk list file not found: {source}")
        with open(str(source), "r", encoding="utf-8") as f:
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


def prepare_alignment_input(
    bible_metadata: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    chunk_list: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Tuple[
    List[TextChunk],
    Dict[str, Dict[str, Any]],
    Callable[[str], str],
    Callable[[str], str],
]:
    """
    Ingests alignment input from either Bible verse metadata or generic chunk list sources,
    resolving the appropriate representation-aware text normalizers for chunks and emissions.

    Args:
        bible_metadata: Path to Bible metadata JSON, or dictionary/list of Bible verse metadata items.
        chunk_list: Path to generic chunk list JSON, or list/dictionary of chunk items.

    Returns:
        A tuple of (chunks, source_lookup, chunk_normalizer, emissions_normalizer):
            chunks: List[TextChunk] loaded and normalized for alignment.
            source_lookup: Dict[str, Dict[str, Any]] mapping chunk IDs to raw metadata dictionaries.
            chunk_normalizer: Callable[[str], str] normalizer strategy for ground-truth chunks.
            emissions_normalizer: Callable[[str], str] normalizer strategy for ASR token emissions.

    Raises:
        ValueError: If neither or both input sources are provided.
    """
    if bible_metadata is not None and chunk_list is not None:
        raise ValueError("Cannot provide both bible_metadata and chunk_list.")

    if bible_metadata is not None:
        chunks, source_lookup = load_bible_chunks(
            bible_metadata, normalizer=normalize_syllabary_for_alignment
        )
        return (
            chunks,
            source_lookup,
            normalize_syllabary_for_alignment,
            normalize_syllabary_for_alignment,
        )

    if chunk_list is not None:
        chunks, source_lookup = load_generic_chunks(
            chunk_list, normalizer=normalize_phonetics_for_alignment
        )
        return (
            chunks,
            source_lookup,
            normalize_phonetics_for_alignment,
            normalize_phonetics_for_alignment,
        )

    raise ValueError("Either bible_metadata or chunk_list must be provided.")


__all__ = [
    "load_bible_chunks",
    "load_generic_chunks",
    "prepare_alignment_input",
]
