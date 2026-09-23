# -*- coding: utf-8 -*-
"""
ingestion.py

Bible chapter and verse ingestion utilities supporting JSON and TSV chapter transcripts.
Parses scripture verse metadata, extracts Syllabary, phonetic Latin, and English text,
and projects into normalized TextChunk domain models and verse lookup dictionaries.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transcription.cherokee.orthography import (
    Orthography,
    convert_orthography,
)
from transcription.core.alignment.models import TextChunk


def default_scripture_phonetic_normalizer(text: str) -> str:
    """
    Default normalizer for raw scripture phonetics (e.g. hyphenated DG in Bible JSON).
    Converts from DG transliteration to canonical TTH phonetics.
    """
    if not text:
        return ""
    # Strip hyphens if present in phonetic syllabification (e.g. A-da-le-ni-s-gv -> Adalenisgv)
    cleaned = text.replace("-", "")
    return convert_orthography(
        cleaned,
        source=Orthography.DG,
        target=Orthography.TTH,
        contextual_preaspiration=True,
    )


def load_chapter_transcript(
    transcript_path: Union[str, Path],
) -> Dict[str, Dict[str, str]]:
    """
    Load a single scripture chapter transcript from a JSON or TSV file.

    Returns a mapping of verse_id -> verse data dict containing keys like
    'cherokee' (syllabary), 'phonetic' (Latin DG), 'english', etc.

    Args:
        transcript_path: Path to .json or .tsv chapter transcript file.

    Returns:
        Dict mapping verse_id to verse metadata dict.

    Raises:
        FileNotFoundError: If the transcript file does not exist.
        ValueError: If file format is unsupported or cannot be parsed.
    """
    path = Path(transcript_path)
    if not path.is_file():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict):
            return {
                str(k): (v if isinstance(v, dict) else {"text": str(v)})
                for k, v in raw_data.items()
            }
        elif isinstance(raw_data, list):
            result: Dict[str, Dict[str, str]] = {}
            for idx, item in enumerate(raw_data):
                if isinstance(item, dict):
                    v_id = str(
                        item.get(
                            "verse_id",
                            item.get("line_id", item.get("id", f"{idx + 1:06d}")),
                        )
                    )
                    result[v_id] = item
                else:
                    result[f"{idx + 1:06d}"] = {"text": str(item)}
            return result
        else:
            raise ValueError(f"Invalid JSON transcript structure in {path}")

    elif suffix in (".tsv", ".txt", ".csv"):
        delimiter = "\t" if suffix in (".tsv", ".txt") else ","
        result = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            # If header is present and contains standard columns
            if reader.fieldnames and any(
                col in reader.fieldnames
                for col in ("verse_id", "id", "cherokee", "phonetic", "text")
            ):
                for idx, row in enumerate(reader):
                    v_id = str(
                        row.get(
                            "verse_id",
                            row.get("id", row.get("line_id", f"{idx + 1:06d}")),
                        )
                    )
                    result[v_id] = dict(row)
            else:
                # Fallback: line-by-line reading assuming tab or comma separated columns:
                # verse_id \t cherokee \t phonetic \t english
                f.seek(0)
                raw_reader = csv.reader(f, delimiter=delimiter)
                for idx, row in enumerate(raw_reader):
                    if not row:
                        continue
                    if len(row) == 1:
                        result[f"{idx + 1:06d}"] = {"text": row[0]}
                    elif len(row) == 2:
                        result[str(row[0])] = {"cherokee": row[1]}
                    elif len(row) == 3:
                        result[str(row[0])] = {
                            "cherokee": row[1],
                            "phonetic": row[2],
                        }
                    else:
                        result[str(row[0])] = {
                            "cherokee": row[1],
                            "phonetic": row[2],
                            "english": row[3],
                        }
        return result

    else:
        raise ValueError(
            f"Unsupported chapter transcript format '{suffix}'. Expected .json, .tsv, or .csv"
        )


def load_bible_chunks(
    source: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]],
    normalizer: Callable[[str], str] = default_scripture_phonetic_normalizer,
) -> Tuple[List[TextChunk], Dict[str, Dict[str, Any]]]:
    """
    Loads Bible verse metadata into TextChunks and a source lookup dictionary.

    Args:
        source: File path to JSON/TSV, or a pre-parsed dictionary / list of verse items.
        normalizer: Function to normalize raw phonetic text. Defaults to default_scripture_phonetic_normalizer.

    Returns:
        A tuple of (chunks, source_lookup) where:
            chunks: List[TextChunk] with chunk_id and normalized text.
            source_lookup: Dict[str, Dict[str, Any]] mapping chunk_id to source metadata.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Bible metadata file not found: {source}")
        data: Union[Dict[str, Any], List[Dict[str, Any]]] = load_chapter_transcript(
            path
        )
    else:
        data = source

    chunks: List[TextChunk] = []
    source_lookup: Dict[str, Dict[str, Any]] = {}

    if isinstance(data, dict):
        for verse_id, verse_info in data.items():
            verse_id_str = str(verse_id)
            if not isinstance(verse_info, dict):
                verse_info = {"text": str(verse_info)}
            raw_text = verse_info.get(
                "phonetic",
                verse_info.get(
                    "raw_phonetic",
                    verse_info.get(
                        "reference_sentence",
                        verse_info.get(
                            "cherokee",
                            verse_info.get(
                                "syllabary",
                                verse_info.get("text", ""),
                            ),
                        ),
                    ),
                ),
            )
            normalized = normalizer(raw_text)
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
            raw_text = item.get(
                "phonetic",
                item.get(
                    "raw_phonetic",
                    item.get(
                        "reference_sentence",
                        item.get(
                            "cherokee",
                            item.get(
                                "syllabary",
                                item.get("text", ""),
                            ),
                        ),
                    ),
                ),
            )
            normalized = normalizer(raw_text)
            chunks.append(TextChunk(chunk_id=verse_id_str, text=normalized))
            source_lookup[verse_id_str] = item
    else:
        raise ValueError(
            "Input data must be a dictionary, a list of dictionaries, or a file path."
        )

    return chunks, source_lookup


__all__ = [
    "load_chapter_transcript",
    "load_bible_chunks",
    "default_scripture_phonetic_normalizer",
]
