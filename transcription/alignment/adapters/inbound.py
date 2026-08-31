"""
Inbound adapters converting external inputs (Bible metadata, generic chunk JSONs) into domain TextChunk entities.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

from transcription.alignment.domain.models import (
    AlignedChunk,
    AlignmentOutput,
    TextChunk,
)
from transcription.alignment.ports.protocols import PhoneticPreprocessor
from transcription.alignment.strategies.preprocessors import (
    CherokeePhoneticPreprocessor,
)


class BibleMetadataVerseAdapter:
    """
    Adapter for Bible chapter metadata dictionaries / JSON files.
    Maps chapter dictionaries (keyed by verse_id or list of verse objects) to TextChunks.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        preprocessor: Optional[PhoneticPreprocessor] = None,
    ):
        self.path = path
        self.preprocessor = preprocessor or CherokeePhoneticPreprocessor()

    def load_chunks(self) -> List[TextChunk]:
        """
        Loads TextChunk domain entities from self.path.
        """
        if not self.path:
            raise ValueError("No path provided to BibleMetadataVerseAdapter.")
        return self.load_chunks_from_file(self.path)

    def load_chunks_from_file(self, path: str) -> List[TextChunk]:
        """
        Loads Bible verse metadata from a JSON file and maps to TextChunks.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Bible metadata file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self.load_chunks_from_dict(data)

    def load_chunks_from_dict(
        self, data: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> List[TextChunk]:
        """
        Maps dictionary or list of verse dictionaries into TextChunk domain entities.
        """
        chunks: List[TextChunk] = []

        if isinstance(data, dict):
            for verse_id, verse_info in data.items():
                raw_phonetic = verse_info.get(
                    "phonetic", verse_info.get("raw_phonetic", "")
                )
                syllabary = verse_info.get(
                    "cherokee", verse_info.get("cherokee_syllabary", "")
                )
                english = verse_info.get("english", "")
                normalized = verse_info.get("normalized_text", "")
                if not normalized:
                    normalized = self.preprocessor.normalize(raw_phonetic)

                metadata = {
                    "line_id": str(verse_id),
                    "verse_id": str(verse_id),
                    "english": english,
                    "raw_phonetic": raw_phonetic,
                }
                # Preserve any extra metadata fields
                for k, v in verse_info.items():
                    if k not in [
                        "phonetic",
                        "raw_phonetic",
                        "cherokee",
                        "cherokee_syllabary",
                        "english",
                        "normalized_text",
                    ]:
                        metadata[k] = v

                chunks.append(
                    TextChunk(
                        chunk_id=str(verse_id),
                        raw_text=raw_phonetic,
                        normalized_text=normalized,
                        syllabary_text=syllabary if syllabary else None,
                        metadata=metadata,
                    )
                )
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                verse_id = str(
                    item.get("line_id", item.get("verse_id", f"{idx + 1:06d}"))
                )
                raw_phonetic = item.get(
                    "raw_phonetic", item.get("phonetic", item.get("text", ""))
                )
                syllabary = item.get("cherokee_syllabary", item.get("cherokee", ""))
                english = item.get("english", "")
                normalized = item.get("normalized_text", "")
                if not normalized:
                    normalized = self.preprocessor.normalize(raw_phonetic)

                metadata = {
                    "line_id": verse_id,
                    "verse_id": verse_id,
                    "english": english,
                    "raw_phonetic": raw_phonetic,
                }
                for k, v in item.items():
                    if k not in [
                        "line_id",
                        "verse_id",
                        "raw_phonetic",
                        "phonetic",
                        "text",
                        "cherokee_syllabary",
                        "cherokee",
                        "english",
                        "normalized_text",
                    ]:
                        metadata[k] = v

                chunks.append(
                    TextChunk(
                        chunk_id=verse_id,
                        raw_text=raw_phonetic,
                        normalized_text=normalized,
                        syllabary_text=syllabary if syllabary else None,
                        metadata=metadata,
                    )
                )
        else:
            raise ValueError("Input data must be a dictionary or list of dictionaries.")

        return chunks


class GenericChunkListAdapter:
    """
    Adapter for arbitrary list of text segments / chunks from JSON files or in-memory lists.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        preprocessor: Optional[PhoneticPreprocessor] = None,
    ):
        self.path = path
        self.preprocessor = preprocessor or CherokeePhoneticPreprocessor()

    def load_chunks(self) -> List[TextChunk]:
        """
        Loads TextChunk domain entities from self.path.
        """
        if not self.path:
            raise ValueError("No path provided to GenericChunkListAdapter.")
        return self.load_chunks_from_file(self.path)

    def load_chunks_from_file(self, path: str) -> List[TextChunk]:
        """
        Loads generic chunk list from JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chunk list file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Chunk list JSON must contain a top-level list.")

        return self.load_chunks_from_list(data)

    def load_chunks_from_list(self, items: List[Dict[str, Any]]) -> List[TextChunk]:
        """
        Maps a list of dictionary items to TextChunk entities.
        """
        chunks: List[TextChunk] = []

        for idx, item in enumerate(items):
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
            syllabary = item.get(
                "syllabary_text",
                item.get("cherokee_syllabary", item.get("cherokee", None)),
            )
            normalized = item.get("normalized_text", "")
            if not normalized:
                normalized = self.preprocessor.normalize(raw_text)

            metadata = dict(item.get("metadata", {}))
            for k, v in item.items():
                if k not in [
                    "chunk_id",
                    "line_id",
                    "id",
                    "raw_text",
                    "raw_phonetic",
                    "phonetic",
                    "text",
                    "syllabary_text",
                    "cherokee_syllabary",
                    "cherokee",
                    "normalized_text",
                    "metadata",
                ]:
                    metadata[k] = v

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    raw_text=raw_text,
                    normalized_text=normalized,
                    syllabary_text=syllabary if syllabary else None,
                    metadata=metadata,
                )
            )

        return chunks
