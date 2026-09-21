# -*- coding: utf-8 -*-
"""
Unit tests for ingestion.py.
"""

import json
import os
import tempfile
import pytest

from transcription.alignment.ingestion import (
    load_bible_chunks,
    load_generic_chunks,
    prepare_alignment_input,
)
from transcription.alignment.models import TextChunk
from transcription.alignment.normalizers import (
    normalize_phonetics_for_alignment,
    normalize_syllabary_for_alignment,
)


def test_load_bible_chunks_dict():
    data = {
        "020101": {
            "phonetic": "A-da-le-ni-s-gv",
            "cherokee": "ᎠᏓᎴᏂᏍᎬ",
            "english": "The beginning",
        },
        "020102": {
            "phonetic": "yi-s-dv",
            "cherokee": "ᏱᏍᏛ",
            "english": "of the gospel",
        },
    }
    chunks, source_lookup = load_bible_chunks(data)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "020101"
    assert chunks[0].text == "atalenihskv"
    assert chunks[1].chunk_id == "020102"
    assert chunks[1].text == "yihstv"

    assert source_lookup["020101"]["english"] == "The beginning"
    assert source_lookup["020102"]["cherokee"] == "ᏱᏍᏛ"


def test_load_bible_chunks_dict_with_str_values():
    data = {
        "1": "A-da-le-ni-s-gv",
        "2": "yi-s-dv",
    }
    chunks, source_lookup = load_bible_chunks(data)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "1"
    assert chunks[0].text == "atalenihskv"
    assert source_lookup["1"]["text"] == "A-da-le-ni-s-gv"


def test_load_bible_chunks_custom_normalizer():
    data = {"1": {"raw_phonetic": "Hello World"}}
    chunks, source_lookup = load_bible_chunks(data, normalizer=lambda s: s.upper())
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "1"
    assert chunks[0].text == "HELLO WORLD"
    assert source_lookup["1"]["raw_phonetic"] == "Hello World"


def test_load_bible_chunks_list():
    data = [
        {"verse_id": "0101", "phonetic": "A-da-le-ni-s-gv", "english": "v1"},
        {"line_id": "0102", "raw_phonetic": "yi-s-dv", "english": "v2"},
        "plain text item",
    ]
    chunks, source_lookup = load_bible_chunks(data)
    assert len(chunks) == 3
    assert chunks[0].chunk_id == "0101"
    assert chunks[0].text == "atalenihskv"
    assert chunks[1].chunk_id == "0102"
    assert chunks[1].text == "yihstv"
    assert chunks[2].chunk_id == "000003"
    assert source_lookup["0101"]["english"] == "v1"
    assert source_lookup["0102"]["english"] == "v2"


def test_load_bible_chunks_file():
    data = {
        "020101": {
            "phonetic": "A-da-le-ni-s-gv",
            "cherokee": "ᎠᏓᎴᏂᏍᎬ",
            "english": "The beginning",
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "meta.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        chunks, source_lookup = load_bible_chunks(json_path)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "020101"
        assert chunks[0].text == "atalenihskv"
        assert source_lookup["020101"]["english"] == "The beginning"


def test_load_bible_chunks_errors():
    with pytest.raises(FileNotFoundError):
        load_bible_chunks("/nonexistent/path/to/meta.json")

    with pytest.raises(ValueError):
        load_bible_chunks(12345)  # type: ignore[arg-type]


def test_load_generic_chunks_list():
    items = [
        {"id": "c1", "text": "osiyo", "speaker": "spk1"},
        {"chunk_id": "c2", "raw_text": "tohiju", "speaker": "spk2"},
        "plain chunk",
    ]
    chunks, source_lookup = load_generic_chunks(items)
    assert len(chunks) == 3
    assert chunks[0].chunk_id == "c1"
    assert chunks[0].text == "osiyo"
    assert chunks[1].chunk_id == "c2"
    assert chunks[1].text == "tohiju"
    assert chunks[2].chunk_id == "chunk_003"
    assert source_lookup["c1"]["speaker"] == "spk1"
    assert source_lookup["c2"]["speaker"] == "spk2"


def test_load_generic_chunks_with_normalizer():
    items = [{"id": "c1", "raw_text": "A-da-le-ni-s-gv"}]
    chunks, source_lookup = load_generic_chunks(
        items, normalizer=lambda s: s.lower().replace("-", "")
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1"
    assert chunks[0].text == "adalenisgv"


def test_load_generic_chunks_dict():
    data = {
        "c1": {"text": "osiyo", "speaker": "spk1"},
        "c2": "tohiju",
    }
    chunks, source_lookup = load_generic_chunks(data)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "c1"
    assert chunks[0].text == "osiyo"
    assert chunks[1].chunk_id == "c2"
    assert chunks[1].text == "tohiju"


def test_load_generic_chunks_file():
    items = [{"id": "c1", "text": "osiyo"}]
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "chunks.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(items, f)

        chunks, source_lookup = load_generic_chunks(json_path)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "c1"
        assert chunks[0].text == "osiyo"


def test_load_generic_chunks_errors():
    with pytest.raises(FileNotFoundError):
        load_generic_chunks("/nonexistent/path/to/chunks.json")

    with pytest.raises(ValueError):
        load_generic_chunks(12345)  # type: ignore[arg-type]


def test_prepare_alignment_input_bible_metadata():
    bible_meta = {
        "020101": {
            "phonetic": "A-da-le-ni-s-gv",
            "cherokee": "ᎠᏓᎴᏂᏍᎬ",
        }
    }
    chunks, source_lookup, chunk_norm, emission_norm = prepare_alignment_input(
        bible_metadata=bible_meta
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "020101"
    assert chunks[0].text == "atalenihskv"
    assert "020101" in source_lookup

    assert chunk_norm is normalize_phonetics_for_alignment
    assert emission_norm is normalize_phonetics_for_alignment
    assert chunk_norm("ho-wa") == "howa"
    assert emission_norm("ho-wa") == "howa"


def test_prepare_alignment_input_chunk_list():
    chunk_list = [
        {
            "id": "c1",
            "raw_text": "A-da-le-ni-s-gv",
            "speaker": "spk1",
        }
    ]
    chunks, source_lookup, chunk_norm, emission_norm = prepare_alignment_input(
        chunk_list=chunk_list
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1"
    # Phonetics normalization preserves 'h'
    assert chunks[0].text == "atalenihskv"
    assert "c1" in source_lookup

    assert chunk_norm is normalize_phonetics_for_alignment
    assert emission_norm is normalize_phonetics_for_alignment
    assert chunk_norm("ho-wa") == "howa"
    assert emission_norm("ho-wa") == "howa"


def test_prepare_alignment_input_errors():
    # Neither provided
    with pytest.raises(
        ValueError, match="Either bible_metadata or chunk_list must be provided"
    ):
        prepare_alignment_input()

    # Both provided
    with pytest.raises(
        ValueError, match="Cannot provide both bible_metadata and chunk_list"
    ):
        prepare_alignment_input(
            bible_metadata={"0101": "test"},
            chunk_list=[{"id": "1", "text": "test"}],
        )
