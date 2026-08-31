"""
Unit tests for inbound and outbound alignment adapters.
"""

import json
import os
import tempfile
import pytest

from transcription.alignment.adapters.inbound import (
    BibleMetadataVerseAdapter,
    GenericChunkListAdapter,
)
from transcription.alignment.adapters.outbound import (
    ManifestJsonAdapter,
    PraatTextGridAdapter,
)
from transcription.alignment.domain.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    TextChunk,
    TokenEmission,
    WordInterval,
)


def test_bible_metadata_verse_adapter_dict():
    adapter = BibleMetadataVerseAdapter()
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
    chunks = adapter.load_chunks_from_dict(data)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "020101"
    assert chunks[0].raw_text == "A-da-le-ni-s-gv"
    assert chunks[0].syllabary_text == "ᎠᏓᎴᏂᏍᎬ"
    assert chunks[0].metadata["english"] == "The beginning"
    assert chunks[0].normalized_text == "ataleniskv"


def test_bible_metadata_verse_adapter_file():
    adapter = BibleMetadataVerseAdapter()
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

        chunks = adapter.load_chunks_from_file(json_path)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "020101"


def test_generic_chunk_list_adapter():
    adapter = GenericChunkListAdapter()
    items = [
        {
            "id": "c1",
            "text": "osiyo",
            "cherokee": "ᎣᏏᏲ",
            "speaker": "spk1",
        },
        {
            "id": "c2",
            "text": "tohiju",
            "cherokee": "ᏙᎯᏧ",
            "speaker": "spk2",
        },
    ]
    chunks = adapter.load_chunks_from_list(items)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "c1"
    assert chunks[0].raw_text == "osiyo"
    assert chunks[0].syllabary_text == "ᎣᏏᏲ"
    assert chunks[0].metadata["speaker"] == "spk1"

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "chunks.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(items, f)

        chunks_from_file = adapter.load_chunks_from_file(json_path)
        assert len(chunks_from_file) == 2
        assert chunks_from_file[1].chunk_id == "c2"


def test_praat_textgrid_adapter_export():
    adapter = PraatTextGridAdapter()
    chunk = TextChunk(
        chunk_id="chunk_01",
        raw_text="A-da-le-ni-s-gv",
        syllabary_text="ᎠᏓᎴᏂᏍᎬ",
    )
    w1 = WordInterval(
        word="word1", start_sec=1.0, end_sec=1.5, reconciled_word="rec_word1"
    )
    aligned = AlignedChunk(
        chunk_id="chunk_01",
        chunk=chunk,
        start_sec=1.0,
        end_sec=1.5,
        words=[w1],
        distance_score=0.05,
    )
    raw_token = TokenEmission(word="word1", start_sec=1.0, end_sec=1.5)
    output = AlignmentOutput(
        source_id="test.wav",
        aligned_chunks=[aligned],
        raw_tokens=[raw_token],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter.export(output, tmpdir)
        tg_path = os.path.join(tmpdir, "alignment.TextGrid")
        assert os.path.exists(tg_path)

        with open(tg_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert 'File type = "ooTextFile"' in content
        assert 'name = "Chunks"' in content
        assert 'name = "Words"' in content
        assert 'name = "Padded Words"' in content
        assert 'name = "Reconciled Words"' in content
        assert 'name = "Raw ASR Emissions"' in content
        assert "size = 5" in content


def test_manifest_json_adapter_export():
    adapter = ManifestJsonAdapter()
    chunk = TextChunk(
        chunk_id="chunk_01",
        raw_text="A-da-le-ni-s-gv",
        syllabary_text="ᎠᏓᎴᏂᏍᎬ",
        metadata={"english": "The beginning"},
    )
    w1 = WordInterval(
        word="adalenisgv",
        start_sec=1.0,
        end_sec=1.5,
        syllabary="ᎠᏓᎴᏂᏍᎬ",
        reconciled_word="adalenisgv_rec",
    )
    aligned = AlignedChunk(
        chunk_id="chunk_01",
        chunk=chunk,
        start_sec=1.0,
        end_sec=1.5,
        words=[w1],
        distance_score=0.0,
        emitted_text="adalenisgv",
    )
    metrics = AlignmentMetrics(
        total_chunks=1,
        matched_chunks=1,
        match_ratio=1.0,
        mean_distance_score=0.0,
        total_ground_truth_chars=10,
        total_emitted_chars=10,
    )
    output = AlignmentOutput(
        source_id="test.wav",
        aligned_chunks=[aligned],
        metrics=metrics,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter.export(output, tmpdir)
        json_path = os.path.join(tmpdir, "alignment_manifest.json")
        assert os.path.exists(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["audio_source"] == "test.wav"
        assert data["metrics"]["total_chunks"] == 1
        assert len(data["lines"]) == 1
        assert data["lines"][0]["line_id"] == "chunk_01"
        assert data["lines"][0]["english"] == "The beginning"
        assert data["lines"][0]["words"][0]["syllabary_word"] == "ᎠᏓᎴᏂᏍᎬ"
        assert data["lines"][0]["words"][0]["reconciled_word"] == "adalenisgv_rec"


def test_inbound_adapters_implement_protocol():
    from transcription.alignment.ports.protocols import InboundChunkAdapter

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

        bible_adapter = BibleMetadataVerseAdapter(path=json_path)
        assert isinstance(bible_adapter, InboundChunkAdapter)
        chunks = bible_adapter.load_chunks()
        assert len(chunks) == 1

        generic_items = [{"id": "c1", "text": "osiyo"}]
        chunk_json_path = os.path.join(tmpdir, "chunks.json")
        with open(chunk_json_path, "w", encoding="utf-8") as f:
            json.dump(generic_items, f)

        generic_adapter = GenericChunkListAdapter(path=chunk_json_path)
        assert isinstance(generic_adapter, InboundChunkAdapter)
        generic_chunks = generic_adapter.load_chunks()
        assert len(generic_chunks) == 1


def test_outbound_adapters_implement_protocol():
    from transcription.alignment.ports.protocols import OutboundAlignmentAdapter

    praat_adapter = PraatTextGridAdapter()
    manifest_adapter = ManifestJsonAdapter()

    assert isinstance(praat_adapter, OutboundAlignmentAdapter)
    assert isinstance(manifest_adapter, OutboundAlignmentAdapter)
