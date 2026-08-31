"""
Unit tests for AlignmentPipeline orchestrator in transcription.alignment.pipeline.
"""

import json
import os
import pytest
from unittest.mock import MagicMock

from transcription.alignment.adapters.inbound import GenericChunkListAdapter
from transcription.alignment.adapters.outbound import (
    DebugJsonAdapter,
    ManifestJsonAdapter,
    PraatTextGridAdapter,
)
from transcription.alignment.core.sliding_window import SlidingWindowDTWAligner
from transcription.alignment.domain.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.alignment.pipeline import AlignmentPipeline
from transcription.alignment.ports.protocols import (
    ASREmissionsExtractor,
    ChunkAlignmentEngine,
    InboundChunkAdapter,
    OutboundAlignmentAdapter,
)
from transcription.alignment.strategies.extractors import PrecomputedEmissionsExtractor


class MockChunkAdapter(InboundChunkAdapter):
    def __init__(self, chunks):
        self._chunks = chunks

    def load_chunks(self):
        return list(self._chunks)


def test_alignment_pipeline_basic_execution(tmp_path):
    chunks = [
        TextChunk(
            chunk_id="chunk_1",
            raw_text="osiyo tohiju",
            normalized_text="osiyo tohiju",
            syllabary_text="ᎣᏏᏲ ᏙᎯᏧ",
            metadata={"english": "Hello how are you"},
        )
    ]
    emissions = [
        TokenEmission(word="osiyo", start_sec=0.5, end_sec=1.0, confidence=0.98),
        TokenEmission(word="tohiju", start_sec=1.1, end_sec=1.8, confidence=0.95),
    ]

    adapter = MockChunkAdapter(chunks)
    extractor = PrecomputedEmissionsExtractor(emissions)
    engine = SlidingWindowDTWAligner()
    exporters = [
        ManifestJsonAdapter(),
        PraatTextGridAdapter(),
        DebugJsonAdapter(),
    ]

    pipeline = AlignmentPipeline(
        chunk_adapter=adapter,
        extractor=extractor,
        engine=engine,
        exporters=exporters,
    )

    out_dir = str(tmp_path / "align_out")
    result = pipeline.run(
        audio_input="dummy.wav",
        output_dir=out_dir,
    )

    assert isinstance(result, AlignmentOutput)
    assert len(result.aligned_chunks) == 1
    assert result.aligned_chunks[0].chunk_id == "chunk_1"
    assert result.aligned_chunks[0].start_sec == 0.5
    assert result.aligned_chunks[0].end_sec == 1.8
    assert result.metrics is not None
    assert result.metrics.matched_chunks == 1

    # Verify exported files
    manifest_path = os.path.join(out_dir, "alignment_manifest.json")
    textgrid_path = os.path.join(out_dir, "alignment.TextGrid")
    debug_path = os.path.join(out_dir, "alignment_debug.json")

    assert os.path.exists(manifest_path)
    assert os.path.exists(textgrid_path)
    assert os.path.exists(debug_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert "lines" in manifest_data
    assert len(manifest_data["lines"]) == 1
    assert manifest_data["lines"][0]["line_id"] == "chunk_1"

    with open(debug_path, "r", encoding="utf-8") as f:
        debug_data = json.load(f)
    assert debug_data["aligned_chunks_count"] == 1
    assert len(debug_data["raw_tokens"]) == 2


def test_alignment_pipeline_with_custom_engine_and_exporter(tmp_path):
    chunks = [TextChunk(chunk_id="c1", raw_text="test text")]
    emissions = [TokenEmission(word="test", start_sec=0.0, end_sec=1.0)]

    mock_adapter = MockChunkAdapter(chunks)
    mock_extractor = PrecomputedEmissionsExtractor(emissions)

    mock_engine = MagicMock(spec=ChunkAlignmentEngine)
    mock_output = AlignmentOutput(
        source_id="custom_src",
        aligned_chunks=[
            AlignedChunk(
                chunk_id="c1",
                chunk=chunks[0],
                start_sec=0.0,
                end_sec=1.0,
                words=[WordInterval(word="test", start_sec=0.0, end_sec=1.0)],
                distance_score=0.0,
                emitted_text="test",
            )
        ],
        raw_tokens=emissions,
        metrics=AlignmentMetrics(1, 1, 1.0, 0.0, 4, 4),
    )
    mock_engine.align_chunks.return_value = mock_output

    mock_custom_exporter = MagicMock(spec=OutboundAlignmentAdapter)

    pipeline = AlignmentPipeline(
        chunk_adapter=mock_adapter,
        extractor=mock_extractor,
        engine=mock_engine,
        exporters=[mock_custom_exporter],
    )

    out_dir = str(tmp_path / "custom_out")
    res = pipeline.run(
        audio_input="audio.wav",
        output_dir=out_dir,
    )

    assert res == mock_output
    mock_engine.align_chunks.assert_called_once()
    mock_custom_exporter.export.assert_called_once_with(mock_output, out_dir)

    # Praat and Manifest shouldn't exist since not in exporters
    assert not os.path.exists(os.path.join(out_dir, "alignment.TextGrid"))
    assert not os.path.exists(os.path.join(out_dir, "alignment_manifest.json"))
