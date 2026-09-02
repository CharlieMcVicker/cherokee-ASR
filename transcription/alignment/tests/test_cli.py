# -*- coding: utf-8 -*-
"""
Unit tests for CLI runner in transcription.alignment.cli.
"""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

from pydub import AudioSegment

from transcription.alignment.cli import main, run_alignment_pipeline
from transcription.alignment.models import AlignmentOutput
from transcription.audio.segment import AudioChunk


@pytest.fixture
def mock_asr_model():
    with patch(
        "transcription.models.asr_model.CherokeeASRModel.from_pretrained"
    ) as mock_from_pretrained:
        mock_model = MagicMock()
        mock_model.get_logits.return_value = "mock_logits"
        mock_model.get_word_confidences.return_value = [
            {"word": "osiyo", "start_time": 0.2, "end_time": 0.8, "confidence": 0.95},
            {"word": "tohiju", "start_time": 0.9, "end_time": 1.5, "confidence": 0.92},
        ]
        mock_from_pretrained.return_value = mock_model
        yield mock_model


def test_run_alignment_pipeline_with_bible_metadata(tmp_path, mock_asr_model):
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "001001": {
                    "phonetic": "osiyo tohiju",
                    "cherokee": "ᎣᏏᏲ ᏙᎯᏧ",
                    "english": "Hello how are you",
                }
            }
        )
    )

    out_dir = str(tmp_path / "output")

    with patch("transcription.alignment.extractors.segment_long_audio") as mock_segment:
        dummy_chunk = AudioChunk(
            chunk_index=0,
            audio=AudioSegment.silent(duration=2000, frame_rate=16000),
            start_sec=0.0,
            end_sec=2.0,
        )
        mock_segment.return_value = [dummy_chunk]

        result = run_alignment_pipeline(
            audio_path="dummy_path.wav",
            output_dir=out_dir,
            bible_metadata_path=str(meta_path),
            export_praat=True,
            export_manifest=True,
            debug_export=True,
            reconcile=True,
        )

    assert isinstance(result, AlignmentOutput)
    assert len(result.aligned_chunks) == 1
    assert result.aligned_chunks[0].chunk_id == "001001"
    assert os.path.exists(os.path.join(out_dir, "alignment_manifest.json"))
    assert os.path.exists(os.path.join(out_dir, "alignment.TextGrid"))
    assert os.path.exists(os.path.join(out_dir, "alignment_debug.json"))


def test_run_alignment_pipeline_with_chunk_list(tmp_path, mock_asr_model):
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "c1",
                    "raw_text": "osiyo",
                    "cherokee_syllabary": "ᎣᏏᏲ",
                }
            ]
        )
    )

    out_dir = str(tmp_path / "output_chunks")

    with patch("transcription.alignment.extractors.segment_long_audio") as mock_segment:
        dummy_chunk = AudioChunk(
            chunk_index=0,
            audio=AudioSegment.silent(duration=1000, frame_rate=16000),
            start_sec=0.0,
            end_sec=1.0,
        )
        mock_segment.return_value = [dummy_chunk]

        result = run_alignment_pipeline(
            audio_path="dummy_path.wav",
            output_dir=out_dir,
            chunk_list_path=str(chunks_path),
            skip_vad=False,
        )

    assert isinstance(result, AlignmentOutput)
    assert len(result.aligned_chunks) == 1
    assert result.aligned_chunks[0].chunk_id == "c1"


def test_run_alignment_pipeline_missing_args(tmp_path):
    with pytest.raises(ValueError):
        run_alignment_pipeline(
            audio_path="audio.wav",
            output_dir=str(tmp_path),
        )


def test_cli_main_argument_parsing(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps([{"chunk_id": "1", "raw_text": "test"}]))
    out_dir = str(tmp_path / "out")

    test_args = [
        "align_cli",
        "--audio",
        "audio.wav",
        "--chunk-list",
        str(chunks_path),
        "--output-dir",
        out_dir,
        "--skip-vad",
        "--debug-export",
        "--reconcile",
    ]

    with patch.object(sys, "argv", test_args):
        with patch("transcription.alignment.cli.run_alignment_pipeline") as mock_run:
            main()
            mock_run.assert_called_once_with(
                audio_path="audio.wav",
                bible_metadata_path=None,
                chunk_list_path=str(chunks_path),
                output_dir=out_dir,
                export_praat=True,
                model_path=None,
                skip_vad=True,
                debug_export=True,
                reconcile=True,
            )
