# -*- coding: utf-8 -*-
"""
Unit tests for CLI runner in transcription.alignment.cli.
"""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

import numpy as np
from pydub import AudioSegment

from transcription.alignment.cli import (
    main as legacy_main,
    run_alignment_pipeline as legacy_run_alignment_pipeline,
)
from transcription.apps.cli import main, run_alignment_pipeline
from transcription.alignment.models import AlignmentOutput
from transcription.audio.segment import AudioChunk
from transcription.core.models.output import ModelOutput


@pytest.fixture
def mock_asr_model():
    with patch(
        "transcription.models.asr_model.CherokeeASRModel.from_pretrained_or_best"
    ) as mock_from_pretrained:
        mock_model = MagicMock()
        # Synthetic ModelOutput with tokens for "osiyo tohiju"
        vocab = {
            "[PAD]": 0,
            "|": 1,
            "o": 2,
            "s": 3,
            "i": 4,
            "y": 5,
            "t": 6,
            "h": 7,
            "j": 8,
            "u": 9,
        }
        lpz = np.full((100, len(vocab)), -10.0, dtype=np.float32)
        lpz[:, 0] = 0.0
        # osiyo: 10..30
        for i, ch in enumerate(["o", "s", "i", "y", "o"]):
            lpz[10 + i * 4 : 10 + i * 4 + 2, vocab[ch]] = 8.0
        lpz[32:34, vocab["|"]] = 8.0
        # tohiju: 40..65
        for i, ch in enumerate(["t", "o", "h", "i", "j", "u"]):
            lpz[40 + i * 4 : 40 + i * 4 + 2, vocab[ch]] = 8.0

        mock_out = ModelOutput(
            lpz=lpz,
            vocab=vocab,
            frame_duration_sec=0.02,
            metadata={"pad_token_id": 0, "word_delimiter_token_id": 1},
        )
        mock_model.infer.return_value = mock_out
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
    manifest_path = os.path.join(out_dir, "alignment_manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert "reconciled_words" in manifest_data
    assert "reconciled_words" in manifest_data["lines"][0]
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
        with patch("transcription.apps.cli.run_alignment_pipeline") as mock_run:
            main()
            mock_run.assert_called_once_with(
                audio_path="audio.wav",
                bible_metadata_path=None,
                chunk_list_path=str(chunks_path),
                transcript_path=None,
                output_dir=out_dir,
                export_praat=True,
                model_path=None,
                skip_vad=True,
                debug_export=True,
                reconcile=True,
                code_switched=False,
            )


def test_cli_main_code_switched(tmp_path):
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text("ᎯᎠ coffee ᎠᎩᏚᎵ")
    out_dir = str(tmp_path / "out_cs")

    test_args = [
        "align_cli",
        "--audio",
        "audio.wav",
        "--transcript",
        str(transcript_path),
        "--code-switched",
        "--output-dir",
        out_dir,
    ]

    with patch.object(sys, "argv", test_args):
        with patch("transcription.apps.cli.run_alignment_pipeline") as mock_run:
            main()
            mock_run.assert_called_once_with(
                audio_path="audio.wav",
                bible_metadata_path=None,
                chunk_list_path=None,
                transcript_path=str(transcript_path),
                output_dir=out_dir,
                export_praat=True,
                model_path=None,
                skip_vad=False,
                debug_export=False,
                reconcile=False,
                code_switched=True,
            )


def test_legacy_cli_shim_forwarding():
    assert legacy_main is main
    assert legacy_run_alignment_pipeline is run_alignment_pipeline
