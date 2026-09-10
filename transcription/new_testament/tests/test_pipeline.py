# -*- coding: utf-8 -*-
"""
Unit tests for New Testament pipeline in transcription.new_testament.pipeline.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from pydub import AudioSegment

from transcription.alignment.distance_metrics import (
    ConfusionMatrixCostMetric,
    DefaultCERDistanceMetric,
)
from transcription.alignment.extractors import (
    ASREmissionsExtractor,
    CachedASREmissionsExtractor,
    PrecomputedEmissionsExtractor,
)
from transcription.alignment.models import AlignmentOutput, TokenEmission
from transcription.audio.segment import AudioChunk
from transcription.new_testament.pipeline import (
    align_chapter,
    load_chapter_transcript,
    reconcile_syllabary_asr,
)


@pytest.fixture
def dummy_transcript(tmp_path):
    t_path = tmp_path / "transcript.json"
    data = {
        "020101": {
            "image_path": "images/020101.png",
            "english": "The beginning of the gospel",
            "cherokee": "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎧᏃᎮᏛ",
            "phonetic": "A-da-le-ni-s-gv yi-s-dv ka-no-he-dv",
        }
    }
    t_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return t_path


@pytest.fixture
def dummy_cost_matrix(tmp_path):
    matrix_path = tmp_path / "confusion_cost_matrix_prebible.json"
    data = {
        "unigram_costs": {
            "a": {"a": 0.0, "e": 0.5},
            "e": {"e": 0.0, "a": 0.5},
        },
        "insertion_cost": 1.0,
        "deletion_cost": 1.0,
        "default_substitution_cost": 0.9,
    }
    matrix_path.write_text(json.dumps(data), encoding="utf-8")
    return matrix_path


def test_load_chapter_transcript(dummy_transcript):
    data = load_chapter_transcript(dummy_transcript)
    assert "020101" in data
    assert data["020101"]["cherokee"] == "ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎧᏃᎮᏛ"

    with pytest.raises(FileNotFoundError):
        load_chapter_transcript("nonexistent_path_xyz.json")


def test_reconcile_syllabary_asr():
    enriched, pairs = reconcile_syllabary_asr("ᎣᏏᏲ", "osiyo")
    assert isinstance(enriched, str)
    assert len(pairs) > 0


def test_align_chapter_with_custom_distance_metric_and_extractor(
    tmp_path, dummy_transcript
):
    dummy_emissions = [
        TokenEmission(word="adalenisgv", start_sec=0.1, end_sec=0.6, confidence=0.95),
        TokenEmission(word="yisdv", start_sec=0.7, end_sec=1.1, confidence=0.92),
        TokenEmission(word="kanohedv", start_sec=1.2, end_sec=1.8, confidence=0.90),
    ]
    extractor = PrecomputedEmissionsExtractor(token_emissions=dummy_emissions)
    metric = DefaultCERDistanceMetric()

    out_dir = tmp_path / "output_custom"

    res = align_chapter(
        audio_path="dummy_audio.wav",
        transcript_path=dummy_transcript,
        output_dir=out_dir,
        export_praat=True,
        reconcile=True,
        distance_metric=metric,
        emissions_extractor=extractor,
    )

    assert isinstance(res, AlignmentOutput)
    assert len(res.aligned_chunks) == 1
    assert res.aligned_chunks[0].chunk_id == "020101"
    assert len(res.aligned_chunks[0].words) > 0
    assert (out_dir / "alignment.TextGrid").exists()
    assert (out_dir / "alignment_manifest.json").exists()


def test_align_chapter_with_cached_extractor_and_confusion_metric(
    tmp_path, dummy_transcript, dummy_cost_matrix
):
    dummy_emissions = [
        TokenEmission(word="adalenisgv", start_sec=0.1, end_sec=0.6, confidence=0.95),
        TokenEmission(word="yisdv", start_sec=0.7, end_sec=1.1, confidence=0.92),
        TokenEmission(word="kanohedv", start_sec=1.2, end_sec=1.8, confidence=0.90),
    ]
    base_extractor = PrecomputedEmissionsExtractor(token_emissions=dummy_emissions)
    cached_extractor = CachedASREmissionsExtractor(
        extractor=base_extractor,
        cache_dir=tmp_path / "cache",
        cache_key_prefix="test_prefix",
    )
    metric = ConfusionMatrixCostMetric.from_json(dummy_cost_matrix)

    out_dir = tmp_path / "output_cached"

    res = align_chapter(
        audio_path="dummy_audio.wav",
        transcript_path=dummy_transcript,
        output_dir=out_dir,
        distance_metric=metric,
        emissions_extractor=cached_extractor,
    )

    assert isinstance(res, AlignmentOutput)
    assert len(res.aligned_chunks) == 1
    assert res.metrics is not None
    assert res.metrics.matched_chunks == 1


def test_align_chapter_default_fallback_instantiation(tmp_path, dummy_transcript):
    out_dir = tmp_path / "output_default"

    with (
        patch(
            "transcription.models.asr_model.CherokeeASRModel.from_pretrained_or_best"
        ) as mock_model_load,
        patch("transcription.alignment.extractors.segment_long_audio") as mock_segment,
    ):

        mock_model = MagicMock()
        mock_model.get_logits.return_value = "mock_logits"
        mock_model.get_word_confidences.return_value = [
            {
                "word": "adalenisgv",
                "start_time": 0.2,
                "end_time": 0.8,
                "confidence": 0.95,
            },
        ]
        mock_model_load.return_value = mock_model

        dummy_chunk = AudioChunk(
            chunk_index=0,
            audio=AudioSegment.silent(duration=2000, frame_rate=16000),
            start_sec=0.0,
            end_sec=2.0,
        )
        mock_segment.return_value = [dummy_chunk]

        res = align_chapter(
            audio_path="dummy_audio.wav",
            transcript_path=dummy_transcript,
            output_dir=out_dir,
            cache_dir=tmp_path / "cache_dir",
            model_revision="5464d15",
        )

        assert mock_model_load.called
        assert isinstance(res, AlignmentOutput)
