# -*- coding: utf-8 -*-
"""
Unit tests for extractors.py.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest
from pydub import AudioSegment

from transcription.alignment.extractors import (
    ASREmissionsExtractor,
    CallbackEmissionsExtractor,
    CherokeeASRExtractor,
    PrecomputedEmissionsExtractor,
    prepare_audio_chunks,
)
from transcription.alignment.models import TokenEmission
from transcription.audio.segment import AudioChunk
from transcription.models.asr_model import CherokeeASRModel, WordConfidence


def test_prepare_audio_chunks_skip_vad():
    silence = AudioSegment.silent(duration=1000, frame_rate=16000)
    chunks = prepare_audio_chunks(silence, skip_vad=True)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_sec == 0.0
    assert chunks[0].end_sec == 1.0

    arr = np.zeros(16000, dtype=np.float32)
    chunks_arr = prepare_audio_chunks(arr, skip_vad=True)
    assert len(chunks_arr) == 1
    assert chunks_arr[0].end_sec == 1.0

    # Test None / unsupported
    assert prepare_audio_chunks(None, skip_vad=True) == []


def test_prepare_audio_chunks_with_vad(monkeypatch):
    dummy_chunk = AudioChunk(
        chunk_index=0,
        audio=AudioSegment.silent(duration=1000, frame_rate=16000),
        start_sec=0.0,
        end_sec=1.0,
    )
    monkeypatch.setattr(
        "transcription.alignment.extractors.segment_long_audio",
        lambda audio_input: [dummy_chunk],
    )
    chunks = prepare_audio_chunks("test.wav", skip_vad=False)
    assert len(chunks) == 1
    assert chunks[0].start_sec == 0.0


def test_precomputed_emissions_extractor():
    raw_list = [
        TokenEmission(word="token1", start_sec=0.0, end_sec=0.5, confidence=0.99),
        {"word": "token2", "start_sec": 0.6, "end_sec": 1.2, "confidence": 0.85},
    ]
    extractor = PrecomputedEmissionsExtractor(token_emissions=raw_list)
    assert isinstance(extractor, ASREmissionsExtractor)

    emissions = extractor.extract(audio_input="ignored.wav")
    assert len(emissions) == 2
    assert emissions[0].word == "token1"
    assert emissions[0].start_sec == 0.0
    assert emissions[0].end_sec == 0.5
    assert emissions[0].confidence == 0.99
    assert emissions[1].word == "token2"
    assert emissions[1].start_sec == 0.6
    assert emissions[1].end_sec == 1.2
    assert emissions[1].confidence == 0.85


def test_callback_emissions_extractor():
    def dummy_callback(samples, sample_rate):
        return [
            {"word": "word1", "start_time": 0.1, "end_time": 0.5, "confidence": 0.88},
            TokenEmission(word="word2", start_sec=0.6, end_sec=1.0, confidence=0.92),
        ]

    extractor = CallbackEmissionsExtractor(callback=dummy_callback, skip_vad=True)
    assert isinstance(extractor, ASREmissionsExtractor)

    assert extractor.extract(None) == []

    audio = AudioSegment.silent(duration=1500, frame_rate=16000)
    emissions = extractor.extract(audio)

    assert len(emissions) == 2
    assert emissions[0].word == "word1"
    assert emissions[0].start_sec == 0.1
    assert emissions[0].end_sec == 0.5
    assert emissions[0].confidence == 0.88
    assert emissions[1].word == "word2"
    assert emissions[1].start_sec == 0.6
    assert emissions[1].end_sec == 1.0
    assert emissions[1].confidence == 0.92


def test_cherokee_asr_extractor():
    mock_model = MagicMock(spec=CherokeeASRModel)
    mock_model.get_logits.return_value = np.zeros((10, 30))
    mock_model.get_word_confidences.return_value = [
        WordConfidence(word="osiyo", start_time=0.2, end_time=0.8, confidence=0.95),
        WordConfidence(word="tohiju", start_time=1.0, end_time=1.5, confidence=0.90),
    ]

    extractor = CherokeeASRExtractor(model=mock_model, skip_vad=True)
    assert isinstance(extractor, ASREmissionsExtractor)

    assert extractor.extract(None) == []

    audio = AudioSegment.silent(duration=2000, frame_rate=16000)
    emissions = extractor.extract(audio)

    assert len(emissions) == 2
    assert emissions[0].word == "osiyo"
    assert emissions[0].start_sec == 0.2
    assert emissions[0].end_sec == 0.8
    assert emissions[0].confidence == 0.95
    assert emissions[1].word == "tohiju"
    assert emissions[1].start_sec == 1.0
    assert emissions[1].end_sec == 1.5
    assert emissions[1].confidence == 0.90
