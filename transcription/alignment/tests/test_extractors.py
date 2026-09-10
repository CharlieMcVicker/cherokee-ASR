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
    CachedASREmissionsExtractor,
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


def test_cached_emissions_extractor_miss_then_hit(tmp_path):
    mock_extractor = MagicMock(spec=ASREmissionsExtractor)
    mock_extractor.extract.return_value = [
        TokenEmission(word="osiyo", start_sec=0.0, end_sec=0.5, confidence=0.95),
        TokenEmission(word="tohiju", start_sec=0.6, end_sec=1.1, confidence=0.88),
    ]

    cached = CachedASREmissionsExtractor(
        extractor=mock_extractor,
        cache_dir=tmp_path / "emissions_cache",
        cache_key_prefix="test_prefix",
    )
    assert isinstance(cached, ASREmissionsExtractor)

    # First call - cache miss
    res1 = cached.extract("some_audio.wav")
    assert mock_extractor.extract.call_count == 1
    assert len(res1) == 2
    assert res1[0].word == "osiyo"
    assert res1[0].confidence == 0.95

    # Second call - cache hit
    res2 = cached.extract("some_audio.wav")
    assert mock_extractor.extract.call_count == 1
    assert len(res2) == 2
    assert res2[0].word == "osiyo"
    assert res2[1].word == "tohiju"


def test_cached_emissions_extractor_file_paths(tmp_path):
    f = tmp_path / "audio.wav"
    f.write_bytes(b"dummy wav data 1")

    mock_extractor = MagicMock(spec=ASREmissionsExtractor)
    mock_extractor.extract.return_value = [
        TokenEmission(word="word1", start_sec=0.1, end_sec=0.4, confidence=0.9)
    ]

    cached = CachedASREmissionsExtractor(
        extractor=mock_extractor,
        cache_dir=tmp_path / "emissions_cache",
    )

    # Pass as Path object
    res_path = cached.extract(f)
    assert mock_extractor.extract.call_count == 1
    assert len(res_path) == 1

    # Pass as str pointing to same file -> cache hit
    res_str = cached.extract(str(f))
    assert mock_extractor.extract.call_count == 1
    assert res_str[0].word == "word1"

    # Modify file -> cache invalidation / cache miss
    f.write_bytes(b"dummy wav data 2 with modification")
    res_mod = cached.extract(f)
    assert mock_extractor.extract.call_count == 2
    assert len(res_mod) == 1


def test_cached_emissions_extractor_in_memory_inputs(tmp_path):
    mock_extractor = MagicMock(spec=ASREmissionsExtractor)
    mock_extractor.extract.return_value = [
        TokenEmission(word="memory_word", start_sec=0.0, end_sec=1.0, confidence=0.99)
    ]

    cached = CachedASREmissionsExtractor(
        extractor=mock_extractor,
        cache_dir=tmp_path / "emissions_cache",
    )

    # 1. AudioSegment
    audio1 = AudioSegment.silent(duration=500, frame_rate=16000)
    res_audio1 = cached.extract(audio1)
    assert mock_extractor.extract.call_count == 1

    # Same AudioSegment -> cache hit
    res_audio1_hit = cached.extract(audio1)
    assert mock_extractor.extract.call_count == 1
    assert res_audio1_hit[0].word == "memory_word"

    # Different AudioSegment -> cache miss
    audio2 = AudioSegment.silent(duration=1000, frame_rate=16000)
    res_audio2 = cached.extract(audio2)
    assert mock_extractor.extract.call_count == 2

    # 2. NumPy ndarray
    arr1 = np.zeros(8000, dtype=np.float32)
    res_arr1 = cached.extract(arr1)
    assert mock_extractor.extract.call_count == 3

    # Same ndarray content -> cache hit
    arr1_same = np.zeros(8000, dtype=np.float32)
    res_arr1_hit = cached.extract(arr1_same)
    assert mock_extractor.extract.call_count == 3

    # 3. Raw bytes
    b1 = b"raw_audio_bytes_1"
    cached.extract(b1)
    assert mock_extractor.extract.call_count == 4
    cached.extract(b1)
    assert mock_extractor.extract.call_count == 4


def test_cached_emissions_extractor_prefix_and_invalidation(tmp_path):
    mock1 = MagicMock(spec=ASREmissionsExtractor)
    mock1.extract.return_value = [TokenEmission(word="v1", start_sec=0.0, end_sec=0.5)]

    mock2 = MagicMock(spec=ASREmissionsExtractor)
    mock2.extract.return_value = [TokenEmission(word="v2", start_sec=0.0, end_sec=0.5)]

    cached_v1 = CachedASREmissionsExtractor(
        extractor=mock1,
        cache_dir=tmp_path / "cache",
        cache_key_prefix="model_v1",
    )
    cached_v2 = CachedASREmissionsExtractor(
        extractor=mock2,
        cache_dir=tmp_path / "cache",
        cache_key_prefix="model_v2",
    )

    audio_file = "test.wav"
    res1 = cached_v1.extract(audio_file)
    res2 = cached_v2.extract(audio_file)

    assert mock1.extract.call_count == 1
    assert mock2.extract.call_count == 1
    assert res1[0].word == "v1"
    assert res2[0].word == "v2"


def test_cached_emissions_extractor_edge_cases(tmp_path):
    mock = MagicMock(spec=ASREmissionsExtractor)
    mock.extract.return_value = [
        TokenEmission(word="valid", start_sec=0.0, end_sec=1.0)
    ]

    cached = CachedASREmissionsExtractor(
        extractor=mock,
        cache_dir=tmp_path / "cache",
    )

    # extract(None) should return [] and not call extractor
    assert cached.extract(None) == []
    assert mock.extract.call_count == 0

    # Normal extract creates cache
    res = cached.extract("sample.wav")
    assert len(res) == 1
    assert mock.extract.call_count == 1

    # Corrupt the cache file
    cache_files = list((tmp_path / "cache").glob("*.json"))
    assert len(cache_files) == 1
    cache_files[0].write_text("invalid json content {{{")

    # Should recover gracefully and re-extract
    res_after_corrupt = cached.extract("sample.wav")
    assert len(res_after_corrupt) == 1
    assert mock.extract.call_count == 2


def test_cherokee_asr_extractor_extract_batch():
    mock_model = MagicMock(spec=CherokeeASRModel)
    mock_model.get_logits_batch.return_value = [
        np.zeros((10, 30)),
        np.zeros((15, 30)),
    ]
    mock_model.get_word_confidences.side_effect = [
        [WordConfidence(word="osiyo", start_time=0.2, end_time=0.8, confidence=0.95)],
        [WordConfidence(word="wado", start_time=0.1, end_time=0.6, confidence=0.90)],
    ]

    extractor = CherokeeASRExtractor(model=mock_model, skip_vad=True)
    audio1 = AudioSegment.silent(duration=1000, frame_rate=16000)
    audio2 = AudioSegment.silent(duration=1500, frame_rate=16000)

    results = extractor.extract_batch([audio1, audio2], batch_size=2)
    assert len(results) == 2
    assert len(results[0]) == 1
    assert results[0][0].word == "osiyo"
    assert results[0][0].start_sec == 0.2
    assert len(results[1]) == 1
    assert results[1][0].word == "wado"
    assert results[1][0].start_sec == 0.1

    # Empty inputs
    assert extractor.extract_batch([]) == []


def test_cached_emissions_extractor_populate_cache(tmp_path):
    f1 = tmp_path / "f1.wav"
    f2 = tmp_path / "f2.wav"
    f3 = tmp_path / "f3.wav"
    f1.write_bytes(b"audio1")
    f2.write_bytes(b"audio2")
    f3.write_bytes(b"audio3")

    mock_extractor = MagicMock(spec=CherokeeASRExtractor)
    mock_extractor.extract_batch.return_value = [
        [TokenEmission(word="w1", start_sec=0.0, end_sec=0.5, confidence=0.95)],
        [TokenEmission(word="w2", start_sec=0.1, end_sec=0.6, confidence=0.92)],
        [TokenEmission(word="w3", start_sec=0.2, end_sec=0.7, confidence=0.90)],
    ]

    cached = CachedASREmissionsExtractor(
        extractor=mock_extractor,
        cache_dir=tmp_path / "cache",
        cache_key_prefix="test_batch",
    )

    # 1. Bulk populate all 3 files
    results = cached.populate_cache([f1, f2, f3], batch_size=2)
    assert len(results) == 3
    assert results[0][0].word == "w1"
    assert results[1][0].word == "w2"
    assert results[2][0].word == "w3"
    assert mock_extractor.extract_batch.call_count == 1

    # Verify cache files were created on disk
    cache_files = list((tmp_path / "cache").glob("*.json"))
    assert len(cache_files) == 3

    # 2. Re-running populate_cache without force should hit disk cache and NOT call extractor
    results_hit = cached.populate_cache([f1, f2, f3], batch_size=2)
    assert len(results_hit) == 3
    assert mock_extractor.extract_batch.call_count == 1  # Still 1!

    # 3. Individual extract should also hit disk cache
    res_ind = cached.extract(f1)
    assert len(res_ind) == 1
    assert res_ind[0].word == "w1"
    assert mock_extractor.extract.call_count == 0

    # 4. Fallback when extractor does not have extract_batch
    mock_simple = MagicMock(spec=ASREmissionsExtractor)
    mock_simple.extract.return_value = [
        TokenEmission(word="simple", start_sec=0.0, end_sec=0.5)
    ]
    cached_simple = CachedASREmissionsExtractor(
        extractor=mock_simple,
        cache_dir=tmp_path / "cache_simple",
        cache_key_prefix="test_simple",
    )
    res_simple = cached_simple.populate_cache([f1, f2])
    assert len(res_simple) == 2
    assert mock_simple.extract.call_count == 2


def test_cached_emissions_extractor_infer_bulk_alias(tmp_path):
    f = tmp_path / "single.wav"
    f.write_bytes(b"single audio")

    mock_extractor = MagicMock(spec=ASREmissionsExtractor)
    mock_extractor.extract.return_value = [
        TokenEmission(word="alias_word", start_sec=0.0, end_sec=1.0)
    ]

    cached = CachedASREmissionsExtractor(
        extractor=mock_extractor,
        cache_dir=tmp_path / "cache",
        cache_key_prefix="alias_test",
    )

    res = cached.infer_bulk([f])
    assert len(res) == 1
    assert res[0][0].word == "alias_word"
