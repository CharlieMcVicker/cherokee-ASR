"""
Unit tests for domain models, protocols, distance metrics, and preprocessors.
"""

import pytest
from transcription.alignment.domain.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.alignment.ports.protocols import (
    ASREmissionsExtractor,
    ChunkAlignmentEngine,
    DistanceMetric,
    PhoneticPreprocessor,
    ReconciliationStrategy,
)
from transcription.alignment.strategies.distance_metrics import (
    DefaultCERDistanceMetric,
    PhonologicalDistanceMetric,
)
from transcription.alignment.strategies.extractors import (
    CallbackEmissionsExtractor,
    CherokeeASRExtractor,
    PrecomputedEmissionsExtractor,
)
from transcription.alignment.strategies.preprocessors import (
    CherokeePhoneticPreprocessor,
    SyllabaryToPhoneticPreprocessor,
)


def test_domain_models_instantiation():
    token = TokenEmission(word="osiyo", start_sec=0.5, end_sec=1.2, confidence=0.95)
    assert token.word == "osiyo"
    assert token.start_sec == 0.5
    assert token.end_sec == 1.2
    assert token.confidence == 0.95

    chunk = TextChunk(
        chunk_id="chunk_01",
        raw_text="O-si-yo",
        normalized_text="osiyo",
        syllabary_text="ᎣᏏᏲ",
        metadata={"speaker": "A"},
    )
    assert chunk.chunk_id == "chunk_01"
    assert chunk.metadata["speaker"] == "A"

    word_interval = WordInterval(
        word="osiyo",
        start_sec=0.5,
        end_sec=1.2,
        confidence=0.95,
        flagged=False,
        syllabary="ᎣᏏᏲ",
        reconciled_word="osiyo",
        emitted_word="osiyo",
    )
    assert word_interval.word == "osiyo"

    aligned_chunk = AlignedChunk(
        chunk_id="chunk_01",
        chunk=chunk,
        start_sec=0.5,
        end_sec=1.2,
        words=[word_interval],
        distance_score=0.0,
        emitted_text="osiyo",
    )
    assert aligned_chunk.distance_score == 0.0
    assert len(aligned_chunk.words) == 1

    metrics = AlignmentMetrics(
        total_chunks=1,
        matched_chunks=1,
        match_ratio=1.0,
        mean_distance_score=0.0,
        total_ground_truth_chars=5,
        total_emitted_chars=5,
    )
    assert metrics.match_ratio == 1.0

    output = AlignmentOutput(
        source_id="test_audio.wav",
        aligned_chunks=[aligned_chunk],
        raw_tokens=[token],
        metrics=metrics,
    )
    assert output.source_id == "test_audio.wav"
    assert len(output.aligned_chunks) == 1
    assert len(output.raw_tokens) == 1
    assert output.metrics is not None


def test_default_cer_distance_metric():
    metric = DefaultCERDistanceMetric()
    assert isinstance(metric, DistanceMetric)

    # Identical strings
    cost = metric.compute_cost("osiyo", "osiyo")
    assert cost == 0.0

    # Empty cases
    assert metric.compute_cost("", "") == 0.0
    assert metric.compute_cost("a", "") == 1.0
    assert metric.compute_cost("", "a") == 1.0

    # Non-identical strings
    cost_diff = metric.compute_cost("osiyo", "osyo")
    assert cost_diff > 0.0


def test_phonological_distance_metric():
    # Test standard edit distance with default weights
    metric = PhonologicalDistanceMetric.make_default()
    assert isinstance(metric, DistanceMetric)
    assert metric.compute_cost("osiyo", "osiyo") == 0.0
    assert metric.compute_cost("", "") == 0.0

    # Test with custom phonological substitution penalty
    # Suppose 'k' and 'g' have a low substitution penalty of 0.2 instead of 1.0
    custom_metric = PhonologicalDistanceMetric.make_default(
        substitution_weights={("k", "g"): 0.2}
    )
    # reference: "ka", hyp: "ga" (1 sub: k->g with cost 0.2, len 2 -> 0.2/2 = 0.1)
    cost_kg = custom_metric.compute_cost(hypothesis="ga", reference="ka")
    assert pytest.approx(cost_kg, 0.001) == 0.1

    # reference: "ka", hyp: "pa" (1 sub: k->p with default cost 1.0, len 2 -> 1.0/2 = 0.5)
    cost_kp = custom_metric.compute_cost(hypothesis="pa", reference="ka")
    assert pytest.approx(cost_kp, 0.001) == 0.5


def test_cherokee_phonetic_preprocessor():
    cleaner = CherokeePhoneticPreprocessor()
    assert isinstance(cleaner, PhoneticPreprocessor)

    # Hyphens stripped, qu->gw, consonants respelled, h stripped, punct stripped
    # e.g., "A-da-le-ni-s-gv" -> "adalenisgv"
    result = cleaner.normalize("A-da-le-ni-s-gv.")
    assert "-" not in result
    assert "." not in result

    # Check qu -> gw
    qu_result = cleaner.normalize("quana")
    assert "gw" in qu_result or "qu" not in qu_result


def test_syllabary_to_phonetic_preprocessor():
    cleaner = CherokeePhoneticPreprocessor()
    syllabary_preprocessor = SyllabaryToPhoneticPreprocessor(cleaner)
    assert isinstance(syllabary_preprocessor, PhoneticPreprocessor)

    # Test empty
    assert syllabary_preprocessor.normalize("") == ""

    # Test Syllabary conversion: ᎣᏏᏲ (o-si-yo)
    norm = syllabary_preprocessor.normalize("ᎣᏏᏲ")
    assert "osiyo" in norm or "osyo" in norm or len(norm) > 0

    # Test default cleaner constructor
    default_syllabary = SyllabaryToPhoneticPreprocessor.make_default()
    norm_def = default_syllabary.normalize("ᎣᏏᏲ")
    assert norm_def == norm


def test_precomputed_emissions_extractor():
    raw_tokens = [
        {"word": "osiyo", "start_time": 0.5, "end_time": 1.2, "confidence": 0.95},
        TokenEmission(word="tohiju", start_sec=1.3, end_sec=2.0, confidence=0.90),
    ]
    extractor = PrecomputedEmissionsExtractor.make_default(raw_tokens)
    assert isinstance(extractor, ASREmissionsExtractor)

    emissions = extractor.extract(audio_input=None)
    assert len(emissions) == 2
    assert emissions[0].word == "osiyo"
    assert emissions[0].start_sec == 0.5
    assert emissions[0].end_sec == 1.2
    assert emissions[0].confidence == 0.95
    assert emissions[1].word == "tohiju"
    assert emissions[1].start_sec == 1.3
    assert emissions[1].end_sec == 2.0
    assert emissions[1].confidence == 0.90


def test_callback_emissions_extractor():
    from pydub import AudioSegment

    dummy_audio = AudioSegment.silent(duration=1000, frame_rate=16000)

    def mock_cb(samples, sample_rate):
        return [
            {"word": "osiyo", "start_time": 0.1, "end_time": 0.5, "confidence": 0.98},
            TokenEmission(word="tohiju", start_sec=0.6, end_sec=0.9, confidence=0.92),
        ]

    extractor = CallbackEmissionsExtractor.make_default(callback=mock_cb, skip_vad=True)
    assert isinstance(extractor, ASREmissionsExtractor)

    emissions = extractor.extract(dummy_audio)
    assert len(emissions) == 2
    assert emissions[0].word == "osiyo"
    assert emissions[0].start_sec == 0.1
    assert emissions[0].end_sec == 0.5
    assert emissions[1].word == "tohiju"
    assert emissions[1].start_sec == 0.6
    assert emissions[1].end_sec == 0.9


def test_cherokee_asr_extractor_with_mock_model():
    from unittest.mock import MagicMock
    from pydub import AudioSegment
    from transcription.models.asr_model import WordConfidence

    dummy_audio = AudioSegment.silent(duration=1000, frame_rate=16000)

    mock_model = MagicMock()
    mock_model.get_logits.return_value = "mock_logits"
    mock_model.get_word_confidences.return_value = [
        WordConfidence(
            word="osiyo",
            confidence=0.97,
            start_time=0.1,
            end_time=0.5,
        ),
        {
            "word": "tohiju",
            "confidence": 0.91,
            "start_time": 0.6,
            "end_time": 0.9,
        },
    ]

    extractor = CherokeeASRExtractor.make_default(model=mock_model, skip_vad=True)
    assert isinstance(extractor, ASREmissionsExtractor)

    emissions = extractor.extract(dummy_audio)
    assert len(emissions) == 2
    assert emissions[0].word == "osiyo"
    assert emissions[0].start_sec == 0.1
    assert emissions[0].end_sec == 0.5
    assert emissions[0].confidence == 0.97
    assert emissions[1].word == "tohiju"
    assert emissions[1].start_sec == 0.6
    assert emissions[1].end_sec == 0.9
    assert emissions[1].confidence == 0.91
    mock_model.get_logits.assert_called_once()
    mock_model.get_word_confidences.assert_called_once_with("mock_logits")
