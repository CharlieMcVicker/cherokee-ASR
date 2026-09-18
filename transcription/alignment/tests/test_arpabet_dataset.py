# -*- coding: utf-8 -*-
"""
transcription.alignment.tests.test_arpabet_dataset

Unit and integration tests for English G2P extraction, torchaudio MMS_FA forced alignment,
phonetic balancing, audio clip extraction, and manifest serialization.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch

from transcription.alignment.arpabet.dataset import (
    PhoneticWordBalancer,
    WordCandidate,
    extract_word_clip,
    load_words_manifest,
    save_words_manifest,
)
from transcription.alignment.arpabet.forced_aligner import (
    AlignedWordSpan,
    MMSForcedAligner,
    get_default_forced_aligner,
)
from transcription.alignment.arpabet.g2p import (
    G2pExtractor,
    extract_arpabet,
    get_default_g2p,
)
from transcription.alignment.arpabet.types import (
    STANDARD_ARPABET_PHONEMES,
    ArpabetToken,
    EnglishToArpabetProtocol,
    G2PExtractorProtocol,
    WordManifestEntry,
)

# ============================================================================
# 1. G2P Extraction Tests
# ============================================================================


def test_g2p_extractor_protocol():
    extractor = G2pExtractor()
    assert isinstance(extractor, EnglishToArpabetProtocol)
    assert isinstance(extractor, G2PExtractorProtocol)


def test_g2p_extraction_stress_stripped():
    extractor = G2pExtractor()
    tokens = extractor.extract("coffee", strip_stress=True)
    phones = [t.phone for t in tokens]
    assert phones == ["K", "AA", "F", "IY"]
    assert all(t.stress is None for t in tokens)


def test_g2p_extraction_with_stress():
    extractor = G2pExtractor()
    tokens = extractor.extract("coffee", strip_stress=False)
    phones = [t.phone for t in tokens]
    assert phones == ["K", "AA", "F", "IY"]
    assert tokens[1].stress == 1  # AA1
    assert tokens[3].stress == 0  # IY0


def test_g2p_punctuation_and_whitespace_cleaning():
    extractor = G2pExtractor()
    tokens = extractor('  "Hello, world!"  ', strip_stress=True)
    phones = [t.phone for t in tokens]
    assert phones == ["HH", "AH", "L", "OW", "W", "ER", "L", "D"]
    assert all(p in STANDARD_ARPABET_PHONEMES for p in phones)


def test_g2p_empty_string():
    extractor = G2pExtractor()
    assert extractor.extract("") == ()
    assert extractor.extract("   ") == ()
    assert extract_arpabet("") == ()


def test_get_default_g2p_singleton():
    g1 = get_default_g2p()
    g2 = get_default_g2p()
    assert g1 is g2


# ============================================================================
# 2. Forced Alignment Tests
# ============================================================================


def test_mms_forced_aligner_clean_word():
    aligner = MMSForcedAligner(device="cpu")
    assert aligner.clean_word("Hello,") == "hello"
    assert aligner.clean_word("it's") == "it's"
    assert aligner.clean_word("well-known") == "well-known"
    assert aligner.clean_word("1234") == ""


def test_mms_forced_aligner_empty_or_short_audio():
    aligner = MMSForcedAligner(device="cpu")
    # Empty audio
    empty_wf = torch.zeros(1, 0)
    assert aligner.align(empty_wf, 16000, ["hello"]) == []

    # Audio too short for many tokens
    short_wf = torch.zeros(1, 160)  # 10ms
    assert aligner.align(short_wf, 16000, ["supercalifragilisticexpialidocious"]) == []


def test_mms_forced_aligner_synthetic_audio():
    aligner = MMSForcedAligner(device="cpu")
    # 2.5 seconds of synthetic audio
    wf = torch.randn(1, 16000 * 2 + 8000)
    words = ["the", "test", "audio"]
    spans = aligner.align(wf, 16000, words)

    assert len(spans) == 3
    for span in spans:
        assert isinstance(span, AlignedWordSpan)
        assert span.start_sec >= 0.0
        assert span.end_sec <= 2.5
        assert span.end_sec >= span.start_sec
        assert span.duration == round(span.end_sec - span.start_sec, 4)
        assert span.token_count > 0


def test_mms_forced_aligner_multichannel_and_resampling():
    aligner = MMSForcedAligner(device="cpu")
    # Stereo audio at 24kHz (2 seconds = 48,000 samples)
    stereo_wf = torch.randn(2, 48000)
    spans = aligner.align(stereo_wf, 24000, ["hello", "world"])
    assert len(spans) == 2


# ============================================================================
# 3. Phonetic Word Balancer Tests
# ============================================================================


def test_balancer_frequency_capping():
    balancer = PhoneticWordBalancer(target_count=10, max_per_word=2)
    toks = (ArpabetToken("T"), ArpabetToken("IY"))

    c1 = WordCandidate(
        candidate_id="c1",
        utterance_id="u1",
        word="TEA",
        start_sec=0.1,
        end_sec=0.4,
        duration=0.3,
        arpabet=toks,
    )
    c2 = WordCandidate(
        candidate_id="c2",
        utterance_id="u2",
        word="TEA",
        start_sec=0.2,
        end_sec=0.5,
        duration=0.3,
        arpabet=toks,
    )
    c3 = WordCandidate(
        candidate_id="c3",
        utterance_id="u3",
        word="TEA",
        start_sec=0.3,
        end_sec=0.6,
        duration=0.3,
        arpabet=toks,
    )

    assert balancer.can_accept(c1.word, c1.arpabet) is True
    assert balancer.add(c1) is True
    assert balancer.can_accept(c2.word, c2.arpabet) is True
    assert balancer.add(c2) is True

    # Third occurrence of TEA must be rejected due to max_per_word=2
    assert balancer.can_accept(c3.word, c3.arpabet) is False
    assert balancer.add(c3) is False
    assert balancer.word_counts["TEA"] == 2


def test_balancer_select_balanced_subset():
    balancer = PhoneticWordBalancer(target_count=3, max_per_word=1)

    # Three candidate words covering rare phoneme ZH and common stops
    c_zh = WordCandidate(
        candidate_id="c_zh",
        utterance_id="u1",
        word="VISION",
        start_sec=0.1,
        end_sec=0.5,
        duration=0.4,
        arpabet=(
            ArpabetToken("V"),
            ArpabetToken("IH"),
            ArpabetToken("ZH"),
            ArpabetToken("AH"),
            ArpabetToken("N"),
        ),
    )
    c_th = WordCandidate(
        candidate_id="c_th",
        utterance_id="u2",
        word="THINK",
        start_sec=0.1,
        end_sec=0.5,
        duration=0.4,
        arpabet=(
            ArpabetToken("TH"),
            ArpabetToken("IH"),
            ArpabetToken("NG"),
            ArpabetToken("K"),
        ),
    )
    c_the1 = WordCandidate(
        candidate_id="c_the1",
        utterance_id="u3",
        word="THE",
        start_sec=0.1,
        end_sec=0.3,
        duration=0.2,
        arpabet=(ArpabetToken("DH"), ArpabetToken("AH")),
    )
    c_the2 = WordCandidate(
        candidate_id="c_the2",
        utterance_id="u4",
        word="THE",
        start_sec=0.5,
        end_sec=0.7,
        duration=0.2,
        arpabet=(ArpabetToken("DH"), ArpabetToken("AH")),
    )

    selected = balancer.select_balanced_subset(
        [c_zh, c_th, c_the1, c_the2], target_count=3
    )
    assert len(selected) == 3

    # Ensure max_per_word=1 enforced on THE
    selected_words = [s.word for s in selected]
    assert selected_words.count("THE") <= 1

    summary = balancer.get_summary()
    assert summary["total_words"] == 3
    assert summary["unique_words"] == 3
    assert summary["phoneme_counts"]["ZH"] == 1
    assert summary["phoneme_counts"]["TH"] == 1


# ============================================================================
# 4. Audio Clip Extraction & Formatting Tests
# ============================================================================


def test_extract_word_clip_padding_and_audio_format(tmp_path: Path):
    sr = 16000
    duration_sec = 2.0
    # 2 seconds tone
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wf = torch.from_numpy(tone).unsqueeze(0)

    out_file = tmp_path / "clip_001.wav"
    word_start = 0.50
    word_end = 0.80

    # Slicing with 25ms padding: [0.475, 0.825] -> duration = 0.35s
    clip_dur = extract_word_clip(
        waveform=wf,
        sample_rate=sr,
        start_sec=word_start,
        end_sec=word_end,
        output_path=out_file,
        padding_sec=0.025,
        target_sample_rate=16000,
    )

    assert out_file.exists()
    assert abs(clip_dur - 0.35) < 0.01

    # Verify physical file properties via soundfile
    info = sf.info(str(out_file))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.format == "WAV"
    assert info.subtype == "PCM_16"
    assert abs(info.duration - 0.35) < 0.01


def test_extract_word_clip_boundary_clamping(tmp_path: Path):
    sr = 16000
    wf = torch.zeros(1, sr)  # 1 second
    out_file = tmp_path / "clamped.wav"

    # Word near start (start=0.01s): start - 0.025s = -0.015s -> must clamp to 0.0
    # Word near end (end=0.99s): end + 0.025s = 1.015s -> must clamp to 1.0
    clip_dur = extract_word_clip(
        waveform=wf,
        sample_rate=sr,
        start_sec=0.01,
        end_sec=0.99,
        output_path=out_file,
        padding_sec=0.025,
        target_sample_rate=16000,
    )

    info = sf.info(str(out_file))
    assert info.duration <= 1.0001
    assert info.duration >= 0.99


# ============================================================================
# 5. Manifest Serialization Tests
# ============================================================================


def test_manifest_roundtrip_and_schema(tmp_path: Path):
    entry1 = WordManifestEntry(
        clip_id="clip_000001",
        audio_path="words/clip_000001.wav",
        word="COFFEE",
        duration=0.45,
        arpabet=(
            ArpabetToken("K"),
            ArpabetToken("AA"),
            ArpabetToken("F"),
            ArpabetToken("IY"),
        ),
        start_sec=1.20,
        end_sec=1.60,
        speaker_id="1272",
    )
    entry2 = WordManifestEntry(
        clip_id="clip_000002",
        audio_path="words/clip_000002.wav",
        word="TEA",
        duration=0.32,
        arpabet=(ArpabetToken("T"), ArpabetToken("IY")),
        start_sec=2.10,
        end_sec=2.37,
        speaker_id="1272",
    )

    manifest_path = tmp_path / "words_manifest.json"
    save_words_manifest([entry1, entry2], manifest_path)
    assert manifest_path.exists()

    loaded = load_words_manifest(manifest_path)
    assert len(loaded) == 2
    assert loaded[0] == entry1
    assert loaded[1] == entry2
    assert loaded[0].arpabet_phones == ("K", "AA", "F", "IY")
    assert loaded[0].speaker_id == "1272"
