# -*- coding: utf-8 -*-
"""
Unit tests for ARPAbet and Cherokee alignment domain types, models, protocols,
and serialization (transcription.alignment.arpabet.types).
"""

from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
from typing import Optional
import pytest

from transcription.alignment.arpabet import (
    EPSILON_TOKEN,
    STANDARD_ARPABET_CONSONANTS,
    STANDARD_ARPABET_PHONEMES,
    STANDARD_ARPABET_VOWELS,
    CANONICAL_CHEROKEE_CONSONANTS,
    CANONICAL_CHEROKEE_PHONEMES,
    CANONICAL_CHEROKEE_TTH_PHONEMES,
    CANONICAL_CHEROKEE_VOWELS,
    AcousticConfusionMatrix,
    AlignedTokenPair,
    ArpabetToken,
    CherokeeToken,
    DPTracebackAccumulatorProtocol,
    EnglishToArpabetProtocol,
    G2PExtractorProtocol,
    InferenceCacheManifest,
    SyntheticCherokeeTarget,
    SyntheticTargetProjectorProtocol,
    TopKHypothesis,
    TracebackAlignerProtocol,
    TracebackAlignmentResult,
    WordInferenceCacheEntry,
    WordManifestEntry,
)
from transcription.utils.orthography import Orthography

# ============================================================================
# 1. Phonetic Tokens Tests
# ============================================================================


def test_arpabet_token_normalization_and_stress():
    tok = ArpabetToken("aa1")
    assert tok.phone == "AA"
    assert tok.stress == 1
    assert tok.is_vowel is True
    assert tok.is_epsilon is False
    assert str(tok) == "AA1"

    tok2 = ArpabetToken("k")
    assert tok2.phone == "K"
    assert tok2.stress is None
    assert tok2.is_vowel is False
    assert str(tok2) == "K"

    tok3 = ArpabetToken("  eh0 ")
    assert tok3.phone == "EH"
    assert tok3.stress == 0
    assert tok3.is_vowel is True


def test_arpabet_token_epsilon():
    eps1 = ArpabetToken("<eps>")
    assert eps1.is_epsilon is True
    assert eps1.phone == EPSILON_TOKEN
    assert eps1.stress is None

    eps2 = ArpabetToken("")
    assert eps2.is_epsilon is True
    assert eps2.phone == EPSILON_TOKEN

    eps3 = ArpabetToken("EPS")
    assert eps3.is_epsilon is True
    assert eps3.phone == EPSILON_TOKEN


def test_arpabet_token_immutability():
    tok = ArpabetToken("T")
    with pytest.raises(FrozenInstanceError):
        tok.phone = "D"  # type: ignore[misc]


def test_arpabet_token_serialization_roundtrip():
    tok = ArpabetToken("OW2")
    d = tok.to_dict()
    assert d == {"phone": "OW", "stress": 2}
    tok_restored = ArpabetToken.from_dict(d)
    assert tok_restored == tok

    j = tok.to_json()
    assert ArpabetToken.from_json(j) == tok

    # String input support in from_dict
    assert ArpabetToken.from_dict("B") == ArpabetToken("B")


def test_cherokee_token_normalization_and_vowel():
    tok = CherokeeToken("TH")
    assert tok.phone == "th"
    assert tok.orthography == Orthography.TTH
    assert tok.is_vowel is False
    assert tok.is_epsilon is False
    assert str(tok) == "th"

    vowel_tok = CherokeeToken("a")
    assert vowel_tok.phone == "a"
    assert vowel_tok.is_vowel is True

    # Syllabary orthography preservation
    syl_tok = CherokeeToken("Ꭷ", orthography=Orthography.SYLLABARY)
    assert syl_tok.phone == "Ꭷ"
    assert syl_tok.orthography == Orthography.SYLLABARY


def test_cherokee_token_epsilon():
    eps1 = CherokeeToken("<eps>")
    assert eps1.is_epsilon is True
    assert eps1.phone == EPSILON_TOKEN

    eps2 = CherokeeToken("")
    assert eps2.is_epsilon is True
    assert eps2.phone == EPSILON_TOKEN


def test_cherokee_token_immutability():
    tok = CherokeeToken("kh")
    with pytest.raises(FrozenInstanceError):
        tok.phone = "g"  # type: ignore[misc]


def test_cherokee_token_serialization_roundtrip():
    tok = CherokeeToken("ts", orthography=Orthography.TTH)
    d = tok.to_dict()
    assert d == {"phone": "ts", "orthography": "tth"}
    assert CherokeeToken.from_dict(d) == tok
    assert CherokeeToken.from_json(tok.to_json()) == tok

    # String input in from_dict
    assert CherokeeToken.from_dict("tl") == CherokeeToken("tl")


# ============================================================================
# 2. Word Manifest Tests
# ============================================================================


def test_word_manifest_entry():
    entry = WordManifestEntry(
        clip_id="librispeech_0001",
        audio_path="words/0001.wav",
        word="coffee",
        duration=0.45,
        arpabet=(
            ArpabetToken("K"),
            ArpabetToken("AA1"),
            ArpabetToken("F"),
            ArpabetToken("IY0"),
        ),
        start_sec=1.20,
        end_sec=1.65,
        speaker_id="spk_42",
    )

    assert entry.clip_id == "librispeech_0001"
    assert entry.duration == 0.45
    assert entry.arpabet_phones == ("K", "AA", "F", "IY")

    # Roundtrip serialization
    d = entry.to_dict()
    assert d["clip_id"] == "librispeech_0001"
    assert len(d["arpabet"]) == 4

    restored = WordManifestEntry.from_dict(d)
    assert restored == entry

    j = entry.to_json()
    assert WordManifestEntry.from_json(j) == entry

    with pytest.raises(FrozenInstanceError):
        entry.duration = 0.50  # type: ignore[misc]


# ============================================================================
# 3. Model Inference & Cache Models Tests
# ============================================================================


def test_top_k_hypothesis():
    hyp = TopKHypothesis(
        rank=1,
        text="khowhi",
        score=-1.85,
        tokens=(
            CherokeeToken("kh"),
            CherokeeToken("o"),
            CherokeeToken("w"),
            CherokeeToken("h"),
            CherokeeToken("i"),
        ),
        token_confidences=(0.95, 0.92, 0.88, 0.85, 0.90),
    )

    assert hyp.rank == 1
    assert hyp.text == "khowhi"
    assert hyp.score == -1.85
    assert len(hyp.tokens) == 5

    # Roundtrip
    d = hyp.to_dict()
    assert TopKHypothesis.from_dict(d) == hyp
    assert TopKHypothesis.from_json(hyp.to_json()) == hyp

    with pytest.raises(FrozenInstanceError):
        hyp.score = -2.0  # type: ignore[misc]


def test_word_inference_cache_entry():
    greedy_tokens = (
        CherokeeToken("kh"),
        CherokeeToken("a"),
        CherokeeToken("w"),
        CherokeeToken("i"),
    )
    top_hyp = TopKHypothesis(
        rank=1,
        text="khawi",
        score=-0.42,
        tokens=greedy_tokens,
        token_confidences=(0.95, 0.90, 0.85, 0.92),
    )
    entry = WordInferenceCacheEntry(
        clip_id="clip_001",
        word="coffee",
        greedy_tokens=greedy_tokens,
        greedy_text="khawi",
        token_confidences=(0.95, 0.90, 0.85, 0.92),
        top_hypotheses=(top_hyp,),
        mean_confidence=0.905,
        duration=0.42,
    )

    assert entry.clip_id == "clip_001"
    assert entry.word == "coffee"
    assert entry.greedy_text == "khawi"
    assert entry.mean_confidence == 0.905

    d = entry.to_dict()
    restored = WordInferenceCacheEntry.from_dict(d)
    assert restored == entry
    assert WordInferenceCacheEntry.from_json(entry.to_json()) == entry

    with pytest.raises(FrozenInstanceError):
        entry.greedy_text = "new"  # type: ignore[misc]


def test_word_inference_cache_entry_auto_mean_confidence():
    entry = WordInferenceCacheEntry(
        clip_id="clip_002",
        word="tea",
        greedy_tokens=(CherokeeToken("th"), CherokeeToken("i")),
        greedy_text="thi",
        token_confidences=(0.80, 0.90),
    )
    d = entry.to_dict()
    d["mean_confidence"] = 0.0
    restored = WordInferenceCacheEntry.from_dict(d)
    assert math.isclose(restored.mean_confidence, 0.85)


def test_inference_cache_manifest_methods_and_persistence(tmp_path: Path):
    entry1 = WordInferenceCacheEntry(
        clip_id="c1",
        word="coffee",
        greedy_tokens=(CherokeeToken("kh"), CherokeeToken("a")),
        greedy_text="kha",
        token_confidences=(0.9,),
        duration=0.3,
    )
    entry2 = WordInferenceCacheEntry(
        clip_id="c2",
        word="tea",
        greedy_tokens=(CherokeeToken("th"), CherokeeToken("i")),
        greedy_text="thi",
        token_confidences=(0.95,),
        duration=0.25,
    )
    manifest = InferenceCacheManifest(
        model_id="cherokee-wav2vec2-base-rev3",
        created_at="2026-09-18T12:00:00Z",
        entries=(entry1, entry2),
        metadata={"num_samples": 2, "sample_rate": 16000},
    )

    assert len(manifest) == 2
    assert list(manifest) == [entry1, entry2]
    assert manifest.get("c1") == entry1
    assert manifest.get("c2") == entry2
    assert manifest.get("c_unknown") is None
    assert manifest.by_clip_id == {"c1": entry1, "c2": entry2}

    # Serialization roundtrip
    d = manifest.to_dict()
    restored = InferenceCacheManifest.from_dict(d)
    assert restored == manifest

    # Dict-shaped entries input support
    dict_shaped = {
        "model_id": "test-model",
        "created_at": "2026-09-18",
        "entries": {"c1": entry1.to_dict(), "c2": entry2.to_dict()},
    }
    from_dict_manifest = InferenceCacheManifest.from_dict(dict_shaped)
    assert len(from_dict_manifest) == 2
    assert from_dict_manifest.get("c1") == entry1

    # File persistence roundtrip
    cache_path = tmp_path / "emissions_cache.json"
    manifest.save(cache_path)
    assert cache_path.exists()

    loaded = InferenceCacheManifest.load(cache_path)
    assert loaded == manifest


# ============================================================================
# 4. Acoustic Confusion Matrix Tests
# ============================================================================


def test_acoustic_confusion_matrix_create_and_queries(tmp_path: Path):
    probs = {
        "K": {"kh": 0.85, "k": 0.10, "h": 0.05},
        "AA": {"a": 0.90, "o": 0.10},
        "F": {"w": 0.70, "wh": 0.30},
    }
    ins_probs = {"a": 0.04, "i": 0.02}
    del_probs = {"K": 0.01, "T": 0.08}

    matrix = AcousticConfusionMatrix.create(
        model_id="cherokee-wav2vec2-base",
        probabilities=probs,
        insertion_probabilities=ins_probs,
        deletion_probabilities=del_probs,
        prune_threshold=0.05,
        iteration=3,
        metadata={"algorithm": "expectation-maximization"},
    )

    assert matrix.model_id == "cherokee-wav2vec2-base"
    assert matrix.iteration == 3
    assert "K" in matrix.arpabet_vocab
    assert "kh" in matrix.cherokee_vocab

    # Test probability query
    assert matrix.get_probability("K", "kh") == 0.85
    assert matrix.get_probability(ArpabetToken("K"), CherokeeToken("kh")) == 0.85
    assert matrix.get_probability("K", "unseen") == 0.0

    # Test log cost computation: -log(0.85) approx 0.1625
    expected_cost = -math.log(0.85)
    actual_cost = matrix.get_substitution_cost("K", "kh")
    assert math.isclose(actual_cost, expected_cost, rel_tol=1e-5)

    # Test unseen pair returns default cost
    assert (
        matrix.get_substitution_cost("K", "unseen") == matrix.default_substitution_cost
    )

    # Test insertion and deletion
    assert matrix.get_insertion_probability("a") == 0.04
    assert math.isclose(matrix.get_insertion_cost("a"), -math.log(0.04), rel_tol=1e-5)
    assert matrix.get_insertion_cost("unseen") == matrix.default_insertion_cost

    assert matrix.get_deletion_probability("K") == 0.01
    assert math.isclose(matrix.get_deletion_cost("K"), -math.log(0.01), rel_tol=1e-5)
    assert matrix.get_deletion_cost("unseen") == matrix.default_deletion_cost

    # Test argmax query best_cherokee_for
    best_c, best_p = matrix.best_cherokee_for("K")
    assert best_c == "kh"
    assert best_p == 0.85

    best_c_f, best_p_f = matrix.best_cherokee_for(ArpabetToken("F"))
    assert best_c_f == "w"
    assert best_p_f == 0.70

    # Test top candidates query
    top_cands = matrix.top_cherokee_candidates("K", top_k=2)
    assert len(top_cands) == 2
    assert top_cands[0] == ("kh", 0.85)
    assert top_cands[1] == ("k", 0.10)

    # Serialization roundtrip
    d = matrix.to_dict()
    restored = AcousticConfusionMatrix.from_dict(d)
    assert restored.model_id == matrix.model_id
    assert restored.probabilities == matrix.probabilities
    assert restored.log_costs == matrix.log_costs
    assert restored.iteration == matrix.iteration
    assert restored == matrix

    # File persistence roundtrip
    mat_file = tmp_path / "matrix.json"
    matrix.save(mat_file)
    assert mat_file.exists()
    loaded = AcousticConfusionMatrix.load(mat_file)
    assert loaded == matrix

    with pytest.raises(FrozenInstanceError):
        matrix.iteration = 4  # type: ignore[misc]


# ============================================================================
# 5. Synthetic Cherokee Target Tests
# ============================================================================


def test_synthetic_cherokee_target():
    target = SyntheticCherokeeTarget(
        source_word="coffee",
        arpabet_tokens=(
            ArpabetToken("K"),
            ArpabetToken("AA"),
            ArpabetToken("F"),
            ArpabetToken("IY"),
        ),
        cherokee_tokens=(
            CherokeeToken("kh"),
            CherokeeToken("a"),
            CherokeeToken("w"),
            CherokeeToken("i"),
        ),
        projected_tth="khawi",
        syllabary="ᎧᏫ",
        confidence_score=0.5355,
        per_token_probabilities=(0.85, 0.90, 0.70, 1.0),
    )

    assert target.source_word == "coffee"
    assert target.projected_tth == "khawi"
    assert target.syllabary == "ᎧᏫ"
    assert math.isclose(target.confidence_score, 0.5355)

    # Roundtrip serialization
    d = target.to_dict()
    restored = SyntheticCherokeeTarget.from_dict(d)
    assert restored == target

    j = target.to_json()
    assert SyntheticCherokeeTarget.from_json(j) == target

    with pytest.raises(FrozenInstanceError):
        target.projected_tth = "other"  # type: ignore[misc]


# ============================================================================
# 6. DP Traceback Models Tests
# ============================================================================


def test_aligned_token_pair():
    # Substitution
    pair_sub = AlignedTokenPair(
        arpabet=ArpabetToken("K"),
        cherokee=CherokeeToken("kh"),
        cost=0.16,
        confidence=0.95,
    )
    assert pair_sub.is_substitution is True
    assert pair_sub.is_insertion is False
    assert pair_sub.is_deletion is False

    # Insertion (epenthetic Cherokee vowel)
    pair_ins = AlignedTokenPair(
        arpabet=None,
        cherokee=CherokeeToken("i"),
        cost=3.21,
    )
    assert pair_ins.is_substitution is False
    assert pair_ins.is_insertion is True
    assert pair_ins.is_deletion is False

    # Deletion (dropped ARPAbet phoneme)
    pair_del = AlignedTokenPair(
        arpabet=ArpabetToken("T"),
        cherokee=None,
        cost=2.52,
    )
    assert pair_del.is_substitution is False
    assert pair_del.is_insertion is False
    assert pair_del.is_deletion is True

    # Roundtrip
    d = pair_sub.to_dict()
    assert AlignedTokenPair.from_dict(d) == pair_sub
    assert AlignedTokenPair.from_json(pair_sub.to_json()) == pair_sub

    d_ins = pair_ins.to_dict()
    assert AlignedTokenPair.from_dict(d_ins) == pair_ins


def test_traceback_alignment_result():
    pair1 = AlignedTokenPair(
        arpabet=ArpabetToken("K"), cherokee=CherokeeToken("kh"), cost=0.1
    )
    pair2 = AlignedTokenPair(
        arpabet=ArpabetToken("AA"), cherokee=CherokeeToken("a"), cost=0.2
    )
    res = TracebackAlignmentResult(
        pairs=(pair1, pair2),
        total_cost=0.3,
        normalized_cost=0.15,
    )

    assert len(res.pairs) == 2
    assert res.total_cost == 0.3
    assert res.normalized_cost == 0.15

    d = res.to_dict()
    restored = TracebackAlignmentResult.from_dict(d)
    assert restored == res
    assert TracebackAlignmentResult.from_json(res.to_json()) == res


# ============================================================================
# 7. Pure Mapping Protocols Runtime Checks
# ============================================================================


class DummyG2PExtractor:
    def __call__(
        self, text: str, strip_stress: bool = True
    ) -> tuple[ArpabetToken, ...]:
        return (ArpabetToken("K"),)

    def extract(self, text: str, strip_stress: bool = True) -> tuple[ArpabetToken, ...]:
        return (ArpabetToken("K"),)


class DummyTracebackAligner:
    def align(
        self,
        arpabet_tokens,
        cherokee_tokens,
        matrix: AcousticConfusionMatrix,
    ) -> TracebackAlignmentResult:
        return TracebackAlignmentResult(pairs=(), total_cost=0.0, normalized_cost=0.0)


class DummyDPAccumulator:
    def accumulate(self, alignment: TracebackAlignmentResult) -> None:
        pass


class DummyProjector:
    def project_word(
        self, word: str, matrix: Optional[AcousticConfusionMatrix] = None
    ) -> SyntheticCherokeeTarget:
        return SyntheticCherokeeTarget(
            source_word=word,
            arpabet_tokens=(),
            cherokee_tokens=(),
            projected_tth="",
        )

    def project_arpabet(
        self,
        arpabet_tokens,
        matrix: Optional[AcousticConfusionMatrix] = None,
        source_word: str = "",
    ) -> SyntheticCherokeeTarget:
        return SyntheticCherokeeTarget(
            source_word=source_word,
            arpabet_tokens=tuple(arpabet_tokens),
            cherokee_tokens=(),
            projected_tth="",
        )

    def project_english_text(
        self,
        text: str,
        matrix: Optional[AcousticConfusionMatrix] = None,
    ) -> str:
        return ""


def test_runtime_protocols():
    g2p = DummyG2PExtractor()
    assert isinstance(g2p, EnglishToArpabetProtocol)
    assert isinstance(g2p, G2PExtractorProtocol)

    aligner = DummyTracebackAligner()
    assert isinstance(aligner, TracebackAlignerProtocol)

    accum = DummyDPAccumulator()
    assert isinstance(accum, DPTracebackAccumulatorProtocol)

    projector = DummyProjector()
    assert isinstance(projector, SyntheticTargetProjectorProtocol)


def test_constants():
    assert len(STANDARD_ARPABET_PHONEMES) == 39
    assert len(STANDARD_ARPABET_VOWELS) == 15
    assert len(STANDARD_ARPABET_CONSONANTS) == 24
    assert "AA" in STANDARD_ARPABET_VOWELS
    assert "K" in STANDARD_ARPABET_CONSONANTS

    assert len(CANONICAL_CHEROKEE_VOWELS) == 6
    assert "a" in CANONICAL_CHEROKEE_VOWELS
    assert "kh" in CANONICAL_CHEROKEE_CONSONANTS
    assert "tsh" in CANONICAL_CHEROKEE_CONSONANTS
    assert "d" not in CANONICAL_CHEROKEE_CONSONANTS
    assert "g" not in CANONICAL_CHEROKEE_CONSONANTS
