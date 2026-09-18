# -*- coding: utf-8 -*-
"""
transcription.alignment.tests.test_arpabet_matrix

Comprehensive unit tests for articulatory seed matrix initialization,
Numba-accelerated Wagner-Fischer DP alignment, iterative Expectation-Maximization
(EM) frequency accumulation, transition probability pruning, and matrix persistence.
"""

from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
from typing import List, Tuple
import pytest

from transcription.alignment.arpabet import (
    CANONICAL_CHEROKEE_CONSONANTS,
    CANONICAL_CHEROKEE_PHONEMES,
    CANONICAL_CHEROKEE_VOWELS,
    STANDARD_ARPABET_CONSONANTS,
    STANDARD_ARPABET_PHONEMES,
    STANDARD_ARPABET_VOWELS,
    AcousticConfusionMatrix,
    AlignedTokenPair,
    ArpabetToken,
    CherokeeToken,
    InferenceCacheManifest,
    TracebackAlignerProtocol,
    TracebackAlignmentResult,
    WagnerFischerAligner,
    WordInferenceCacheEntry,
    WordManifestEntry,
    align_word_pair,
    build_articulatory_seed_matrix,
    get_articulatory_distance,
    train_acoustic_confusion_matrix,
)
from transcription.alignment.arpabet.matrix import parse_args, main

# ============================================================================
# 1. Articulatory Seed Matrix & Distance Tests
# ============================================================================


def test_articulatory_distance_place_manner():
    """Verify place and manner distances reflect articulatory phonetics."""
    # Vowels matching close vowels
    d_aa_a = get_articulatory_distance("AA", "a")
    d_aa_o = get_articulatory_distance("AA", "o")
    d_aa_i = get_articulatory_distance("AA", "i")
    assert d_aa_a < d_aa_o < d_aa_i

    # Stops matching stops
    d_t_th = get_articulatory_distance("T", "th")
    d_t_t = get_articulatory_distance("T", "t")
    d_t_k = get_articulatory_distance("T", "k")
    assert d_t_th <= 0.3
    assert d_t_t <= 0.3
    assert d_t_k > d_t_t

    # Voiced stops (Cherokee has no b/d/g)
    d_d_t = get_articulatory_distance("D", "t")
    assert d_d_t <= 0.3
    d_g_k = get_articulatory_distance("G", "k")
    assert d_g_k <= 0.3

    # Affricates / Sibilants
    d_ch_tsh = get_articulatory_distance("CH", "tsh")
    d_jh_ts = get_articulatory_distance("JH", "ts")
    d_s_s = get_articulatory_distance("S", "s")
    d_s_hs = get_articulatory_distance("S", "hs")
    assert d_ch_tsh <= 0.3
    assert d_jh_ts <= 0.3
    assert d_s_s <= 0.3
    assert d_s_hs <= 0.3

    # Nasals
    d_m_m = get_articulatory_distance("M", "m")
    d_n_n = get_articulatory_distance("N", "n")
    assert d_m_m <= 0.3
    assert d_n_n <= 0.3

    # Liquids
    d_l_l = get_articulatory_distance("L", "l")
    assert d_l_l <= 0.3

    # Glides
    d_w_w = get_articulatory_distance("W", "w")
    d_y_y = get_articulatory_distance("Y", "y")
    d_hh_h = get_articulatory_distance("HH", "h")
    assert d_w_w <= 0.3
    assert d_y_y <= 0.3
    assert d_hh_h <= 0.3

    # Cross-category penalty (vowel to consonant or vice-versa)
    d_cross_1 = get_articulatory_distance("AA", "t")
    d_cross_2 = get_articulatory_distance("T", "a")
    assert d_cross_1 == 2.4
    assert d_cross_2 == 2.4


def test_build_articulatory_seed_matrix_properties():
    """Verify seed matrix creates valid non-negative finite cost distributions."""
    seed = build_articulatory_seed_matrix(model_id="test_seed")
    assert seed.model_id == "test_seed"
    assert len(seed.arpabet_vocab) == len(STANDARD_ARPABET_PHONEMES)
    assert len(seed.cherokee_vocab) == len(CANONICAL_CHEROKEE_PHONEMES)

    # Every ARPAbet phone has valid probability distribution
    for a in seed.arpabet_vocab:
        probs = seed.probabilities[a]
        assert len(probs) == len(seed.cherokee_vocab)
        total_p = sum(probs.values())
        assert pytest.approx(total_p, abs=1e-5) == 1.0

        for c, p in probs.items():
            assert p > 0.0
            cost = seed.get_substitution_cost(a, c)
            assert cost >= 0.0
            assert math.isfinite(cost)
            # Cost equals -log P
            assert pytest.approx(cost, abs=1e-5) == -math.log(p)

    # No degenerate vowel collapse:
    # T -> th should be much cheaper than T -> a
    cost_t_th = seed.get_substitution_cost("T", "th")
    cost_t_a = seed.get_substitution_cost("T", "a")
    assert cost_t_th < cost_t_a
    assert (cost_t_a - cost_t_th) > 3.0

    # AA -> a should be much cheaper than AA -> t
    cost_aa_a = seed.get_substitution_cost("AA", "a")
    cost_aa_t = seed.get_substitution_cost("AA", "t")
    assert cost_aa_a < cost_aa_t
    assert (cost_aa_t - cost_aa_a) > 3.0

    # Insertions and deletions
    assert sum(seed.insertion_probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    for c, p in seed.insertion_probabilities.items():
        assert p > 0.0
        ins_cost = seed.get_insertion_cost(c)
        assert ins_cost >= 0.0
        assert math.isfinite(ins_cost)

    for a, p in seed.deletion_probabilities.items():
        assert 0.0 < p < 1.0
        del_cost = seed.get_deletion_cost(a)
        assert del_cost >= 0.0
        assert math.isfinite(del_cost)


# ============================================================================
# 2. Dynamic Programming Alignment Tests
# ============================================================================


def test_align_word_pair_exact_match():
    """Verify DP alignment correctly aligns phonetically matched pairs."""
    seed = build_articulatory_seed_matrix()

    # Align "CAT" (K AE T) to "kheti" (kh e t i)
    arp = ["K", "AE", "T"]
    chr_phones = ["kh", "e", "t"]
    confs = [0.95, 0.90, 0.88]

    res = align_word_pair(arp, chr_phones, seed, token_confidences=confs)
    assert isinstance(res, TracebackAlignmentResult)
    assert len(res.pairs) == 3
    assert res.total_cost > 0.0
    assert res.normalized_cost == pytest.approx(res.total_cost / 3.0, abs=1e-6)

    # Check pair properties
    p0, p1, p2 = res.pairs
    assert p0.arpabet is not None and p0.arpabet.phone == "K"
    assert p0.cherokee is not None and p0.cherokee.phone == "kh"
    assert p0.confidence == 0.95
    assert p0.is_substitution is True

    assert p1.arpabet is not None and p1.arpabet.phone == "AE"
    assert p1.cherokee is not None and p1.cherokee.phone == "e"
    assert p1.confidence == 0.90

    assert p2.arpabet is not None and p2.arpabet.phone == "T"
    assert p2.cherokee is not None and p2.cherokee.phone == "t"
    assert p2.confidence == 0.88


def test_align_word_pair_insertion_and_deletion():
    """Verify DP alignment handles epenthetic insertions and coda deletions."""
    seed = build_articulatory_seed_matrix()

    # English word ending with coda L: "TEST" -> "th e hs" (T deleted, i inserted)
    arp = ["T", "EH", "S", "T"]
    chr_phones = ["th", "e", "hs", "i"]  # final T dropped, epenthetic 'i' added
    confs = [0.95, 0.92, 0.90, 0.85]

    res = align_word_pair(arp, chr_phones, seed, token_confidences=confs)

    # Should contain at least one insertion or deletion
    insertions = [p for p in res.pairs if p.is_insertion]
    deletions = [p for p in res.pairs if p.is_deletion]
    substitutions = [p for p in res.pairs if p.is_substitution]

    assert len(substitutions) >= 2
    assert len(insertions) + len(deletions) >= 1

    for ins in insertions:
        assert ins.arpabet is None
        assert ins.cherokee is not None
        assert ins.cost == pytest.approx(
            seed.get_insertion_cost(ins.cherokee), abs=1e-5
        )

    for del_step in deletions:
        assert del_step.arpabet is not None
        assert del_step.cherokee is None
        assert del_step.cost == pytest.approx(
            seed.get_deletion_cost(del_step.arpabet), abs=1e-5
        )


def test_align_word_pair_boundary_cases():
    """Verify DP aligner handles empty inputs safely."""
    seed = build_articulatory_seed_matrix()

    # Both empty
    res_empty = align_word_pair([], [], seed)
    assert len(res_empty.pairs) == 0
    assert res_empty.total_cost == 0.0
    assert res_empty.normalized_cost == 0.0

    # ARPAbet empty (all insertions)
    res_ins = align_word_pair(
        [], ["a", "th", "i"], seed, token_confidences=[0.9, 0.8, 0.7]
    )
    assert len(res_ins.pairs) == 3
    assert all(p.is_insertion for p in res_ins.pairs)
    assert res_ins.pairs[0].cherokee is not None
    assert res_ins.pairs[0].cherokee.phone == "a"
    assert res_ins.pairs[0].confidence == 0.9

    # Cherokee empty (all deletions)
    res_del = align_word_pair(["T", "AA"], [], seed)
    assert len(res_del.pairs) == 2
    assert all(p.is_deletion for p in res_del.pairs)
    assert res_del.pairs[0].arpabet is not None
    assert res_del.pairs[0].arpabet.phone == "T"
    assert res_del.pairs[0].confidence == 1.0


def test_wagner_fischer_aligner_protocol():
    """Verify WagnerFischerAligner implements TracebackAlignerProtocol."""
    aligner = WagnerFischerAligner()
    assert isinstance(aligner, TracebackAlignerProtocol)

    seed = build_articulatory_seed_matrix()
    res = aligner.align(["T"], ["th"], seed)
    assert len(res.pairs) == 1
    assert res.pairs[0].arpabet is not None
    assert res.pairs[0].cherokee is not None
    assert res.pairs[0].arpabet.phone == "T"
    assert res.pairs[0].cherokee.phone == "th"


# ============================================================================
# 3. Iterative EM Frequency Accumulator Tests
# ============================================================================


def test_em_matrix_convergence_and_pruning():
    """Verify EM frequency accumulation converges probabilities and prunes low-density mappings."""
    seed = build_articulatory_seed_matrix()

    # Create synthetic dataset with repetitive mappings:
    # "T" consistently maps to "th" with high confidence
    # "AA" consistently maps to "a"
    # "S" consistently maps to "hs"
    # An epenthetic "i" is inserted at the end
    word_entries: List[WordManifestEntry] = []
    emissions_entries: List[WordInferenceCacheEntry] = []

    for i in range(25):
        clip_id = f"test_clip_{i:03d}"
        word_entries.append(
            WordManifestEntry(
                clip_id=clip_id,
                audio_path=f"audio/{clip_id}.wav",
                word="TEST",
                duration=0.5,
                arpabet=(
                    ArpabetToken("T"),
                    ArpabetToken("AA"),
                    ArpabetToken("S"),
                ),
            )
        )
        emissions_entries.append(
            WordInferenceCacheEntry(
                clip_id=clip_id,
                word="TEST",
                greedy_tokens=(
                    CherokeeToken("th"),
                    CherokeeToken("a"),
                    CherokeeToken("hs"),
                    CherokeeToken("i"),  # Epenthetic insertion
                ),
                greedy_text="thahsi",
                token_confidences=(0.95, 0.98, 0.92, 0.85),
                mean_confidence=0.925,
                duration=0.5,
            )
        )

    cache = InferenceCacheManifest(
        model_id="synthetic_test_model",
        created_at="2026-09-18T12:00:00Z",
        entries=tuple(emissions_entries),
    )

    trained = train_acoustic_confusion_matrix(
        manifest_entries=word_entries,
        emissions_manifest=cache,
        num_iterations=3,
        prune_threshold=0.05,
        seed_matrix=seed,
    )

    assert trained.iteration == 3
    assert trained.model_id == "synthetic_test_model"

    # Verify probability shifts towards empirical observations
    p_t_th = trained.get_probability("T", "th")
    assert p_t_th > 0.60, f"Expected P(th|T) > 0.60, got {p_t_th}"

    p_aa_a = trained.get_probability("AA", "a")
    assert p_aa_a > 0.80, f"Expected P(a|AA) > 0.80, got {p_aa_a}"

    p_s_hs = trained.get_probability("S", "hs")
    assert p_s_hs > 0.60, f"Expected P(hs|S) > 0.60, got {p_s_hs}"

    # Verify pruning (< 5% pruned to 0.0)
    for a in trained.arpabet_vocab:
        probs = trained.probabilities[a]
        # Any surviving probability in the map must be >= 0.05
        for c, p in probs.items():
            assert p >= 0.05
        # Probabilities sum to 1.0
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)

    # Pruned sound cost falls back to default substitution cost (10.0)
    assert trained.get_probability("AA", "tlh") == 0.0
    assert (
        trained.get_substitution_cost("AA", "tlh") == trained.default_substitution_cost
    )


def test_em_delta_shrinkage_stability():
    """Verify EM delta shrinks across successive cycles showing convergence stability."""
    seed = build_articulatory_seed_matrix()

    word_entries: List[WordManifestEntry] = []
    emissions_entries: List[WordInferenceCacheEntry] = []

    for i in range(20):
        clip_id = f"clip_{i:02d}"
        word_entries.append(
            WordManifestEntry(
                clip_id=clip_id,
                audio_path=f"audio/{clip_id}.wav",
                word="SAMPLE",
                duration=0.4,
                arpabet=(ArpabetToken("S"), ArpabetToken("AE")),
            )
        )
        emissions_entries.append(
            WordInferenceCacheEntry(
                clip_id=clip_id,
                word="SAMPLE",
                greedy_tokens=(CherokeeToken("s"), CherokeeToken("e")),
                greedy_text="se",
                token_confidences=(0.95, 0.90),
                mean_confidence=0.925,
                duration=0.4,
            )
        )

    cache = InferenceCacheManifest(
        model_id="delta_test_model",
        created_at="2026-09-18T12:00:00Z",
        entries=tuple(emissions_entries),
    )

    trained_1 = train_acoustic_confusion_matrix(
        manifest_entries=word_entries,
        emissions_manifest=cache,
        num_iterations=1,
        seed_matrix=seed,
    )
    trained_3 = train_acoustic_confusion_matrix(
        manifest_entries=word_entries,
        emissions_manifest=cache,
        num_iterations=3,
        seed_matrix=seed,
    )

    # Final delta at iteration 3 should be smaller than initial delta at iteration 1
    delta_1 = trained_1.metadata.get("final_delta", 1.0)
    delta_3 = trained_3.metadata.get("final_delta", 1.0)
    assert delta_3 < delta_1


# ============================================================================
# 4. Serialization and Persistence Tests
# ============================================================================


def test_matrix_serialization_roundtrip(tmp_path: Path):
    """Verify AcousticConfusionMatrix roundtrip JSON persistence and recovery."""
    seed = build_articulatory_seed_matrix(model_id="serialize_test_model")
    save_file = tmp_path / "matrix.json"

    seed.save(save_file)
    assert save_file.exists()

    loaded = AcousticConfusionMatrix.load(save_file)
    assert loaded.model_id == seed.model_id
    assert loaded.arpabet_vocab == seed.arpabet_vocab
    assert loaded.cherokee_vocab == seed.cherokee_vocab
    assert loaded.iteration == seed.iteration
    assert loaded.prune_threshold == seed.prune_threshold

    # Cost equality
    for a in seed.arpabet_vocab[:5]:
        for c in seed.cherokee_vocab[:5]:
            assert seed.get_substitution_cost(a, c) == pytest.approx(
                loaded.get_substitution_cost(a, c), abs=1e-6
            )


# ============================================================================
# 5. CLI Runner Tests
# ============================================================================


def test_cli_parse_args(monkeypatch):
    """Verify CLI argument parsing accepts options."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "matrix.py",
            "--manifest-path",
            "data/arpabet_alignment/words_manifest.json",
            "--num-iterations",
            "5",
            "--prune-threshold",
            "0.08",
        ],
    )
    args = parse_args()
    assert args.manifest_path == "data/arpabet_alignment/words_manifest.json"
    assert args.num_iterations == 5
    assert args.prune_threshold == 0.08
