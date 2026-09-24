# -*- coding: utf-8 -*-
"""
Tests for language-agnostic codeswitching types, G2P, matrix, trainer, and projector
in digohwelisgi.core.codeswitching.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from digohwelisgi.core.codeswitching import (
    EPSILON_TOKEN,
    STANDARD_ARPABET_CONSONANTS,
    STANDARD_ARPABET_PHONEMES,
    STANDARD_ARPABET_VOWELS,
    AcousticConfusionMatrix,
    AlignmentPath,
    ARPAbetPhone,
    BeamHypothesis,
    ConfusionEntry,
    ConfusionMatrixTrainer,
    CrossEntropyLoss,
    EnglishToArpabetProtocol,
    G2PEngine,
    GenericConfusionMatrixProtocol,
    GenericSyntheticTargetProjectorProtocol,
    JointNgram,
    ProjectedTarget,
    StressPattern,
    SubstitutionMapping,
    SyntheticTargetProjector,
    TargetPhone,
    TracebackResult,
    align_word_pair_generic,
    bucket_by_duration,
    ctc_prefix_beam_search,
    extract_arpabet,
    generate_static_dictionary,
    get_default_g2p,
    normalize_source_key,
    normalize_target_key,
    sanitize_model_id,
)


class TestARPAbetPhone:
    def test_phone_normalization(self):
        phone = ARPAbetPhone("ah0")
        assert phone.phone == "AH"
        assert phone.stress == 0
        assert phone.is_vowel is True
        assert phone.is_epsilon is False

        phone2 = ARPAbetPhone("K")
        assert phone2.phone == "K"
        assert phone2.stress is None
        assert phone2.is_vowel is False

    def test_epsilon_handling(self):
        eps = ARPAbetPhone("<eps>")
        assert eps.is_epsilon is True
        assert eps.phone == EPSILON_TOKEN

        eps2 = ARPAbetPhone("EPS")
        assert eps2.is_epsilon is True

    def test_serialization(self):
        tok = ARPAbetPhone("EY1")
        data = tok.to_dict()
        assert data["phone"] == "EY"
        assert data["stress"] == 1
        loaded = ARPAbetPhone.from_dict(data)
        assert loaded == tok


class TestTargetPhone:
    def test_target_phone(self):
        t = TargetPhone("k")
        assert t.phone == "k"
        assert t.is_epsilon is False

        eps = TargetPhone("<eps>")
        assert eps.is_epsilon is True
        assert eps.phone == EPSILON_TOKEN


class TestG2PEngine:
    def test_g2p_extraction(self):
        engine = G2PEngine()
        tokens = engine.extract("coffee", strip_stress=True)
        assert len(tokens) > 0
        assert all(isinstance(t, ARPAbetPhone) for t in tokens)
        assert all(t.stress is None for t in tokens)
        assert all(t.phone in STANDARD_ARPABET_PHONEMES for t in tokens)

    def test_g2p_with_stress(self):
        engine = G2PEngine()
        tokens = engine.extract("coffee", strip_stress=False)
        assert len(tokens) > 0
        # At least one vowel should have a stress number
        vowels = [t for t in tokens if t.is_vowel]
        assert any(v.stress is not None for v in vowels)

    def test_convenience_extract_arpabet(self):
        tokens = extract_arpabet("hello")
        assert len(tokens) > 0


class TestAcousticConfusionMatrix:
    @pytest.fixture
    def simple_matrix(self) -> AcousticConfusionMatrix:
        probs = {
            "K": {"k": 0.8, "kh": 0.2},
            "AA": {"a": 0.9, "o": 0.1},
            "S T": {"st": 0.7, "t": 0.3},
        }
        ins_probs = {"i": 0.5, "a": 0.5}
        del_probs = {"T": 0.2, "D": 0.3}
        return AcousticConfusionMatrix.create(
            model_id="test_matrix",
            probabilities=probs,
            insertion_probabilities=ins_probs,
            deletion_probabilities=del_probs,
            source_vocab=["K", "AA", "S T", "T", "D"],
            target_vocab=["k", "kh", "a", "o", "st", "t", "i"],
        )

    def test_costs_and_probabilities(self, simple_matrix):
        assert simple_matrix.get_probability("K", "k") == 0.8
        assert simple_matrix.get_probability("K", "kh") == 0.2
        assert simple_matrix.get_probability("K", "z") == 0.0

        cost = simple_matrix.get_substitution_cost("K", "k")
        assert cost > 0.0
        assert cost < simple_matrix.get_substitution_cost("K", "kh")

        assert simple_matrix.best_target_for("K") == ("k", 0.8)
        assert simple_matrix.best_target_for("S T") == ("st", 0.7)

    def test_serialization(self, simple_matrix, tmp_path):
        json_file = tmp_path / "matrix.json"
        simple_matrix.save(json_file)
        loaded = AcousticConfusionMatrix.load(json_file)
        assert loaded.model_id == simple_matrix.model_id
        assert loaded.probabilities == simple_matrix.probabilities

        npz_file = tmp_path / "matrix.npz"
        simple_matrix.save_npz(npz_file)
        loaded_npz = AcousticConfusionMatrix.load_npz(npz_file)
        assert loaded_npz.model_id == simple_matrix.model_id
        assert loaded_npz.probabilities == simple_matrix.probabilities


class TestSyntheticTargetProjector:
    @pytest.fixture
    def matrix_and_projector(self) -> SyntheticTargetProjector:
        probs = {
            "K": {"k": 0.8, "kh": 0.2},
            "AA": {"a": 0.9, "o": 0.1},
            "F": {"f": 0.8, "v": 0.2},
            "IY": {"i": 0.9, "e": 0.1},
        }
        matrix = AcousticConfusionMatrix.create(
            model_id="toy_model",
            probabilities=probs,
        )
        return SyntheticTargetProjector(matrix=matrix)

    def test_project_arpabet(self, matrix_and_projector):
        tokens = (
            ARPAbetPhone("K"),
            ARPAbetPhone("AA"),
            ARPAbetPhone("F"),
            ARPAbetPhone("IY"),
        )
        proj = matrix_and_projector.project_arpabet(tokens, source_word="coffee")
        assert proj.source_word == "coffee"
        assert proj.projected_text == "kafi"
        assert len(proj.target_tokens) == 4
        assert proj.confidence_score > 0.5

    def test_static_dictionary(self, tmp_path):
        probs = {"K": {"k": 1.0}}
        matrix = AcousticConfusionMatrix.create(model_id="toy", probabilities=probs)
        dict_path = tmp_path / "dict.json"
        generate_static_dictionary(["cat"], matrix, dict_path)

        projector = SyntheticTargetProjector(matrix=matrix, dictionary=dict_path)
        proj = projector.project_word("cat")
        assert proj.source_word == "cat"


class TestConfusionMatrixTrainer:
    def test_em_training(self):
        seed_probs = {
            "K": {"k": 0.5, "kh": 0.5},
            "T": {"t": 0.5, "th": 0.5},
        }
        seed = AcousticConfusionMatrix.create(
            model_id="seed",
            probabilities=seed_probs,
            source_vocab=["K", "T"],
            target_vocab=["k", "kh", "t", "th"],
        )

        pairs = [
            (["K"], ["k"], [0.95]),
            (["K"], ["k"], [0.90]),
            (["K"], ["kh"], [0.85]),
            (["T"], ["t"], [0.99]),
        ]

        trainer = ConfusionMatrixTrainer(num_iterations=2)
        trained = trainer.train(pairs, seed, model_id="trained_em")
        assert trained.get_probability("K", "k") > trained.get_probability("K", "kh")
        assert trained.get_probability("T", "t") > trained.get_probability("T", "th")

    def test_bucket_by_duration(self):
        items = [
            {"duration": 1.5},
            {"duration": 0.2},
            {"duration": 3.0},
            {"duration": 0.5},
        ]
        batches = bucket_by_duration(items, batch_size=2)
        assert len(batches) == 2
        assert batches[0][0]["duration"] == 0.2
        assert batches[0][1]["duration"] == 0.5
        assert batches[1][0]["duration"] == 1.5
        assert batches[1][1]["duration"] == 3.0

    def test_ctc_prefix_beam_search(self):
        vocab = ["<pad>", "a", "b"]
        logits = np.array(
            [
                [5.0, 0.0, 0.0],  # pad
                [0.0, 5.0, 0.0],  # a
                [5.0, 0.0, 0.0],  # pad
                [0.0, 0.0, 5.0],  # b
            ]
        )
        hyps = ctc_prefix_beam_search(logits, vocab, pad_id=0, beam_width=5, top_k=1)
        assert len(hyps) == 1
        assert hyps[0].text == "ab"
