# -*- coding: utf-8 -*-
"""
digohwelisgi.evaluation.tests.test_evaluator

Unit and integration tests for EvaluationRecord schema, NoisyEvaluator,
CTC peak and top-K extraction, waveform evaluation, and JSONL streaming resume.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from digohwelisgi.evaluation.evaluator import (
    EvaluationRecord,
    NoisyEvaluator,
    calculate_cer,
)
from digohwelisgi.evaluation.perturbations import AdditiveNoise


class DummyTokenizer:
    """Mock Wav2Vec2 tokenizer for unit testing."""

    def __init__(self) -> None:
        self.vocab = {
            "[PAD]": 0,
            "|": 1,
            "a": 2,
            "t": 3,
            "s": 4,
            "i": 5,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 0
        self.word_delimiter_token_id = 1

    def decode(self, token_ids: list[int]) -> str:
        res = []
        for tid in token_ids:
            tok = self.inv_vocab.get(tid, "")
            if tok not in ("[PAD]", "|"):
                res.append(tok)
        return "".join(res)


class DummyProcessor:
    """Mock Wav2Vec2 processor for unit testing."""

    def __init__(self) -> None:
        self.tokenizer = DummyTokenizer()

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)


class DummyASRModel:
    """Mock CherokeeASRModel returning predictable softmax probabilities."""

    def __init__(self, processor: DummyProcessor, fixed_probs: np.ndarray) -> None:
        self.processor = processor
        self.fixed_probs = fixed_probs
        self.call_count = 0

    def get_probabilities(
        self, waveform: torch.Tensor, sample_rate: int = 16000
    ) -> torch.Tensor:
        self.call_count += 1
        return torch.tensor(self.fixed_probs, dtype=torch.float32)


# ============================================================================
# 1. Tests for EvaluationRecord Schema
# ============================================================================


def test_evaluation_record_schema() -> None:
    rec = EvaluationRecord(
        audio_id="utt_001",
        reference="atsi",
        snr_tier=15.0,
        noise_type="white",
        hypothesis="atsi",
        top_k_tokens=[
            [("a", 0.9), ("t", 0.05)],
            [("t", 0.85), ("s", 0.1)],
            [("s", 0.8), ("i", 0.1)],
            [("i", 0.95), ("a", 0.02)],
        ],
        cer=0.0,
        metadata={"duration_sec": 1.5},
    )

    assert rec.audio_id == "utt_001"
    assert rec.reference == "atsi"
    assert rec.snr_tier == 15.0
    assert rec.noise_type == "white"
    assert rec.hypothesis == "atsi"
    assert len(rec.top_k_tokens) == 4
    assert rec.cer == 0.0
    assert rec.metadata == {"duration_sec": 1.5}

    json_str = rec.model_dump_json()
    loaded_rec = EvaluationRecord.model_validate_json(json_str)
    assert loaded_rec.audio_id == rec.audio_id
    assert loaded_rec.cer == rec.cer
    assert loaded_rec.top_k_tokens == rec.top_k_tokens


# ============================================================================
# 2. Tests for calculate_cer
# ============================================================================


def test_calculate_cer_exact_match() -> None:
    assert calculate_cer("athaleniskv", "athaleniskv") == 0.0


def test_calculate_cer_mismatch() -> None:
    cer = calculate_cer("abc", "abd")
    assert pytest.approx(cer, abs=1e-3) == 1.0 / 3.0


def test_calculate_cer_empty() -> None:
    assert calculate_cer("", "") == 0.0
    assert calculate_cer("abc", "") == 1.0


# ============================================================================
# 3. Tests for extract_peaks_and_topk
# ============================================================================


def test_extract_peaks_and_topk() -> None:
    processor = DummyProcessor()
    # Vocab: [PAD]:0, '|':1, 'a':2, 't':3, 's':4, 'i':5
    # Let's create a sequence of 7 frames:
    # Frame 0: [PAD] (id 0)
    # Frame 1: 'a' (id 2)
    # Frame 2: 'a' (id 2, repeated -> collapsed)
    # Frame 3: '|' (id 1, word delim -> space)
    # Frame 4: 't' (id 3)
    # Frame 5: [PAD] (id 0)
    # Frame 6: 's' (id 4)
    probs = np.zeros((7, 6), dtype=np.float32)
    probs[0, 0] = 0.9  # PAD
    probs[1, 2] = 0.85  # a
    probs[1, 3] = 0.10  # alt t
    probs[2, 2] = 0.90  # a (repeat)
    probs[3, 1] = 0.95  # |
    probs[4, 3] = 0.70  # t
    probs[4, 4] = 0.20  # alt s
    probs[5, 0] = 0.99  # PAD
    probs[6, 4] = 0.80  # s
    probs[6, 5] = 0.15  # alt i

    evaluator = NoisyEvaluator(model=MagicMock(), top_k=2)
    evaluator.processor = processor

    hyp, top_k = evaluator.extract_peaks_and_topk(probs, processor=processor, top_k=2)
    assert hyp == "a ts"
    assert len(top_k) == 4  # 'a', ' ', 't', 's'
    # Check top_k for 'a'
    assert top_k[0][0][0] == "a"
    assert pytest.approx(top_k[0][0][1], abs=1e-2) == 0.85
    assert top_k[0][1][0] == "t"
    # Check top_k for ' '
    assert top_k[1][0][0] == " "
    # Check top_k for 't'
    assert top_k[2][0][0] == "t"
    # Check top_k for 's'
    assert top_k[3][0][0] == "s"


# ============================================================================
# 4. Tests for evaluate_waveform
# ============================================================================


def test_evaluate_waveform() -> None:
    processor = DummyProcessor()
    # Emits 'a', 't'
    probs = np.zeros((3, 6), dtype=np.float32)
    probs[0, 2] = 0.9  # a
    probs[1, 0] = 0.9  # PAD
    probs[2, 3] = 0.9  # t

    dummy_model = DummyASRModel(processor=processor, fixed_probs=probs)
    evaluator = NoisyEvaluator(model=dummy_model, top_k=3)

    waveform = torch.zeros(16000, dtype=torch.float32)
    noise = AdditiveNoise(snr_db=20.0, noise_type="white")

    record = evaluator.evaluate_waveform(
        waveform=waveform,
        reference="at",
        audio_id="test_audio_1",
        sample_rate=16000,
        snr_tier=20.0,
        noise_type="white",
        perturbation=noise,
    )

    assert isinstance(record, EvaluationRecord)
    assert record.audio_id == "test_audio_1"
    assert record.hypothesis == "at"
    assert record.reference == "at"
    assert record.cer == 0.0
    assert record.snr_tier == 20.0
    assert record.noise_type == "white"
    assert len(record.top_k_tokens) == 2


# ============================================================================
# 5. Tests for stream_evaluate_dataset and Resume Checkpointing
# ============================================================================


def test_stream_evaluate_dataset_with_resume(tmp_path: Path) -> None:
    processor = DummyProcessor()
    probs = np.zeros((2, 6), dtype=np.float32)
    probs[0, 2] = 0.95  # a
    probs[1, 3] = 0.95  # t

    dummy_model = DummyASRModel(processor=processor, fixed_probs=probs)
    evaluator = NoisyEvaluator(model=dummy_model, top_k=2)

    samples = [
        {"audio_id": "utt_1", "reference": "at", "waveform": torch.zeros(16000)},
        {"audio_id": "utt_2", "reference": "at", "waveform": torch.zeros(16000)},
    ]

    perturbations = [
        (25.0, "white", None),
        (10.0, "white", AdditiveNoise(snr_db=10.0, noise_type="white")),
    ]

    jsonl_path = tmp_path / "eval_records.jsonl"

    # 1. First run: 2 samples x 2 tiers = 4 records
    records_pass1 = evaluator.stream_evaluate_dataset(
        samples=samples,
        perturbations_by_tier=perturbations,
        output_jsonl_path=jsonl_path,
        resume=True,
    )
    assert len(records_pass1) == 4
    assert jsonl_path.exists()
    assert dummy_model.call_count == 4

    # Verify JSONL lines count
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 4

    # 2. Second run with resume=True and identical samples -> should skip all 4 evaluations
    records_pass2 = evaluator.stream_evaluate_dataset(
        samples=samples,
        perturbations_by_tier=perturbations,
        output_jsonl_path=jsonl_path,
        resume=True,
    )
    assert len(records_pass2) == 4
    assert dummy_model.call_count == 4  # No additional calls!

    # 3. Third run: add a 3rd sample -> only the 2 new records are evaluated and appended
    samples_extended = list(samples) + [
        {"audio_id": "utt_3", "reference": "at", "waveform": torch.zeros(16000)},
    ]
    records_pass3 = evaluator.stream_evaluate_dataset(
        samples=samples_extended,
        perturbations_by_tier=perturbations,
        output_jsonl_path=jsonl_path,
        resume=True,
    )
    assert len(records_pass3) == 6
    assert dummy_model.call_count == 6  # Only 2 additional calls!

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 6

    # 4. Run with resume=False -> overwrites file
    dummy_model.call_count = 0
    records_pass4 = evaluator.stream_evaluate_dataset(
        samples=samples,
        perturbations_by_tier=perturbations,
        output_jsonl_path=jsonl_path,
        resume=False,
    )
    assert len(records_pass4) == 4
    assert dummy_model.call_count == 4
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 4


# ============================================================================
# 6. Tests for CLI Driver Dry Run Pipeline
# ============================================================================


def test_cli_driver_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.run_noisy_eval import main

    out_dir = tmp_path / "eval_out"
    test_args = [
        "run_noisy_eval.py",
        "--dry-run",
        "--snrs",
        "20.0",
        "0.0",
        "--noise-types",
        "white",
        "--max-samples",
        "3",
        "--output-dir",
        str(out_dir),
    ]
    monkeypatch.setattr("sys.argv", test_args)
    main()

    assert (out_dir / "eval_records.jsonl").exists()
    assert (out_dir / "confusion_matrix.csv").exists()
    assert (out_dir / "confusion_cost_matrix.json").exists()
    assert (out_dir / "confusion_vs_cost_heatmap.png").exists()
    assert (out_dir / "snr_drift.png").exists()
    assert (out_dir / "confusion_manifold_3d.png").exists()
