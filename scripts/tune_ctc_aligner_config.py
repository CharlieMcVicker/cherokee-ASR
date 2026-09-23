#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/tune_ctc_aligner_config.py

Hyperparameter tuning and evaluation framework for CTCAlignerConfig using underspecified
Cherokee Syllabary as the input template, evaluating against ground-truth spoken phonetics
on both clean and pink-noise perturbed audio (15-20 dB SNR).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jiwer
import numpy as np
import soundfile as sf
import torch

# Ensure repository root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ctc_segmentation import (  # type: ignore
    CtcSegmentationParameters,
    ctc_segmentation,
)
from transcription.alignment.ctc_aligner import CTCSegmentationAligner
from transcription.alignment.models import CTCAlignerConfig, TextChunk, WordInterval
from transcription.alignment.normalizers import (
    normalize_phonetics_for_alignment,
    normalize_syllabary_for_alignment,
)
from transcription.alignment.phonotactics import prepare_cherokee_text
from transcription.evaluation.perturbations import AdditiveNoise
from transcription.cherokee.models import CherokeeASRModel
from transcription.utils.syllabary_map import cherokee_to_bad_phonetics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tune_ctc_aligner_config")


@dataclass(frozen=True)
class DatasetSample:
    """A single evaluation dataset item."""

    sample_id: str
    audio_path: Path
    syllabary_raw: str
    input_template: str
    gt_target: str


@dataclass(frozen=True)
class EvaluationMetricResult:
    """Evaluation metrics for a given dataset pass."""

    cer: float
    wer: float
    total_char_errors: int
    total_ref_chars: int
    total_word_errors: int
    total_ref_words: int
    avg_confidence: float
    flagged_word_count: int
    unaligned_word_count: int
    total_aligned_words: int


@dataclass
class SweepCandidateResult:
    """Sweep evaluation results for a single candidate configuration."""

    candidate_name: str
    syncope_penalty: float
    intrusive_penalty: float
    intrusive_penalties: Optional[Dict[str, float]]
    intrusive_min_logprobs: Optional[Dict[str, float]]
    enforce_phonotactics: bool
    clean_metrics: EvaluationMetricResult
    noisy_metrics: EvaluationMetricResult
    composite_cer: float
    composite_wer: float


def load_syllabary_target_dataset(
    csv_path: Path, split: str = "test"
) -> List[DatasetSample]:
    """
    Loads dataset samples where the input template is derived from Cherokee Syllabary
    and the ground truth is the fine-grained phonetic transcript.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    samples: List[DatasetSample] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if "split" in row and row["split"] != split:
                continue

            p = Path(row.get("audio_path") or row.get("path", ""))
            if not p.is_absolute():
                p = BASE_DIR / p

            syl_raw = row.get("syllabary", "").strip()
            gt_raw = row.get("target") or row.get("sentence", "")

            # Syllabary -> Underspecified base phonetic template
            input_template = normalize_syllabary_for_alignment(syl_raw)
            # Ground truth spoken phonetics
            gt_target = normalize_phonetics_for_alignment(gt_raw)

            if not input_template or not gt_target:
                continue

            samples.append(
                DatasetSample(
                    sample_id=p.stem,
                    audio_path=p,
                    syllabary_raw=syl_raw,
                    input_template=input_template,
                    gt_target=gt_target,
                )
            )

    logger.info("Loaded %d '%s' samples from %s", len(samples), split, csv_path)
    return samples


def precompute_logits(
    samples: Sequence[DatasetSample],
    model: CherokeeASRModel,
    transform: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    buffer_lead_ms: int = 100,
    buffer_trail_ms: int = 300,
) -> Dict[str, Tuple[np.ndarray, float, float]]:
    """
    Precomputes acoustic log-probability tensors (lpz) for all dataset samples.
    """
    cache: Dict[str, Tuple[np.ndarray, float, float]] = {}
    total = len(samples)

    for idx, sample in enumerate(samples):
        data, sr = sf.read(str(sample.audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=-1)
        if sr != 16000:
            import torchaudio.transforms as T

            resampler = T.Resample(orig_freq=sr, new_freq=16000)
            data_tensor = resampler(torch.from_numpy(data).unsqueeze(0)).squeeze(0)
            data = data_tensor.numpy()
            sr = 16000

        if transform is not None:
            t_data = torch.from_numpy(data).unsqueeze(0)  # [1, N]
            t_perturbed = transform(t_data, sr).squeeze(0).numpy()
            data = t_perturbed

        total_audio_sec = float(len(data)) / 16000.0

        lead_n = int(16000 * (buffer_lead_ms / 1000.0))
        trail_n = int(16000 * (buffer_trail_ms / 1000.0))
        lead_pad = (
            np.zeros(lead_n, dtype=np.float32)
            if lead_n > 0
            else np.array([], dtype=np.float32)
        )
        trail_pad = (
            np.zeros(trail_n, dtype=np.float32)
            if trail_n > 0
            else np.array([], dtype=np.float32)
        )
        buffered_samples = np.concatenate([lead_pad, data, trail_pad])
        lead_offset_sec = buffer_lead_ms / 1000.0

        logits = model.get_logits(buffered_samples, sample_rate=16000)
        if logits.ndim == 3:
            logits = logits.squeeze(0)
        lpz = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()

        cache[sample.sample_id] = (lpz, total_audio_sec, lead_offset_sec)
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            logger.info("Computed logits for %d/%d samples...", idx + 1, total)

    return cache


def evaluate_candidate(
    config: CTCAlignerConfig,
    samples: Sequence[DatasetSample],
    logits_cache: Dict[str, Tuple[np.ndarray, float, float]],
    char_list: List[str],
    pad_id: int,
) -> EvaluationMetricResult:
    """
    Evaluates a candidate CTCAlignerConfig on cached logits across all dataset samples.
    Measures CER/WER of emitted path against gt_target.
    """
    ctc_params = CtcSegmentationParameters(
        char_list=char_list,
        blank=pad_id,
        syncope_tokens=config.syncope_tokens,
        syncope_penalty=float(config.syncope_penalty),
        intrusive_tokens=config.intrusive_tokens,
        intrusive_penalty=float(config.intrusive_penalty),
        intrusive_penalties=config.intrusive_penalties,
        intrusive_min_logprobs=config.intrusive_min_logprobs,
        intrusive_max_stride=int(config.intrusive_max_stride),
        index_duration=float(config.index_duration),
        score_min_mean_over_L=2,
        replace_spaces_with_blanks=False,
    )

    total_ref_chars = 0
    total_char_errors = 0
    total_ref_words = 0
    total_word_errors = 0
    total_confidence = 0.0
    total_words_aligned = 0
    flagged_word_count = 0
    unaligned_word_count = 0

    aligner_dummy = CTCSegmentationAligner(config=config)

    for sample in samples:
        lpz, dur_sec, lead_offset_sec = logits_cache[sample.sample_id]
        # Input template words to align
        words = sample.input_template.split()
        if not words or lpz.shape[0] == 0:
            continue

        gt_mat, utt_indices = prepare_cherokee_text(
            ctc_params,
            words,
            char_list,
            enforce_phonotactics=config.enforce_phonotactics,
        )
        timings, char_probs, state_list = ctc_segmentation(ctc_params, lpz, gt_mat)

        word_intervals = aligner_dummy._extract_word_intervals(
            words=words,
            timings=timings,
            char_probs=char_probs,
            state_list=state_list,
            utt_indices=utt_indices,
            start_word_idx=0,
            dur_sec=dur_sec,
            lead_offset_sec=lead_offset_sec,
            initial_prev_end=0.0,
        )

        emitted_words = [w.emitted_word for w in word_intervals if w.emitted_word]
        emitted_text = " ".join(emitted_words)
        reference_text = sample.gt_target

        # Compute error counts against ground truth target
        char_err_count = (
            jiwer.process_characters(reference_text, emitted_text).substitutions
            + jiwer.process_characters(reference_text, emitted_text).insertions
            + jiwer.process_characters(reference_text, emitted_text).deletions
        )
        ref_char_len = len(reference_text)

        word_err_count = (
            jiwer.process_words(reference_text, emitted_text).substitutions
            + jiwer.process_words(reference_text, emitted_text).insertions
            + jiwer.process_words(reference_text, emitted_text).deletions
        )
        ref_word_len = len(reference_text.split())

        total_ref_chars += ref_char_len
        total_char_errors += char_err_count
        total_ref_words += ref_word_len
        total_word_errors += word_err_count

        for w in word_intervals:
            total_words_aligned += 1
            total_confidence += w.confidence
            if w.flagged:
                flagged_word_count += 1
            if not w.emitted_word:
                unaligned_word_count += 1

    overall_cer = total_char_errors / max(1, total_ref_chars)
    overall_wer = total_word_errors / max(1, total_ref_words)
    avg_conf = total_confidence / max(1, total_words_aligned)

    return EvaluationMetricResult(
        cer=round(overall_cer, 6),
        wer=round(overall_wer, 6),
        total_char_errors=total_char_errors,
        total_ref_chars=total_ref_chars,
        total_word_errors=total_word_errors,
        total_ref_words=total_ref_words,
        avg_confidence=round(avg_conf, 6),
        flagged_word_count=flagged_word_count,
        unaligned_word_count=unaligned_word_count,
        total_aligned_words=total_words_aligned,
    )


def generate_candidate_grid() -> List[Tuple[str, CTCAlignerConfig]]:
    """
    Generates structured list of CTCAlignerConfig candidate parameter sets.
    """
    candidates: List[Tuple[str, CTCAlignerConfig]] = []

    # 1. Baseline default config
    candidates.append(("baseline_default", CTCAlignerConfig()))

    # 2. Phonotactics enforcement toggle
    for enforce in [True, False]:
        candidates.append(
            (
                f"phonotactics_{enforce}",
                CTCAlignerConfig(enforce_phonotactics=enforce),
            )
        )

    # 3. Syncope penalty sweep
    syncope_penalties = [0.5, 1.0, 2.0, 3.5, 5.0, 6.0, 8.0, 10.0, 15.0]
    for sp in syncope_penalties:
        candidates.append(
            (
                f"syncope_{sp}",
                CTCAlignerConfig(
                    syncope_penalty=sp,
                    intrusive_penalties={"h": 4.5, "'": 0.8},
                    intrusive_min_logprobs={"h": -1.0498, "'": -1.6094},
                    enforce_phonotactics=True,
                ),
            )
        )

    # 4. Intrusive /h/ and /'/ penalty grid
    h_penalties = [0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 12.0]
    glottal_penalties = [0.2, 0.5, 0.8, 1.2, 2.0, 3.5]
    for hp in h_penalties:
        for gp in glottal_penalties:
            candidates.append(
                (
                    f"h_{hp}_glottal_{gp}",
                    CTCAlignerConfig(
                        syncope_penalty=5.0,
                        intrusive_penalties={"h": hp, "'": gp},
                        intrusive_min_logprobs={"h": -1.0498, "'": -1.6094},
                        enforce_phonotactics=True,
                    ),
                )
            )

    # 5. Min logprob sweeps
    min_logprob_candidates = [
        ("no_mlp", None),
        ("p0.15_h_p0.10_g", {"h": -1.8971, "'": -2.3026}),
        ("p0.35_h_p0.20_g", {"h": -1.0498, "'": -1.6094}),
        ("p0.50_h_p0.30_g", {"h": -0.6931, "'": -1.2040}),
        ("p0.70_h_p0.50_g", {"h": -0.3567, "'": -0.6931}),
    ]
    for mlp_name, mlp in min_logprob_candidates:
        for sp in [3.5, 5.0, 6.0]:
            candidates.append(
                (
                    f"mlp_{mlp_name}_sync_{sp}",
                    CTCAlignerConfig(
                        syncope_penalty=sp,
                        intrusive_penalties={"h": 3.0, "'": 0.8},
                        intrusive_min_logprobs=mlp,
                        enforce_phonotactics=True,
                    ),
                )
            )

    # 6. Presets
    presets = [
        (
            "permissive_laryngeals",
            CTCAlignerConfig(
                syncope_penalty=3.5,
                intrusive_penalties={"h": 1.5, "'": 0.5},
                intrusive_min_logprobs={"h": -1.8971, "'": -2.3026},
                enforce_phonotactics=True,
            ),
        ),
        (
            "balanced_robust",
            CTCAlignerConfig(
                syncope_penalty=5.0,
                intrusive_penalties={"h": 3.0, "'": 0.8},
                intrusive_min_logprobs={"h": -1.0498, "'": -1.6094},
                enforce_phonotactics=True,
            ),
        ),
        (
            "conservative_transitions",
            CTCAlignerConfig(
                syncope_penalty=8.0,
                intrusive_penalties={"h": 6.0, "'": 1.5},
                intrusive_min_logprobs={"h": -0.6931, "'": -1.2040},
                enforce_phonotactics=True,
            ),
        ),
    ]
    for p_name, p_cfg in presets:
        candidates.append((f"preset_{p_name}", p_cfg))

    return candidates


def run_tuning(
    csv_path: Path = BASE_DIR
    / "training_data"
    / "processed"
    / "split_audio_syl_target.csv",
    split: str = "test",
    snr_db: float = 18.0,
    output_json: Path = BASE_DIR
    / "runs"
    / "evaluation"
    / "ctc_aligner_config_tuning_results.json",
) -> Dict[str, Any]:
    """
    Main tuning driver executing clean and noisy hyperparameter sweeps on Syllabary->GT task.
    """
    output_json.parent.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Initializing CherokeeASRModel on device %s...", device)
    token = os.environ.get("HF_TOKEN", None)
    model = CherokeeASRModel.from_pretrained_or_best(device=device, token=token)

    vocab = model.processor.tokenizer.get_vocab()
    char_list = [token for token, idx in sorted(vocab.items(), key=lambda x: x[1])]
    pad_id = getattr(model.processor.tokenizer, "pad_token_id", None)
    if pad_id is None or not isinstance(pad_id, (int, np.integer)):
        pad_id = vocab.get("[PAD]", 0)
    pad_id = int(pad_id)

    # 1. Load dataset
    samples = load_syllabary_target_dataset(csv_path, split=split)

    # 2. Precompute clean logits
    logger.info("--- Computing Clean Acoustic Logits ---")
    t0 = time.time()
    clean_logits = precompute_logits(samples, model, transform=None)
    logger.info("Clean logits extracted in %.2f s", time.time() - t0)

    # 3. Precompute noisy logits with additive pink noise
    logger.info(
        "--- Computing Noisy Acoustic Logits (Pink Noise @ %.1f dB SNR) ---", snr_db
    )
    torch.manual_seed(42)
    np.random.seed(42)
    pink_noise_transform = AdditiveNoise(snr_db=snr_db, noise_type="pink")
    t0 = time.time()
    noisy_logits = precompute_logits(samples, model, transform=pink_noise_transform)
    logger.info("Noisy logits extracted in %.2f s", time.time() - t0)

    # 4. Generate candidate grid
    candidates = generate_candidate_grid()
    logger.info(
        "Evaluating %d candidate configurations across clean and noisy sets...",
        len(candidates),
    )

    results: List[SweepCandidateResult] = []

    t_eval_start = time.time()
    for name, cfg in candidates:
        clean_res = evaluate_candidate(cfg, samples, clean_logits, char_list, pad_id)
        noisy_res = evaluate_candidate(cfg, samples, noisy_logits, char_list, pad_id)
        composite_cer = round(0.5 * clean_res.cer + 0.5 * noisy_res.cer, 6)
        composite_wer = round(0.5 * clean_res.wer + 0.5 * noisy_res.wer, 6)

        rec = SweepCandidateResult(
            candidate_name=name,
            syncope_penalty=float(cfg.syncope_penalty),
            intrusive_penalty=float(cfg.intrusive_penalty),
            intrusive_penalties=cfg.intrusive_penalties,
            intrusive_min_logprobs=cfg.intrusive_min_logprobs,
            enforce_phonotactics=bool(cfg.enforce_phonotactics),
            clean_metrics=clean_res,
            noisy_metrics=noisy_res,
            composite_cer=composite_cer,
            composite_wer=composite_wer,
        )
        results.append(rec)

    total_eval_time = time.time() - t_eval_start
    logger.info(
        "Completed %d grid evaluations in %.2f s", len(candidates), total_eval_time
    )

    # Sort results by composite CER
    sorted_by_composite_cer = sorted(
        results, key=lambda r: (r.composite_cer, r.composite_wer)
    )
    sorted_by_clean_cer = sorted(
        results, key=lambda r: (r.clean_metrics.cer, r.clean_metrics.wer)
    )
    sorted_by_noisy_cer = sorted(
        results, key=lambda r: (r.noisy_metrics.cer, r.noisy_metrics.wer)
    )

    best_clean = sorted_by_clean_cer[0]
    best_noisy = sorted_by_noisy_cer[0]
    best_composite = sorted_by_composite_cer[0]

    logger.info("=== BEST CLEAN CONFIGURATION ===")
    logger.info(
        "Candidate: %s | Clean CER: %.4f | Clean WER: %.4f | Conf: %.4f",
        best_clean.candidate_name,
        best_clean.clean_metrics.cer,
        best_clean.clean_metrics.wer,
        best_clean.clean_metrics.avg_confidence,
    )

    logger.info("=== BEST NOISY CONFIGURATION ===")
    logger.info(
        "Candidate: %s | Noisy CER: %.4f | Noisy WER: %.4f | Conf: %.4f",
        best_noisy.candidate_name,
        best_noisy.noisy_metrics.cer,
        best_noisy.noisy_metrics.wer,
        best_noisy.noisy_metrics.avg_confidence,
    )

    logger.info("=== BEST BALANCED COMPOSITE CONFIGURATION ===")
    logger.info(
        "Candidate: %s | Comp CER: %.4f | Clean CER: %.4f | Noisy CER: %.4f",
        best_composite.candidate_name,
        best_composite.composite_cer,
        best_composite.clean_metrics.cer,
        best_composite.noisy_metrics.cer,
    )

    export_payload = {
        "dataset": str(csv_path),
        "split": split,
        "total_samples": len(samples),
        "noise_profile": {
            "type": "pink",
            "snr_db": snr_db,
            "seed": 42,
        },
        "best_clean": asdict(best_clean),
        "best_noisy": asdict(best_noisy),
        "best_composite": asdict(best_composite),
        "all_candidates": [asdict(r) for r in sorted_by_composite_cer],
    }

    with open(output_json, mode="w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    logger.info("Saved full tuning results to %s", output_json)

    return export_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune CTCAlignerConfig on Syllabary->Target task."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=BASE_DIR / "training_data" / "processed" / "split_audio_syl_target.csv",
        help="Path to evaluation CSV dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (e.g. test, valid).",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=18.0,
        help="Signal to Noise Ratio (dB) for pink noise perturbation.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=BASE_DIR
        / "runs"
        / "evaluation"
        / "ctc_aligner_config_tuning_results.json",
        help="Path to export results JSON.",
    )
    args = parser.parse_args()
    run_tuning(
        csv_path=args.csv,
        split=args.split,
        snr_db=args.snr_db,
        output_json=args.output_json,
    )
