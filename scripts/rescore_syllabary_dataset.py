#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/rescore_syllabary_dataset.py

Rescores greedy ASR inference vs. syllabary-guided CTC segmentation (with
zero-hyperparameter relative contrastive gating) on clean and pink-noise
perturbed audio across the Cherokee Syllabary dataset (split_audio_syl_target.csv).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
from transcription.alignment.models import CTCAlignerConfig
from transcription.alignment.normalizers import (
    normalize_phonetics_for_alignment,
    normalize_syllabary_for_alignment,
)
from transcription.alignment.phonotactics import prepare_cherokee_text
from transcription.evaluation.perturbations import AdditiveNoise
from transcription.cherokee.models import CherokeeASRModel
from transcription.utils.orthography import (
    clean_punctuation_and_whitespace,
    strip_tones_and_colons,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rescore_syllabary_dataset")


@dataclass(frozen=True)
class SyllabarySample:
    sample_id: str
    split: str
    audio_path: Path
    syllabary_raw: str
    input_template: str
    gt_target: str


@dataclass(frozen=True)
class RescoreMetrics:
    total_samples: int
    cer: float
    wer: float
    total_char_errors: int
    total_ref_chars: int
    total_word_errors: int
    total_ref_words: int
    char_substitutions: int
    char_insertions: int
    char_deletions: int
    word_substitutions: int
    word_insertions: int
    word_deletions: int
    avg_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_dataset(
    csv_path: Path, split_filter: Optional[str] = None
) -> List[SyllabarySample]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    samples: List[SyllabarySample] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "all")
            if split_filter and split != split_filter:
                continue

            audio_path = Path(row.get("audio_path") or row.get("path", ""))
            if not audio_path.is_absolute():
                audio_path = BASE_DIR / audio_path

            syl_raw = row.get("syllabary", "").strip()
            gt_raw = row.get("target") or row.get("sentence", "")

            # Convert Syllabary to base phonetic template (TTH)
            input_template = normalize_syllabary_for_alignment(syl_raw)
            # Spoken target in TTH (strip any tones/colons and normalize)
            gt_target = clean_punctuation_and_whitespace(strip_tones_and_colons(gt_raw))

            if not input_template or not gt_target or not audio_path.exists():
                continue

            samples.append(
                SyllabarySample(
                    sample_id=audio_path.stem,
                    split=split,
                    audio_path=audio_path,
                    syllabary_raw=syl_raw,
                    input_template=input_template,
                    gt_target=gt_target,
                )
            )

    logger.info(
        "Loaded %d samples (split_filter=%s) from %s",
        len(samples),
        split_filter,
        csv_path,
    )
    return samples


def compute_metrics(
    references: Sequence[str], hypotheses: Sequence[str], confidences: Sequence[float]
) -> RescoreMetrics:
    total_ref_chars = 0
    total_char_errors = 0
    total_ref_words = 0
    total_word_errors = 0

    char_subs = 0
    char_ins = 0
    char_dels = 0

    word_subs = 0
    word_ins = 0
    word_dels = 0

    for ref, hyp in zip(references, hypotheses):
        c_res = jiwer.process_characters(ref, hyp)
        char_subs += c_res.substitutions
        char_ins += c_res.insertions
        char_dels += c_res.deletions
        total_char_errors += c_res.substitutions + c_res.insertions + c_res.deletions
        total_ref_chars += len(ref)

        w_res = jiwer.process_words(ref, hyp)
        word_subs += w_res.substitutions
        word_ins += w_res.insertions
        word_dels += w_res.deletions
        total_word_errors += w_res.substitutions + w_res.insertions + w_res.deletions
        total_ref_words += len(ref.split())

    cer = (total_char_errors / total_ref_chars) if total_ref_chars > 0 else 0.0
    wer = (total_word_errors / total_ref_words) if total_ref_words > 0 else 0.0
    avg_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0.0

    return RescoreMetrics(
        total_samples=len(references),
        cer=round(cer, 4),
        wer=round(wer, 4),
        total_char_errors=total_char_errors,
        total_ref_chars=total_ref_chars,
        total_word_errors=total_word_errors,
        total_ref_words=total_ref_words,
        char_substitutions=char_subs,
        char_insertions=char_ins,
        char_deletions=char_dels,
        word_substitutions=word_subs,
        word_insertions=word_ins,
        word_deletions=word_dels,
        avg_confidence=round(avg_conf, 4),
    )


CACHE_DIR = BASE_DIR / "runs" / "cache" / "rescore_syllabary"


def get_cache_path(is_noisy: bool, snr_db: float, cache_dir: Path = CACHE_DIR) -> Path:
    if is_noisy:
        return cache_dir / f"lpz_cache_noisy_pink_{int(snr_db)}db.pt"
    return cache_dir / "lpz_cache_clean.pt"


def run_rescoring(
    samples: Sequence[SyllabarySample],
    model: CherokeeASRModel,
    aligner_config: CTCAlignerConfig,
    is_noisy: bool = False,
    snr_db: float = 18.0,
    force_recompute: bool = False,
    cache_dir: Path = CACHE_DIR,
) -> Tuple[RescoreMetrics, RescoreMetrics, List[Dict[str, Any]]]:
    """
    Evaluates both greedy ASR inference and syllabary-guided CTC segmentation
    for a given list of samples under either clean or noisy conditions.
    Utilizes on-disk caching of acoustic LPZ matrices and greedy predictions
    to eliminate redundant neural forward passes and noise synthesis.
    Returns: (greedy_metrics, guided_metrics, per_sample_records)
    """
    cache_path = get_cache_path(is_noisy, snr_db, cache_dir)
    cached_entries: Dict[str, Dict[str, Any]] = {}

    if not force_recompute and cache_path.exists():
        try:
            cached_entries = torch.load(
                cache_path, map_location="cpu", weights_only=False
            )
            logger.info(
                "Loaded %d cached LPZ entries from: %s", len(cached_entries), cache_path
            )
        except Exception as e:
            logger.warning(
                "Failed to load cache from %s (%s). Recomputing.", cache_path, e
            )
            cached_entries = {}

    noise_transform = (
        AdditiveNoise(snr_db=snr_db, noise_type="pink")
        if (is_noisy and any(s.sample_id not in cached_entries for s in samples))
        else None
    )

    char_list = [
        token
        for token, idx in sorted(
            model.processor.tokenizer.get_vocab().items(), key=lambda x: x[1]
        )
    ]
    pad_id = getattr(model.processor.tokenizer, "pad_token_id", 0) or 0

    ctc_params = CtcSegmentationParameters(
        char_list=char_list,
        blank=pad_id,
        syncope_tokens=aligner_config.syncope_tokens,
        intrusive_tokens=aligner_config.intrusive_tokens,
        intrusive_max_stride=int(aligner_config.intrusive_max_stride),
        index_duration=float(aligner_config.index_duration),
        score_min_mean_over_L=2,
        replace_spaces_with_blanks=False,
    )

    aligner = CTCSegmentationAligner(model=model, config=aligner_config)

    refs: List[str] = []
    greedy_hyps: List[str] = []
    greedy_confs: List[float] = []
    guided_hyps: List[str] = []
    guided_confs: List[float] = []
    sample_records: List[Dict[str, Any]] = []

    new_cache_entries: Dict[str, Dict[str, Any]] = dict(cached_entries)
    cache_updated = False

    total = len(samples)
    t0 = time.time()

    for idx, sample in enumerate(samples):
        if sample.sample_id in cached_entries:
            cached = cached_entries[sample.sample_id]
            lpz = cached["lpz"]
            dur_sec = cached["dur_sec"]
            lead_offset_sec = cached["lead_offset_sec"]
            greedy_text = cached["greedy_text"]
            greedy_conf = cached["greedy_conf"]
        else:
            # 1. Load Audio
            data, sr = sf.read(str(sample.audio_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=-1)
            if sr != 16000:
                import torchaudio.transforms as T

                resampler = T.Resample(orig_freq=sr, new_freq=16000)
                data_t = resampler(torch.from_numpy(data).unsqueeze(0)).squeeze(0)
                data = data_t.numpy()
                sr = 16000

            if noise_transform is not None:
                t_data = torch.from_numpy(data).unsqueeze(0)
                data = noise_transform(t_data, sr).squeeze(0).numpy()

            dur_sec = len(data) / 16000.0

            # Buffer lead/trail for accurate CTC boundary tracking
            lead_n = int(16000 * (aligner_config.buffer_lead_ms / 1000.0))
            trail_n = int(16000 * (aligner_config.buffer_trail_ms / 1000.0))
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
            buffered_audio = np.concatenate([lead_pad, data, trail_pad])
            lead_offset_sec = aligner_config.buffer_lead_ms / 1000.0

            # 2. Extract acoustic logits
            logits = model.get_logits(buffered_audio, sample_rate=16000)
            if logits.ndim == 3:
                logits = logits.squeeze(0)
            lpz = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()

            # 3. Greedy Inference
            asr_res = model.decode(lpz)
            greedy_text = clean_punctuation_and_whitespace(
                strip_tones_and_colons(asr_res.text)
            )
            greedy_conf = asr_res.confidence

            new_cache_entries[sample.sample_id] = {
                "lpz": lpz,
                "dur_sec": dur_sec,
                "lead_offset_sec": lead_offset_sec,
                "greedy_text": greedy_text,
                "greedy_conf": greedy_conf,
            }
            cache_updated = True

        # 4. Syllabary-Guided CTC Segmentation (Always executed live through current trellis logic)
        words = sample.input_template.split()
        gt_mat, utt_indices = prepare_cherokee_text(
            ctc_params,
            words,
            char_list,
            enforce_phonotactics=aligner_config.enforce_phonotactics,
        )
        timings, char_probs, state_list = ctc_segmentation(ctc_params, lpz, gt_mat)

        word_intervals = aligner._extract_word_intervals(
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
        guided_text = clean_punctuation_and_whitespace(" ".join(emitted_words))
        guided_conf = (
            float(np.mean([w.confidence for w in word_intervals]))
            if word_intervals
            else 0.0
        )

        refs.append(sample.gt_target)
        greedy_hyps.append(greedy_text)
        greedy_confs.append(greedy_conf)
        guided_hyps.append(guided_text)
        guided_confs.append(guided_conf)

        sample_records.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "syllabary": sample.syllabary_raw,
                "template": sample.input_template,
                "ground_truth": sample.gt_target,
                "greedy_hyp": greedy_text,
                "greedy_conf": round(greedy_conf, 4),
                "guided_hyp": guided_text,
                "guided_conf": round(guided_conf, 4),
            }
        )

        if (idx + 1) % 200 == 0 or idx + 1 == total:
            elapsed = time.time() - t0
            logger.info(
                "[%s] Processed %d/%d samples (%.1fs elapsed, %.1f samp/s)...",
                "NOISY" if is_noisy else "CLEAN",
                idx + 1,
                total,
                elapsed,
                (idx + 1) / elapsed if elapsed > 0 else 0,
            )

    if cache_updated:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(new_cache_entries, cache_path)
        logger.info(
            "Saved %d LPZ entries to cache: %s", len(new_cache_entries), cache_path
        )

    greedy_metrics = compute_metrics(refs, greedy_hyps, greedy_confs)
    guided_metrics = compute_metrics(refs, guided_hyps, guided_confs)

    return greedy_metrics, guided_metrics, sample_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore greedy vs syllabary-guided ASR on Cherokee Syllabary dataset."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=BASE_DIR / "training_data" / "processed" / "split_audio_syl_target.csv",
        help="Path to split_audio_syl_target.csv",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "test", "valid", "train"],
        help="Dataset split to evaluate ('all', 'test', 'valid', 'train')",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=18.0,
        help="Pink noise SNR in dB for noisy evaluation (default: 18.0)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=BASE_DIR / "runs" / "evaluation" / "rescore_syllabary_results.json",
        help="Path to save evaluation output JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples evaluated (for debugging)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of audio LPZ logits and greedy inference, ignoring cache",
    )

    args = parser.parse_args()

    split_filter = None if args.split == "all" else args.split
    samples = load_dataset(args.csv_path, split_filter=split_filter)

    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]
        logger.info("Limited evaluation to first %d samples.", len(samples))

    if not samples:
        logger.error("No samples found to evaluate.")
        sys.exit(1)

    logger.info("Loading Cherokee ASR model...")
    model = CherokeeASRModel.from_pretrained_or_best()
    logger.info("Loaded model: %s", model.model_name)

    aligner_config = CTCAlignerConfig(cache=False, enforce_phonotactics=True)

    # 1. Clean Evaluation
    logger.info("=== Running Clean Evaluation (%d samples) ===", len(samples))
    clean_greedy, clean_guided, clean_records = run_rescoring(
        samples=samples,
        model=model,
        aligner_config=aligner_config,
        is_noisy=False,
        force_recompute=args.force_recompute,
    )

    # 2. Noisy Evaluation (Pink Noise 18 dB SNR)
    logger.info(
        "=== Running Noisy Evaluation (Pink Noise %.1f dB SNR) ===", args.snr_db
    )
    noisy_greedy, noisy_guided, noisy_records = run_rescoring(
        samples=samples,
        model=model,
        aligner_config=aligner_config,
        is_noisy=True,
        snr_db=args.snr_db,
        force_recompute=args.force_recompute,
    )

    # Aggregate by split if evaluating "all"
    splits_summary: Dict[str, Any] = {}
    splits_to_eval = (
        ["test", "valid", "train", "all"] if args.split == "all" else [args.split]
    )

    for s_name in splits_to_eval:
        s_filter = None if s_name == "all" else s_name
        s_indices = [
            i for i, s in enumerate(samples) if s_filter is None or s.split == s_filter
        ]
        if not s_indices:
            continue

        s_clean_greedy = compute_metrics(
            [clean_records[i]["ground_truth"] for i in s_indices],
            [clean_records[i]["greedy_hyp"] for i in s_indices],
            [clean_records[i]["greedy_conf"] for i in s_indices],
        )
        s_clean_guided = compute_metrics(
            [clean_records[i]["ground_truth"] for i in s_indices],
            [clean_records[i]["guided_hyp"] for i in s_indices],
            [clean_records[i]["guided_conf"] for i in s_indices],
        )
        s_noisy_greedy = compute_metrics(
            [noisy_records[i]["ground_truth"] for i in s_indices],
            [noisy_records[i]["greedy_hyp"] for i in s_indices],
            [noisy_records[i]["greedy_conf"] for i in s_indices],
        )
        s_noisy_guided = compute_metrics(
            [noisy_records[i]["ground_truth"] for i in s_indices],
            [noisy_records[i]["guided_hyp"] for i in s_indices],
            [noisy_records[i]["guided_conf"] for i in s_indices],
        )

        splits_summary[s_name] = {
            "total_samples": len(s_indices),
            "clean": {
                "greedy": s_clean_greedy.to_dict(),
                "syllabary_guided": s_clean_guided.to_dict(),
                "cer_delta": round(s_clean_guided.cer - s_clean_greedy.cer, 4),
                "wer_delta": round(s_clean_guided.wer - s_clean_greedy.wer, 4),
            },
            "noisy": {
                "greedy": s_noisy_greedy.to_dict(),
                "syllabary_guided": s_noisy_guided.to_dict(),
                "cer_delta": round(s_noisy_guided.cer - s_noisy_greedy.cer, 4),
                "wer_delta": round(s_noisy_guided.wer - s_noisy_greedy.wer, 4),
            },
        }

    # Format Summary Table
    print("\n" + "=" * 88)
    print(
        f"{'EVALUATION RESULTS SUMMARY: GREEDY VS SYLLABARY-GUIDED CTC SEGMENTATION':^88}"
    )
    print(
        f"{'(ctc-segmentation PR #7 Relative Contrastive Gating - Zero Tuned Hyperparams)':^88}"
    )
    print("=" * 88)
    print(
        f"{'Split':<8} | {'Condition':<6} | {'Greedy CER':<10} | {'Guided CER':<10} | {'Δ CER':<8} | {'Greedy WER':<10} | {'Guided WER':<10} | {'Δ WER':<8}"
    )
    print("-" * 88)

    for s_name, data in splits_summary.items():
        c_g_cer = data["clean"]["greedy"]["cer"] * 100
        c_gd_cer = data["clean"]["syllabary_guided"]["cer"] * 100
        c_d_cer = (
            data["clean"]["syllabary_guided"]["cer"] - data["clean"]["greedy"]["cer"]
        ) * 100
        c_g_wer = data["clean"]["greedy"]["wer"] * 100
        c_gd_wer = data["clean"]["syllabary_guided"]["wer"] * 100
        c_d_wer = (
            data["clean"]["syllabary_guided"]["wer"] - data["clean"]["greedy"]["wer"]
        ) * 100

        n_g_cer = data["noisy"]["greedy"]["cer"] * 100
        n_gd_cer = data["noisy"]["syllabary_guided"]["cer"] * 100
        n_d_cer = (
            data["noisy"]["syllabary_guided"]["cer"] - data["noisy"]["greedy"]["cer"]
        ) * 100
        n_g_wer = data["noisy"]["greedy"]["wer"] * 100
        n_gd_wer = data["noisy"]["syllabary_guided"]["wer"] * 100
        n_d_wer = (
            data["noisy"]["syllabary_guided"]["wer"] - data["noisy"]["greedy"]["wer"]
        ) * 100

        print(
            f"{s_name:<8} | {'Clean':<6} | {c_g_cer:>9.2f}% | {c_gd_cer:>9.2f}% | {c_d_cer:>+7.2f}% | {c_g_wer:>9.2f}% | {c_gd_wer:>9.2f}% | {c_d_wer:>+7.2f}%"
        )
        print(
            f"{s_name:<8} | {'Noisy':<6} | {n_g_cer:>9.2f}% | {n_gd_cer:>9.2f}% | {n_d_cer:>+7.2f}% | {n_g_wer:>9.2f}% | {n_gd_wer:>9.2f}% | {n_d_wer:>+7.2f}%"
        )
        print("-" * 88)

    print("=" * 88 + "\n")

    # Save Results JSON
    output_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model.model_name,
        "snr_db": args.snr_db,
        "splits_summary": splits_summary,
        "clean_samples": clean_records,
        "noisy_samples": noisy_records,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved full evaluation payload to: %s", args.output_json)


if __name__ == "__main__":
    main()
