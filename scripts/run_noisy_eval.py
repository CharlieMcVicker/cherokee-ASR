#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_noisy_eval.py

Unified CLI driver for running multi-SNR acoustic evaluation sweeps, extracting
empirical character confusion matrices, generating distance cost JSON artifacts,
and rendering phonetic manifold visualizations.
"""

from __future__ import annotations

import os
import sys

# Auto-reexec with cherokee-asr conda environment if executed with an unconfigured python
try:
    import numpy as np
    import torch
except ImportError:
    for cand in [
        "/opt/homebrew/Caskroom/miniconda/base/envs/cherokee-asr/bin/python",
        os.path.expanduser("~/miniconda3/envs/cherokee-asr/bin/python"),
        os.path.expanduser("~/anaconda3/envs/cherokee-asr/bin/python"),
    ]:
        if os.path.exists(cand) and sys.executable != cand:
            os.execv(cand, [cand] + sys.argv)
    raise

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from transcription.evaluation.confusion import (
    ConfusionAccumulator,
    character_levenshtein_align,
)
from transcription.evaluation.cost_engine import ConfusionCostEngine
from transcription.evaluation.evaluator import EvaluationRecord, NoisyEvaluator
from transcription.evaluation.manifold import PhoneticManifoldAnalyzer
from transcription.evaluation.perturbations import AdditiveNoise, AudioTransform
from transcription.evaluation.visualizer import ManifoldVisualizer
from transcription.inference.infer import normalize_text
from transcription.models.asr_model import CherokeeASRModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_noisy_eval")


class MockTokenizer:
    """Mock tokenizer for dry-run evaluations."""

    def __init__(self, vocab: Optional[list[str]] = None) -> None:
        if vocab is None:
            vocab = [
                "[PAD]",
                "|",
                "a",
                "e",
                "i",
                "o",
                "u",
                "v",
                "s",
                "t",
                "l",
                "k",
                "h",
                "n",
                "w",
                "y",
                "m",
                "d",
                "g",
                "j",
                "b",
                ":",
                "'",
                "^",
            ]
        self.vocab = {tok: idx for idx, tok in enumerate(vocab)}
        self.inv_vocab = {idx: tok for idx, tok in enumerate(vocab)}
        self.pad_token_id = self.vocab.get("[PAD]", 0)
        self.word_delimiter_token_id = self.vocab.get("|", 1)

    def decode(self, token_ids: list[int]) -> str:
        res = []
        for tid in token_ids:
            tok = self.inv_vocab.get(tid, "")
            if tok not in ("[PAD]", "|"):
                res.append(tok)
        return "".join(res)


class MockProcessor:
    """Mock processor for dry-run evaluations."""

    def __init__(self) -> None:
        self.tokenizer = MockTokenizer()

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)


class MockASRModel:
    """Mock ASR model producing synthetic probability distributions."""

    def __init__(self, processor: Optional[MockProcessor] = None) -> None:
        self.processor = processor or MockProcessor()
        self.device = "cpu"

    def get_probabilities(
        self, waveform: torch.Tensor, sample_rate: int = 16000
    ) -> torch.Tensor:
        v_size = len(self.processor.tokenizer.vocab)
        num_frames = 20
        probs = np.random.uniform(0.01, 0.05, size=(num_frames, v_size)).astype(
            np.float32
        )
        for t in range(num_frames):
            if t % 3 == 0:
                probs[t, self.processor.tokenizer.pad_token_id] = 0.8
            else:
                tok_idx = (t * 3) % (v_size - 2) + 2
                probs[t, tok_idx] = 0.75
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        return torch.tensor(probs, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified CLI driver for multi-SNR evaluation sweeps and confusion analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default="training_data/processed/cim-wav2vec2-test.csv",
        help="Path to evaluation dataset CSV (columns: 'path', 'sentence').",
    )
    parser.add_argument(
        "--snrs",
        nargs="+",
        type=float,
        default=[25.0, 15.0, 5.0, 0.0, -5.0],
        help="SNR levels in dB to sweep over.",
    )
    parser.add_argument(
        "--noise-types",
        nargs="+",
        default=["white", "pink"],
        help="Noise profile types to evaluate (e.g. 'white', 'pink').",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K alternative posterior probabilities to extract per frame.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/evaluation",
        help="Directory to save evaluation records, matrices, and plots.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path or HuggingFace repo ID of the model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Git commit hash, branch name, or tag of the HuggingFace model checkpoint.",
    )
    parser.add_argument(
        "--output-suffix",
        "--suffix",
        type=str,
        default=None,
        help="Optional suffix to append to output file names (e.g. '_prebible' or 'prebible').",
    )
    parser.add_argument(
        "--eval-records-filename",
        type=str,
        default=None,
        help="Explicit filename for evaluation records JSONL (overrides default/suffix).",
    )
    parser.add_argument(
        "--confusion-matrix-filename",
        type=str,
        default=None,
        help="Explicit filename for confusion matrix CSV (overrides default/suffix).",
    )
    parser.add_argument(
        "--cost-matrix-filename",
        type=str,
        default=None,
        help="Explicit filename for confusion cost matrix JSON (overrides default/suffix).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device backend (e.g. 'mps', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of dataset samples to evaluate (useful for quick testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute synthetic mock evaluation without downloading/loading HF model weights.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume; re-evaluates all samples and overwrites JSONL sink.",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.1,
        help="Dirichlet smoothing prior alpha for confusion probability normalization.",
    )
    return parser.parse_args()


def load_dataset(
    csv_path: str, max_samples: Optional[int] = None, dry_run: bool = False
) -> list[dict[str, Any]]:
    path = Path(csv_path)
    samples: list[dict[str, Any]] = []

    if path.exists():
        logger.info("Loading dataset records from %s...", path)
        dropped_dg = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                raw_ref = row.get("sentence") or row.get("reference") or ""
                clean_ref = normalize_text(raw_ref)

                # Filter out bad data: rows containing 'd' or 'g' in reference
                if "d" in clean_ref or "g" in clean_ref:
                    dropped_dg += 1
                    continue

                if not clean_ref.strip():
                    continue

                item: dict[str, Any] = {
                    "audio_id": row.get("id")
                    or (
                        Path(row["path"]).stem
                        if "path" in row and row["path"]
                        else f"utt_{idx:04d}"
                    ),
                    "reference": clean_ref,
                    "path": row.get("path"),
                }
                if dry_run and (not item["path"] or not os.path.exists(item["path"])):
                    item["waveform"] = torch.randn(16000, dtype=torch.float32) * 0.05
                samples.append(item)
        if dropped_dg > 0:
            logger.info(
                "Filtered out %d rows containing 'd' or 'g' from dataset.", dropped_dg
            )
    else:
        if dry_run:
            logger.warning(
                "Dataset CSV %s not found. Creating synthetic dry-run samples.", path
            )
            dummy_sentences = [
                normalize_text(s)
                for s in [
                    "atsilo:nu'e:lha",
                    "na ku:ku ama",
                    "tsi:tayiha lha",
                    "hihska:ho:'ihs",
                    "sta:ya yikawo:ni:hsa",
                ]
                if "d" not in normalize_text(s) and "g" not in normalize_text(s)
            ]
            for idx, sent in enumerate(dummy_sentences):
                samples.append(
                    {
                        "audio_id": f"dry_sample_{idx:03d}",
                        "reference": sent,
                        "waveform": torch.randn(16000, dtype=torch.float32) * 0.05,
                    }
                )
        else:
            raise FileNotFoundError(f"Dataset CSV not found: {path}")

    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info("Loaded %d dataset sample(s) for evaluation.", len(samples))
    return samples


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting noisy ASR evaluation pipeline...")
    logger.info("Target output directory: %s", out_dir.resolve())
    logger.info("SNR tiers: %s | Noise profiles: %s", args.snrs, args.noise_types)

    # 1. Load samples
    samples = load_dataset(
        args.dataset_csv, max_samples=args.max_samples, dry_run=args.dry_run
    )
    if not samples:
        logger.warning("No samples found to evaluate. Exiting.")
        return

    # 2. Initialize Model and Evaluator
    if args.dry_run:
        logger.info("[Dry Run] Initializing MockCherokeeASRModel...")
        mock_model = MockASRModel()
        evaluator = NoisyEvaluator(model=mock_model, top_k=args.top_k)
    else:
        logger.info(
            "Instantiating CherokeeASRModel (checkpoint: %s, revision: %s, device: %s)...",
            args.checkpoint,
            args.revision,
            args.device,
        )
        model = CherokeeASRModel.from_pretrained_or_best(
            path_or_repo=args.checkpoint,
            revision=args.revision,
            device=args.device,
        )
        evaluator = NoisyEvaluator(model=model, top_k=args.top_k)

    # 3. Setup perturbation tiers
    perturbations_by_tier: list[tuple[float, str, Optional[AudioTransform]]] = []
    for snr in args.snrs:
        for ntype in args.noise_types:
            transform = AdditiveNoise(snr_db=snr, noise_type=ntype)
            perturbations_by_tier.append((float(snr), str(ntype), transform))

    # Resolve output artifact filenames
    suffix = f"_{args.output_suffix.lstrip('_')}" if args.output_suffix else ""
    eval_jsonl_filename = args.eval_records_filename or f"eval_records{suffix}.jsonl"
    eval_jsonl_path = out_dir / eval_jsonl_filename

    confusion_csv_filename = (
        args.confusion_matrix_filename or f"confusion_matrix{suffix}.csv"
    )
    confusion_csv_path = out_dir / confusion_csv_filename

    cost_json_filename = (
        args.cost_matrix_filename or f"confusion_cost_matrix{suffix}.json"
    )
    cost_json_path = out_dir / cost_json_filename

    heatmap_path = out_dir / f"confusion_vs_cost_heatmap{suffix}.png"
    mesh_path = out_dir / f"confusion_manifold_3d{suffix}.png"
    drift_path = out_dir / f"snr_drift{suffix}.png"

    # 4. Run stream evaluation
    logger.info(
        "Streaming evaluation across %d conditions to %s...",
        len(perturbations_by_tier),
        eval_jsonl_path,
    )
    records = evaluator.stream_evaluate_dataset(
        samples=samples,
        perturbations_by_tier=perturbations_by_tier,
        output_jsonl_path=eval_jsonl_path,
        resume=not args.no_resume,
    )
    logger.info(
        "Completed evaluation: %d total evaluation record(s) processed.", len(records)
    )

    # 5. Accumulate confusion matrix
    logger.info("Accumulating character alignments and top-K candidate emissions...")
    dataset_vocab = sorted(list(set("".join(s["reference"] for s in samples))))
    logger.info(
        "Active dataset vocabulary (%d tokens): %s", len(dataset_vocab), dataset_vocab
    )
    accumulator = ConfusionAccumulator(vocab=dataset_vocab)
    for rec in records:
        aligned = character_levenshtein_align(rec.reference, rec.hypothesis)
        accumulator.update_from_alignment(aligned, top_k_per_hyp_char=rec.top_k_tokens)

    cond_probs, labels = accumulator.get_conditional_probabilities(
        dirichlet_alpha=args.dirichlet_alpha
    )

    # Export confusion matrix CSV
    with open(confusion_csv_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, ref in enumerate(labels):
            row_vals = [f"{cond_probs[i, j]:.6f}" for j in range(len(labels))]
            f.write(f"{ref}," + ",".join(row_vals) + "\n")
    logger.info("Saved confusion matrix CSV to %s", confusion_csv_path)

    # 6. Compute logarithmic substitution costs
    logger.info(
        "Computing normalized logarithmic distance costs via ConfusionCostEngine..."
    )
    cost_engine = ConfusionCostEngine()
    raw_counts, _ = accumulator.get_raw_counts(labels=labels)
    cost_dict = cost_engine.compute_costs(
        cond_probs=cond_probs,
        labels=labels,
        raw_counts=raw_counts,
    )
    cost_engine.save_cost_artifact(cost_dict, cost_json_path)
    logger.info("Saved cost matrix artifact JSON to %s", cost_json_path)

    # 7. Phonetic Manifold Analysis & Block-Diagonal Permutation
    if len(labels) >= 2:
        logger.info(
            "Extracting phonetic manifold topology and block-diagonal ordering..."
        )
        analyzer = PhoneticManifoldAnalyzer(cond_probs, labels)
        permuted_indices, permuted_labels = analyzer.get_block_diagonal_reordering()
        component_slices = analyzer.get_component_slices()

        perm_conf = cond_probs[np.ix_(permuted_indices, permuted_indices)]
        cost_matrix = np.zeros_like(cond_probs)
        for i, ref in enumerate(labels):
            for j, hyp in enumerate(labels):
                cost_matrix[i, j] = cost_dict["unigram_costs"][ref][hyp]
        perm_cost = cost_matrix[np.ix_(permuted_indices, permuted_indices)]

        # 8. Render Visualizations
        logger.info(
            "Rendering side-by-side confusion & cost heatmaps to %s...", heatmap_path
        )
        ManifoldVisualizer.plot_side_by_side(
            perm_conf,
            perm_cost,
            permuted_labels,
            save_path=heatmap_path,
            component_slices=component_slices,
            title="Phonetic Confusion Probabilities vs. Substitution Costs",
        )

        logger.info("Rendering 3D confusion manifold mesh to %s...", mesh_path)
        ManifoldVisualizer.plot_3d_confusion_mesh(
            perm_conf,
            permuted_labels,
            save_path=mesh_path,
            title="3D Phonetic Confusion Probability Surface",
        )
    else:
        logger.warning(
            "Vocabulary size < 2; skipping heatmap and 3D manifold visualizer plots."
        )

    # SNR Drift Plot
    unique_snrs = sorted(list(set(args.snrs)), reverse=True)
    snr_comp_counts: list[int] = []
    snr_cer_scores: list[float] = []

    for snr in unique_snrs:
        snr_recs = [r for r in records if abs(r.snr_tier - snr) < 1e-4]
        avg_cer = float(np.mean([r.cer for r in snr_recs])) if snr_recs else 0.0
        snr_cer_scores.append(avg_cer)

        if len(labels) >= 2:
            snr_accum = ConfusionAccumulator(vocab=labels)
            for r in snr_recs:
                snr_align = character_levenshtein_align(r.reference, r.hypothesis)
                snr_accum.update_from_alignment(
                    snr_align, top_k_per_hyp_char=r.top_k_tokens
                )
            snr_cond, _ = snr_accum.get_conditional_probabilities(labels=labels)
            snr_an = PhoneticManifoldAnalyzer(snr_cond, labels)
            n_comp = len(snr_an.get_connected_components())
        else:
            n_comp = 1
        snr_comp_counts.append(n_comp)

    logger.info("Rendering SNR drift diagnostics to %s...", drift_path)
    ManifoldVisualizer.plot_snr_drift(
        snr_levels=unique_snrs,
        component_counts=snr_comp_counts,
        cer_scores=snr_cer_scores,
        save_path=drift_path,
        title="Phonetic Manifold Stability & Error Drift across SNR Tiers",
    )

    # 9. Summary Report
    print("\n" + "=" * 60)
    print("NOISY ASR EVALUATION & MANIFOLD REPORT")
    print("=" * 60)
    print(f"Total Evaluation Records: {len(records)}")
    print(f"Active Phonetic Vocabulary: {len(labels)} tokens")
    print("\n--- Performance by SNR Tier ---")
    print(f"{'SNR (dB)':<12} {'Noise Type':<15} {'Samples':<10} {'Mean CER':<10}")
    print("-" * 50)
    for snr in sorted(list(set(args.snrs)), reverse=True):
        for ntype in args.noise_types:
            tier_recs = [
                r
                for r in records
                if abs(r.snr_tier - snr) < 1e-4 and r.noise_type == ntype
            ]
            if tier_recs:
                mean_cer = float(np.mean([r.cer for r in tier_recs]))
                print(
                    f"{snr:<12.1f} {ntype:<15} {len(tier_recs):<10} {mean_cer:<10.4f}"
                )

    print("\n--- Generated Artifacts ---")
    print(f"  • Eval Records JSONL: {eval_jsonl_path}")
    print(f"  • Confusion Matrix:   {confusion_csv_path}")
    print(f"  • Cost Matrix JSON:   {cost_json_path}")
    print(f"  • Heatmaps Plot:      {heatmap_path}")
    print(f"  • SNR Drift Plot:     {drift_path}")
    print(f"  • 3D Manifold Plot:   {mesh_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
