#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_local_checkpoints.py

Evaluates all local checkpoints in a directory on the test dataset.
Computes standard metrics as well as vowel-length-masked and tone-masked metrics.
Avoids downloading from Hugging Face Hub.
Uses centralized greedy decoding.
"""

import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import Dataset, Audio, Features, Value
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from tqdm import tqdm

from transcription.inference.infer import (
    greedy_inference,
    normalize_text,
    strip_tones,
    strip_length,
    strip_both,
    TARGET_SAMPLE_RATE,
)


def _try_read_csv(path):
    for sep in [",", "\t", ";", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    return pd.read_csv(path)


def _detect_columns(df):
    audio_candidates = [
        "path",
        "audio",
        "wav",
        "file",
        "filename",
        "filepath",
        "audio_path",
    ]
    text_candidates = [
        "sentence",
        "text",
        "transcription",
        "transcript",
        "label",
        "target",
    ]
    cols_lower = {c.lower(): c for c in df.columns}

    audio_col = None
    text_col = None
    for cand in audio_candidates:
        if cand in cols_lower:
            audio_col = cols_lower[cand]
            break
    for cand in text_candidates:
        if cand in cols_lower:
            text_col = cols_lower[cand]
            break
    if audio_col is None or text_col is None:
        raise ValueError(f"Could not auto-detect columns from: {list(df.columns)}")
    return audio_col, text_col


def _resolve_audio_path(p, dataset_path):
    p = str(p).strip()
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand = os.path.join(dataset_path, p)
    if os.path.exists(cand):
        return cand
    for sub in ["wavs", "clips", "audio", "data"]:
        cand2 = os.path.join(dataset_path, sub, os.path.basename(p))
        if os.path.exists(cand2):
            return cand2
    return p


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all local Wav2Vec2 checkpoints on a test dataset."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="output_w2v2/wav2vec2-large-xlsr",
        help="Directory containing checkpoint directories.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="training_data/processed/cim-wav2vec2-test.csv",
        help="Path to the test CSV file.",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="training_data/processed/sentence_audio",
        help="Directory containing audio files.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/results/local_checkpoint_scores.csv",
        help="Output path for the scoring CSV.",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default=None,
        help="Path or HF repo ID to the processor (defaults to checkpoints-dir).",
    )
    args = parser.parse_args()

    print(f"Loading test CSV: {args.test_csv}")
    df_test = _try_read_csv(args.test_csv)
    audio_col, text_col = _detect_columns(df_test)
    print(f"Using audio column: '{audio_col}' | text column: '{text_col}'")

    df_test[audio_col] = df_test[audio_col].apply(
        lambda p: _resolve_audio_path(p, args.audio_dir)
    )
    df_test[text_col] = df_test[text_col].apply(normalize_text)
    df_test.dropna(subset=[audio_col, text_col], inplace=True)
    print(f"Number of test items: {len(df_test)}")

    from transcription.utils.evaluation import yield_local_checkpoints, run_evaluation

    # Prepare Dataset
    print("Preparing HuggingFace dataset...")
    data_dict = {
        "audio": df_test[audio_col].tolist(),
        "sentence": df_test[text_col].tolist(),
    }
    features = Features(
        {"audio": Audio(sampling_rate=TARGET_SAMPLE_RATE), "sentence": Value("string")}
    )
    test_ds = Dataset.from_dict(data_dict, features=features)

    processor_path = args.processor
    if not processor_path:
        ckpt_dirs = glob.glob(os.path.join(args.checkpoints_dir, "checkpoint-*"))

        def _step(p):
            m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
            return int(m.group(1)) if m else -1

        ckpt_dirs = sorted(ckpt_dirs, key=_step)
        candidates = [args.checkpoints_dir]
        if ckpt_dirs:
            candidates.append(ckpt_dirs[0])
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "vocab.json")):
                processor_path = cand
                break
        if not processor_path:
            processor_path = args.checkpoints_dir

    print(f"Loading processor for preparing dataset from {processor_path}...")
    processor = Wav2Vec2Processor.from_pretrained(processor_path)

    def prepare_batch(batch):
        audio = batch["audio"]
        input_vals = processor(
            audio["array"], sampling_rate=TARGET_SAMPLE_RATE  # type: ignore
        ).input_values  # type: ignore
        batch["input_values"] = input_vals[0]
        return batch

    test_ds_prepared = test_ds.map(
        prepare_batch,
        remove_columns=[c for c in test_ds.column_names if c != "sentence"],
        num_proc=1,
    )

    # Yield checkpoints and evaluate
    generator = yield_local_checkpoints(args.checkpoints_dir, args.processor)
    if not generator:
        print(f"No valid checkpoints found in {args.checkpoints_dir}")
        return

    _, ranking_df = run_evaluation(generator, test_ds_prepared)

    if ranking_df.empty:
        print("\nNo checkpoints were successfully evaluated.")
        return

    # Map the output ranking columns to match evaluate_local_checkpoints expected output
    scores_df = ranking_df.rename(
        columns={
            "agg_wer_greedy": "greedy_wer",
            "agg_cer_greedy": "greedy_cer",
            "agg_wer_greedy_masked": "greedy_vowel_masked_wer",
            "agg_cer_greedy_masked": "greedy_vowel_masked_cer",
            "agg_wer_tone_masked": "greedy_tone_masked_wer",
            "agg_cer_tone_masked": "greedy_tone_masked_cer",
            "agg_wer_both_masked": "greedy_both_masked_wer",
            "agg_cer_both_masked": "greedy_both_masked_cer",
        }
    )

    # Save scores to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    scores_df.to_csv(args.output_csv, index=False)
    print(f"\nSuccessfully saved scoring results to {args.output_csv}")
    print("\nRanking table (ordered by greedy_wer):")
    print(
        scores_df[
            [
                "checkpoint",
                "greedy_wer",
                "greedy_cer",
                "greedy_vowel_masked_wer",
                "greedy_vowel_masked_cer",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
