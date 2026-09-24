# -*- coding: utf-8 -*-
"""trainer_w2v2_local.py

Local adaptation of the Wav2Vec2 training script.
Supports local CSV files, configurable local paths, and subprocess/OS-based calls.
"""

import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import evaluate
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, Dataset, Features, Value, interleave_datasets
from huggingface_hub import snapshot_download
from jiwer import cer as jiwer_cer, wer as jiwer_wer
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from digohwelisgi.utils.evaluation import (
    run_evaluation,
    strip_length,
    yield_local_checkpoints,
)

TARGET_SAMPLE_RATE = 16000
apostrophe_variants = r"[’‘ʼʻ`´‛]"  # curly, modifier letter, grave/acute, etc.
chars_to_remove_regex = r"[\,\?\.\!\-\;\:\"\“\%\”\\(\)\[\]\{\}«»…]"

# CONFIGURATION DICTIONARY
CONFIG = {
    "train_orig_csv": "training_data/processed/cim-wav2vec2-train.csv",
    "train_bible_csv": "training_data/processed/bible-wav2vec2-train.csv",
    "valid_orig_csv": "training_data/processed/cim-wav2vec2-valid.csv",
    "valid_bible_csv": "training_data/processed/bible-wav2vec2-valid.csv",
    "test_orig_csv": "training_data/processed/cim-wav2vec2-test.csv",
    "test_bible_csv": "training_data/processed/bible-wav2vec2-test.csv",
    "audio_dir": "training_data/processed/sentence_audio",
    "output_dir": "output_w2v2",
    "base_checkpoint": "facebook/wav2vec2-large-xlsr-53",
    "asr_lang": "cim",
    "run_id": "01",
    "epochs": 50,
    "ngrams": 4,
    "lmplz_path": "lmplz",  # Expected in system PATH, or specify full local path (e.g. /usr/local/bin/lmplz)
    "audio_column": None,  # Auto-detects columns like 'path', 'wav'
    "text_column": None,  # Auto-detects columns like 'sentence', 'text'
    "max_steps": -1,
    "eval_batch_size": 16,
}


def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(apostrophe_variants, "'", text)  # unify apostrophes -> '
    text = re.sub(chars_to_remove_regex, "", text)  # remove other punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _try_read_csv(path):
    for sep in [",", "\t", ";", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    return pd.read_csv(path)


def _detect_columns(df, audio_col_override=None, text_col_override=None):
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

    audio_col = audio_col_override
    text_col = text_col_override
    if audio_col is None:
        for cand in audio_candidates:
            if cand in cols_lower:
                audio_col = cols_lower[cand]
                break
    if text_col is None:
        for cand in text_candidates:
            if cand in cols_lower:
                text_col = cols_lower[cand]
                break

    if audio_col is None or text_col is None:
        raise ValueError(
            f"Could not auto-detect columns. Found columns: {list(df.columns)}. "
            f"Please specify audio_column and text_column in CONFIG."
        )
    return audio_col, text_col


def _resolve_audio_path(p, dataset_path):
    p = str(p).strip()
    if os.path.isabs(p) and os.path.exists(p):
        return p
    # Try direct relative to CSV parent directory / dataset path
    cand = os.path.join(dataset_path, p)
    if os.path.exists(cand):
        return cand
    # try a 'clips' or 'wavs' subfolder fallback
    for sub in ["wavs", "clips", "audio", "data"]:
        cand2 = os.path.join(dataset_path, sub, os.path.basename(p))
        if os.path.exists(cand2):
            return cand2
    return p


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.pad(
            labels=label_features, padding=self.padding, return_tensors="pt"
        )
        if batch is None or labels_batch is None:
            raise ValueError("Processor padding returned None.")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels
        return batch  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Wav2Vec2 on local or remote machine."
    )
    parser.add_argument(
        "--train-orig-csv",
        type=str,
        default=CONFIG["train_orig_csv"],
        help="Path to original training CSV split.",
    )
    parser.add_argument(
        "--train-bible-csv",
        type=str,
        default=CONFIG["train_bible_csv"],
        help="Path to Bible training CSV split.",
    )
    parser.add_argument(
        "--valid-orig-csv",
        type=str,
        default=CONFIG["valid_orig_csv"],
        help="Path to original validation CSV split.",
    )
    parser.add_argument(
        "--valid-bible-csv",
        type=str,
        default=CONFIG["valid_bible_csv"],
        help="Path to Bible validation CSV split.",
    )
    parser.add_argument(
        "--test-orig-csv",
        type=str,
        default=CONFIG["test_orig_csv"],
        help="Path to original test CSV split.",
    )
    parser.add_argument(
        "--test-bible-csv",
        type=str,
        default=CONFIG["test_bible_csv"],
        help="Path to Bible test CSV split.",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=CONFIG["audio_dir"],
        help="Directory containing audio files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=CONFIG["output_dir"],
        help="Output directory for logs and models.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG["epochs"],
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lmplz-path",
        type=str,
        default=CONFIG["lmplz_path"],
        help="Path to KenLM lmplz binary.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=CONFIG["max_steps"],
        help="Max training steps (-1 for unlimited).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push checkpoints to Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Hugging Face Hub model ID (e.g. username/model_name).",
    )
    parser.add_argument(
        "--hub-token",
        type=str,
        default=None,
        help="Hugging Face Hub Authentication Token.",
    )
    parser.add_argument(
        "--resume-from-repo",
        type=str,
        default=None,
        help="Hugging Face repo ID to resume training from.",
    )
    parser.add_argument(
        "--resume-from-revision",
        type=str,
        default=None,
        help="Hugging Face repo revision (commit hash or branch) to resume training from.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to local checkpoint or 'latest' to resume from the latest local checkpoint.",
    )
    args = parser.parse_args()

    CONFIG["train_orig_csv"] = args.train_orig_csv
    CONFIG["train_bible_csv"] = args.train_bible_csv
    CONFIG["valid_orig_csv"] = args.valid_orig_csv
    CONFIG["valid_bible_csv"] = args.valid_bible_csv
    CONFIG["test_orig_csv"] = args.test_orig_csv
    CONFIG["test_bible_csv"] = args.test_bible_csv
    CONFIG["audio_dir"] = args.audio_dir
    CONFIG["output_dir"] = args.output_dir
    CONFIG["epochs"] = args.epochs
    CONFIG["lmplz_path"] = args.lmplz_path
    CONFIG["max_steps"] = args.max_steps
    CONFIG["push_to_hub"] = args.push_to_hub
    CONFIG["hub_model_id"] = args.hub_model_id
    CONFIG["hub_token"] = (
        args.hub_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    if CONFIG["push_to_hub"]:
        run_start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        prefix = f"length-only-{run_start_time}"
        base_name = (
            CONFIG["hub_model_id"] if CONFIG["hub_model_id"] else "wav2vec2-large-xlsr"
        )
        if "/" in base_name:
            parts = base_name.split("/")
            if len(parts) == 2:
                CONFIG["hub_model_id"] = f"{parts[0]}/{prefix}-{parts[1]}"
            else:
                CONFIG["hub_model_id"] = f"{prefix}-{base_name}"
        else:
            CONFIG["hub_model_id"] = f"{prefix}-{base_name}"
        print(f"Hugging Face Hub Model ID set to: {CONFIG['hub_model_id']}")

    return args


def setup_directories():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    folder_log_files = os.path.join(CONFIG["output_dir"], "logs-wav2vec2-res")
    folder_model_files = os.path.join(CONFIG["output_dir"], "wav2vec2-large-xlsr")
    os.makedirs(folder_log_files, exist_ok=True)
    os.makedirs(folder_model_files, exist_ok=True)

    print(f"Logs folder: {folder_log_files}")
    print(f"Model folder: {folder_model_files}")
    return folder_log_files, folder_model_files


def load_and_prepare_csvs():
    print("Loading 6 CSV splits...")
    dfs = {
        "train_orig": _try_read_csv(CONFIG["train_orig_csv"]),
        "train_bible": _try_read_csv(CONFIG["train_bible_csv"]),
        "valid_orig": _try_read_csv(CONFIG["valid_orig_csv"]),
        "valid_bible": _try_read_csv(CONFIG["valid_bible_csv"]),
        "test_orig": _try_read_csv(CONFIG["test_orig_csv"]),
        "test_bible": _try_read_csv(CONFIG["test_bible_csv"]),
    }

    audio_col, text_col = _detect_columns(
        dfs["train_orig"], CONFIG["audio_column"], CONFIG["text_column"]
    )
    print(f"Detected columns -> Audio: '{audio_col}' | Text: '{text_col}'")

    for key, df in dfs.items():
        df[audio_col] = df[audio_col].apply(
            lambda p: _resolve_audio_path(p, CONFIG["audio_dir"])
        )
        df[text_col] = df[text_col].apply(normalize_text)
        df.dropna(subset=[audio_col, text_col], inplace=True)
        print(f"Split {key}: {len(df)} samples")

    missing = [
        p for p in dfs["train_orig"][audio_col].tolist()[:50] if not os.path.exists(p)
    ]
    if missing:
        print("WARNING: some audio files not found, e.g.:", missing[:5])
    else:
        print("Audio path checks passed (sampled).")

    return dfs, audio_col, text_col


def build_vocabulary_and_processor(dfs, text_col, folder_model_files):
    print("Building vocabulary across all 6 CSV splits...")
    all_text = " ".join(pd.concat([df[text_col] for df in dfs.values()]).tolist())
    vocab = sorted(set(all_text))
    vocab_dict = {v: k for k, v in enumerate(vocab)}
    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    vocab_path = os.path.join(CONFIG["output_dir"], "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)
    print(f"Vocab saved (size={len(vocab_dict)}).")

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=TARGET_SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    processor.save_pretrained(folder_model_files)
    print("Skipping KenLM language model building as requested.")
    return processor


def prepare_datasets(dfs, audio_col, text_col, processor):
    print("Preparing HuggingFace Datasets and interleaving train splits...")

    def df_to_ds(df):
        data_dict = {
            "audio_path": df[audio_col].tolist(),
            "sentence": df[text_col].tolist(),
        }
        return Dataset.from_dict(data_dict)

    def prepare_batch(batch):
        audio_array, sr = sf.read(batch["audio_path"])
        batch["input_values"] = processor(audio_array, sampling_rate=sr).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch

    # Prepare individual splits
    ds_train_orig = df_to_ds(dfs["train_orig"]).map(
        prepare_batch, remove_columns=["audio_path", "sentence"], num_proc=1
    )
    ds_train_bible = df_to_ds(dfs["train_bible"]).map(
        prepare_batch, remove_columns=["audio_path", "sentence"], num_proc=1
    )

    ds_valid_orig = df_to_ds(dfs["valid_orig"]).map(
        prepare_batch, remove_columns=["audio_path", "sentence"], num_proc=1
    )
    ds_valid_bible = df_to_ds(dfs["valid_bible"]).map(
        prepare_batch, remove_columns=["audio_path", "sentence"], num_proc=1
    )

    ds_test_orig = df_to_ds(dfs["test_orig"]).map(
        prepare_batch,
        remove_columns=[
            c for c in ["audio_path"] if c in df_to_ds(dfs["test_orig"]).column_names
        ],
        num_proc=1,
    )
    ds_test_bible = df_to_ds(dfs["test_bible"]).map(
        prepare_batch,
        remove_columns=[
            c for c in ["audio_path"] if c in df_to_ds(dfs["test_bible"]).column_names
        ],
        num_proc=1,
    )

    # 1. Live 50/50 Interleaved Train Stream
    train_ds = interleave_datasets(
        [ds_train_orig, ds_train_bible],
        probabilities=[0.5, 0.5],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    MAX_INPUT_LENGTH = TARGET_SAMPLE_RATE * 20
    train_ds = train_ds.filter(
        lambda x: x < MAX_INPUT_LENGTH, input_columns=["input_length"]
    )

    # 2. Validation set strategy: Anchor on original domain for early stopping
    valid_ds = ds_valid_orig

    return train_ds, valid_ds, ds_test_orig, ds_test_bible


def initialize_model_and_trainer(processor, train_ds, valid_ds, folder_model_files):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    print(f"Loading base checkpoint: {CONFIG['base_checkpoint']}")
    model = Wav2Vec2ForCTC.from_pretrained(
        CONFIG["base_checkpoint"],
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    model.freeze_feature_encoder()  # type: ignore

    training_args = TrainingArguments(
        output_dir=folder_model_files,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=400,
        num_train_epochs=CONFIG["epochs"],
        fp16=torch.cuda.is_available(),
        learning_rate=3e-4,
        warmup_ratio=0.1,
        save_total_limit=20,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        group_by_length=True,
        length_column_name="input_length",
        report_to="none",
        max_steps=CONFIG["max_steps"],
        push_to_hub=CONFIG["push_to_hub"],
        hub_model_id=CONFIG["hub_model_id"],
        hub_token=CONFIG["hub_token"],
        hub_private_repo=True,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    return trainer, data_collator


def resolve_resume_checkpoint(args, folder_model_files):
    resume_checkpoint = None
    if args.resume_from_repo:
        print(
            f"Downloading checkpoint from repository: {args.resume_from_repo} (revision: {args.resume_from_revision or 'main'})"
        )
        downloaded_dir = snapshot_download(
            repo_id=args.resume_from_repo,
            revision=args.resume_from_revision,
            token=CONFIG["hub_token"],
        )
        print(f"Checkpoint downloaded to: {downloaded_dir}")
        if os.path.exists(os.path.join(downloaded_dir, "trainer_state.json")):
            resume_checkpoint = downloaded_dir
            print(
                f"Using downloaded repository root as checkpoint: {resume_checkpoint}"
            )
        else:
            ckpt_dirs = glob.glob(os.path.join(downloaded_dir, "checkpoint-*"))
            if ckpt_dirs:

                def _step(p):
                    m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
                    return int(m.group(1)) if m else -1

                resume_checkpoint = max(ckpt_dirs, key=_step)
                print(
                    f"Found latest checkpoint subdirectory in snapshot: {resume_checkpoint}"
                )
            else:
                print(
                    f"WARNING: No trainer_state.json or checkpoint-* directory found in downloaded snapshot {downloaded_dir}."
                )
                print(
                    "Cannot resume training state (optimizers/scheduler). Setting resume_checkpoint to None."
                )
                print(
                    f"Updating base_checkpoint to load model weights from: {downloaded_dir}"
                )
                CONFIG["base_checkpoint"] = downloaded_dir
                resume_checkpoint = None
    elif args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "latest":
            ckpt_dirs = glob.glob(os.path.join(folder_model_files, "checkpoint-*"))
            if ckpt_dirs:

                def _step(p):
                    m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
                    return int(m.group(1)) if m else -1

                resume_checkpoint = max(ckpt_dirs, key=_step)
                print(f"Found latest local checkpoint: {resume_checkpoint}")
            else:
                print(
                    "No local checkpoints found under the output directory. Starting from scratch."
                )
                resume_checkpoint = None
        else:
            resume_checkpoint = args.resume_from_checkpoint
            print(f"Resuming from local checkpoint: {resume_checkpoint}")

    return resume_checkpoint


def train_model(trainer, resume_checkpoint, folder_model_files, processor):
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(folder_model_files)
    processor.save_pretrained(folder_model_files)
    print("Training complete. Base model saved.")


def evaluate_checkpoints_dual(
    folder_model_files, test_orig_ds, test_bible_ds, data_collator, processor
):
    print("\n==========================================")
    print("STARTING DISAGGREGATED POST-TRAINING EVALUATION")
    print("==========================================\n")

    # Pass 1: Original Test Set (Baseline Progress & Forgetfulness Anchor)
    print("Evaluating checkpoints on ORIGINAL Test Set...")
    gen_orig = yield_local_checkpoints(
        folder_model_files, processor_path=folder_model_files
    )
    rows_orig, ranking_orig = run_evaluation(
        gen_orig,
        test_orig_ds,
        data_collator,
        batch_size=CONFIG.get("eval_batch_size", 16),
    )

    # Pass 2: Bible Test Set (Unseen Chapters/Books)
    print("Evaluating checkpoints on BIBLE Test Set...")
    gen_bible = yield_local_checkpoints(
        folder_model_files, processor_path=folder_model_files
    )
    rows_bible, ranking_bible = run_evaluation(
        gen_bible,
        test_bible_ds,
        data_collator,
        batch_size=CONFIG.get("eval_batch_size", 16),
    )

    # Summarize best checkpoints on original test set
    ranking_unmasked = ranking_orig.sort_values(
        by=[
            "median_wer_greedy",
            "median_cer_greedy",
            "agg_wer_greedy",
            "agg_cer_greedy",
        ],
        ascending=True,
    ).reset_index(drop=True)
    best_unmasked_ckpt = ranking_unmasked.iloc[0]["checkpoint"]
    best_unmasked_wer = ranking_unmasked.iloc[0]["agg_wer_greedy"]

    ranking_masked = ranking_orig.sort_values(
        by=[
            "median_wer_greedy_masked",
            "median_cer_greedy_masked",
            "agg_wer_greedy_masked",
            "agg_cer_greedy_masked",
        ],
        ascending=True,
    ).reset_index(drop=True)
    best_masked_ckpt = ranking_masked.iloc[0]["checkpoint"]
    best_masked_wer = ranking_masked.iloc[0]["agg_wer_greedy_masked"]

    print(f"\n==========================================")
    print(f"ORIGINAL TEST SET EVALUATION SUMMARY:")
    print(
        f"Best Unmasked Checkpoint: {best_unmasked_ckpt} (WER: {best_unmasked_wer:.4f})"
    )
    print(f"Best Masked Checkpoint:   {best_masked_ckpt} (WER: {best_masked_wer:.4f})")
    print(f"==========================================\n")

    best_ckpt_label = best_unmasked_ckpt
    best_ckpt_path = ranking_orig[ranking_orig["checkpoint"] == best_ckpt_label].iloc[
        0
    ]["path"]

    return (
        best_unmasked_ckpt,
        best_unmasked_wer,
        best_masked_ckpt,
        best_masked_wer,
        best_ckpt_label,
        best_ckpt_path,
        rows_orig,
        ranking_orig,
        rows_bible,
        ranking_bible,
    )


def promote_best_checkpoint(
    best_ckpt_label, best_ckpt_path, folder_model_files, processor
):
    print(f"\nBest checkpoint identified (for promotion): {best_ckpt_label}")
    for fname in os.listdir(best_ckpt_path):
        if fname in (
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
            "training_args.bin",
            "scaler.pt",
        ):
            continue
        src = os.path.join(best_ckpt_path, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(folder_model_files, fname))

    processor.save_pretrained(folder_model_files)
    print(f"Promoted {best_ckpt_label} to final model directory.")


def save_results_summary(
    rows_orig,
    ranking_orig,
    rows_bible,
    ranking_bible,
    best_unmasked_ckpt,
    best_unmasked_wer,
    best_masked_ckpt,
    best_masked_wer,
    best_ckpt_label,
    folder_log_files,
):
    output_prefix = f"{CONFIG['asr_lang']}-wav2vec2"

    # Original test set results
    all_rows_orig = []
    for ckpt_label, rows in rows_orig.items():
        all_rows_orig.extend(rows)
    results_orig_df = pd.DataFrame(all_rows_orig)
    per_sentence_orig_csv = os.path.join(
        folder_log_files,
        f"{output_prefix}-run{CONFIG['run_id']}-orig-test-results.csv",
    )
    results_orig_df.to_csv(per_sentence_orig_csv, index=False, encoding="utf-8")

    # Bible test set results
    all_rows_bible = []
    for ckpt_label, rows in rows_bible.items():
        all_rows_bible.extend(rows)
    results_bible_df = pd.DataFrame(all_rows_bible)
    per_sentence_bible_csv = os.path.join(
        folder_log_files,
        f"{output_prefix}-run{CONFIG['run_id']}-bible-test-results.csv",
    )
    results_bible_df.to_csv(per_sentence_bible_csv, index=False, encoding="utf-8")

    summary_txt = os.path.join(
        folder_log_files, f"{output_prefix}-run{CONFIG['run_id']}-summary.txt"
    )
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"ASR Language: {CONFIG['asr_lang']}\nRun ID: {CONFIG['run_id']}\n")
        f.write(
            f"Best Unmasked Checkpoint (Orig): {best_unmasked_ckpt} (WER: {best_unmasked_wer:.4f})\n"
        )
        f.write(
            f"Best Masked Checkpoint (Orig):   {best_masked_ckpt} (WER: {best_masked_wer:.4f})\n"
        )
        f.write(f"Promoted checkpoint: {best_ckpt_label}\n\n")
        f.write(f"ORIGINAL TEST SET RANKING:\n{ranking_orig.to_string()}\n\n")
        f.write(f"BIBLE TEST SET RANKING:\n{ranking_bible.to_string()}\n")
    print(f"Summary written to {summary_txt}")


def main():
    args = parse_args()
    folder_log_files, folder_model_files = setup_directories()
    dfs, audio_col, text_col = load_and_prepare_csvs()
    processor = build_vocabulary_and_processor(dfs, text_col, folder_model_files)
    train_ds, valid_ds, ds_test_orig, ds_test_bible = prepare_datasets(
        dfs, audio_col, text_col, processor
    )
    resume_checkpoint = resolve_resume_checkpoint(args, folder_model_files)
    trainer, data_collator = initialize_model_and_trainer(
        processor, train_ds, valid_ds, folder_model_files
    )
    train_model(trainer, resume_checkpoint, folder_model_files, processor)

    (
        best_unmasked_ckpt,
        best_unmasked_wer,
        best_masked_ckpt,
        best_masked_wer,
        best_ckpt_label,
        best_ckpt_path,
        rows_orig,
        ranking_orig,
        rows_bible,
        ranking_bible,
    ) = evaluate_checkpoints_dual(
        folder_model_files, ds_test_orig, ds_test_bible, data_collator, processor
    )

    promote_best_checkpoint(
        best_ckpt_label, best_ckpt_path, folder_model_files, processor
    )

    save_results_summary(
        rows_orig,
        ranking_orig,
        rows_bible,
        ranking_bible,
        best_unmasked_ckpt,
        best_unmasked_wer,
        best_masked_ckpt,
        best_masked_wer,
        best_ckpt_label,
        folder_log_files,
    )


if __name__ == "__main__":
    # Prevent OpenMP duplicate initialization crash on macOS conda environments
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Enable CPU fallback for ops missing native MPS implementation
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    main()
