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

# Configure safe globals for torch.load weights_only=True compatibility in PyTorch 2.6+
if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
    import numpy as np

    globals_to_add = [
        np.ndarray,
        np.dtype,
    ]
    multiarray_mod = getattr(np, "_core", getattr(np, "core", None))
    if multiarray_mod and hasattr(multiarray_mod, "multiarray"):
        globals_to_add.append(multiarray_mod.multiarray._reconstruct)
    try:
        import numpy.random._pickle

        globals_to_add.append(numpy.random._pickle.__randomstate_ctor)
    except (ImportError, AttributeError):
        pass
    if hasattr(np, "dtypes") and hasattr(np.dtypes, "UInt32DType"):
        globals_to_add.append(np.dtypes.UInt32DType)
    torch.serialization.add_safe_globals(globals_to_add)

import torchaudio
from datasets import (
    Audio,
    Dataset,
    Features,
    Value,
    concatenate_datasets,
    interleave_datasets,
)
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

from digohwelisgi.cherokee.models.dual_head import (
    DualDataCollatorCTCWithPadding,
    Wav2Vec2ForDualCTC,
)
from digohwelisgi.cherokee.orthography.tones import (
    MORAIC_TONE_VOCAB,
    SEGMENTAL_CV_VOCAB,
    decode_tone_ids,
    encode_cv_text,
    encode_tone_tokens,
    reconcile_dual_stream,
    save_dual_vocabularies,
)
from digohwelisgi.core.audio import (
    AudioAugmentationConfig,
    AudioAugmenter,
    create_audio_augmenter,
)
from digohwelisgi.utils.evaluation import (
    run_evaluation,
    yield_local_checkpoints,
)

TARGET_SAMPLE_RATE = 16000
apostrophe_variants = r"[’‘ʼʻ`´‛]"  # curly, modifier letter, grave/acute, etc.
chars_to_remove_regex = r"[\,\?\.\!\-\;\:\"\“\%\”\\(\)\[\]\{\}«»…]"

# CONFIGURATION DICTIONARY
CONFIG = {
    "train_orig_csv": "data/training/processed/chr-wav2vec2-train.csv",
    "train_bible_csv": "data/training/processed/bible-wav2vec2-train.csv",
    "valid_orig_csv": "data/training/processed/chr-wav2vec2-valid.csv",
    "valid_bible_csv": "data/training/processed/bible-wav2vec2-valid.csv",
    "test_orig_csv": "data/training/processed/chr-wav2vec2-test.csv",
    "test_bible_csv": "data/training/processed/bible-wav2vec2-test.csv",
    "audio_dir": "data/training/processed/sentence_audio",
    "output_dir": "output_w2v2",
    "base_checkpoint": "facebook/wav2vec2-large-xlsr-53",
    "asr_lang": "chr",
    "run_id": "01",
    "epochs": 30,
    "ngrams": 4,
    "lmplz_path": "lmplz",  # Expected in system PATH, or specify full local path (e.g. /usr/local/bin/lmplz)
    "audio_column": None,  # Auto-detects columns like 'path', 'wav'
    "text_column": None,  # Auto-detects columns like 'cv_text', 'sentence', 'text'
    "max_steps": -1,
    "max_train_samples": None,
    "max_eval_samples": None,
    "batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "eval_strategy": "steps",
    "save_strategy": "steps",
    "eval_steps": 200,
    "save_steps": 400,
    "save_total_limit": 3,
    "hub_strategy": "all_checkpoints",
    "hub_prefix": "dual-tone-moraic",
    "gradient_checkpointing": False,
    "tone_loss_weight": 1.0,
    "num_branch_layers": 2,
    "vocab_tone_path": None,
    "augment": True,
    "time_stretch_min_rate": 0.9,
    "time_stretch_max_rate": 1.1,
    "time_stretch_p": 0.5,
    "pitch_shift_min_semitones": -1.5,
    "pitch_shift_max_semitones": 1.5,
    "pitch_shift_p": 0.5,
    "gaussian_noise_min_amplitude": 0.001,
    "gaussian_noise_max_amplitude": 0.01,
    "gaussian_noise_p": 0.3,
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
        "cv_text",
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
    if os.path.exists(p):
        return p

    # Direct candidate inside dataset_path
    cand = os.path.join(dataset_path, p)
    if os.path.exists(cand):
        return cand

    # Basename directly inside dataset_path
    cand_base = os.path.join(dataset_path, os.path.basename(p))
    if os.path.exists(cand_base):
        return cand_base

    # Subfolders inside dataset_path
    for sub in ["wavs", "clips", "audio", "data", "sentence_audio", "conrad_wavs"]:
        cand_sub = os.path.join(dataset_path, sub, os.path.basename(p))
        if os.path.exists(cand_sub):
            return cand_sub

    # Common repository root data directories (supporting Bible and related audio)
    for root_prefix in ["data", "data/projects", "data/training"]:
        cand_root = os.path.join(root_prefix, p)
        if os.path.exists(cand_root):
            return cand_root

    # Remap legacy training_data -> data/training paths
    if p.startswith("training_data/"):
        migrated = os.path.join("data/training", p[len("training_data/") :])
        if os.path.exists(migrated):
            return migrated

    return p


DataCollatorCTCWithPadding = DualDataCollatorCTCWithPadding


class CTCTrainer(Trainer):
    """Custom Trainer subclass to toggle live audio augmentation and persist vocab_tone.json."""

    def get_train_dataloader(self):
        if hasattr(self.data_collator, "augment"):
            setattr(self.data_collator, "augment", bool(CONFIG.get("augment", True)))
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if hasattr(self.data_collator, "augment"):
            setattr(self.data_collator, "augment", False)
        return super().get_eval_dataloader(eval_dataset)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        target_dir = output_dir or self.args.output_dir
        if target_dir and os.path.isdir(target_dir):
            vocab_tone = getattr(self.model, "vocab_tone", dict(MORAIC_TONE_VOCAB))
            vt_path = os.path.join(target_dir, "vocab_tone.json")
            try:
                with open(vt_path, "w", encoding="utf-8") as vf:
                    json.dump(vocab_tone, vf, ensure_ascii=False, indent=2)
            except Exception:
                pass


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
        "--batch-size",
        type=int,
        default=CONFIG["batch_size"],
        help="Per device train batch size (default: 8).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=CONFIG["eval_batch_size"],
        help="Per device eval batch size (default: 8).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=CONFIG["gradient_accumulation_steps"],
        help="Gradient accumulation steps (default: 2).",
    )
    parser.add_argument(
        "--eval-strategy",
        type=str,
        default=CONFIG["eval_strategy"],
        help="Evaluation strategy ('steps' or 'epoch', default: 'steps').",
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default=CONFIG["save_strategy"],
        help="Save strategy ('steps' or 'epoch', default: 'steps').",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=CONFIG["eval_steps"],
        help="Evaluation step interval when eval_strategy='steps' (default: 100).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=CONFIG["save_steps"],
        help="Save step interval when save_strategy='steps' (default: 400).",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=CONFIG["save_total_limit"],
        help="Maximum number of checkpoints to keep locally on disk (default: 3).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=CONFIG["gradient_checkpointing"],
        help="Enable gradient checkpointing to reduce memory usage during training (default: False).",
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
        "--hub-strategy",
        type=str,
        default=CONFIG["hub_strategy"],
        choices=["all_checkpoints", "every_save", "end", "checkpoint"],
        help="Strategy to use when pushing to Hugging Face Hub (default: 'all_checkpoints').",
    )
    parser.add_argument(
        "--hub-prefix",
        type=str,
        default=CONFIG["hub_prefix"],
        help="Prefix prepended to Hub model ID before run timestamp (default: 'length-only').",
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
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["augment"],
        help="Enable live audio data augmentation during training (default: True).",
    )
    parser.add_argument(
        "--time-stretch-p",
        type=float,
        default=CONFIG["time_stretch_p"],
        help="Probability of applying TimeStretch augmentation (default: 0.5).",
    )
    parser.add_argument(
        "--time-stretch-min-rate",
        type=float,
        default=CONFIG["time_stretch_min_rate"],
        help="Minimum rate for TimeStretch (default: 0.9).",
    )
    parser.add_argument(
        "--time-stretch-max-rate",
        type=float,
        default=CONFIG["time_stretch_max_rate"],
        help="Maximum rate for TimeStretch (default: 1.1).",
    )
    parser.add_argument(
        "--pitch-shift-p",
        type=float,
        default=CONFIG["pitch_shift_p"],
        help="Probability of applying PitchShift augmentation (default: 0.5).",
    )
    parser.add_argument(
        "--pitch-shift-min-semitones",
        type=float,
        default=CONFIG["pitch_shift_min_semitones"],
        help="Minimum semitones for PitchShift (default: -2.0).",
    )
    parser.add_argument(
        "--pitch-shift-max-semitones",
        type=float,
        default=CONFIG["pitch_shift_max_semitones"],
        help="Maximum semitones for PitchShift (default: 2.0).",
    )
    parser.add_argument(
        "--gaussian-noise-p",
        type=float,
        default=CONFIG["gaussian_noise_p"],
        help="Probability of applying AddGaussianNoise augmentation (default: 0.3).",
    )
    parser.add_argument(
        "--gaussian-noise-min-amplitude",
        type=float,
        default=CONFIG["gaussian_noise_min_amplitude"],
        help="Minimum amplitude for AddGaussianNoise (default: 0.001).",
    )
    parser.add_argument(
        "--gaussian-noise-max-amplitude",
        type=float,
        default=CONFIG["gaussian_noise_max_amplitude"],
        help="Maximum amplitude for AddGaussianNoise (default: 0.01).",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=CONFIG["base_checkpoint"],
        help="Base checkpoint path or repo (default: facebook/wav2vec2-large-xlsr-53).",
    )
    parser.add_argument(
        "--tone-loss-weight",
        type=float,
        default=CONFIG["tone_loss_weight"],
        help="Multi-task loss weight lambda_tone for Moraic Tone CTC head (default: 1.0).",
    )
    parser.add_argument(
        "--num-branch-layers",
        type=int,
        default=CONFIG["num_branch_layers"],
        help="Number of top Transformer encoder layers forked into Branch A and Branch B (default: 2).",
    )
    parser.add_argument(
        "--vocab-tone-path",
        type=str,
        default=CONFIG["vocab_tone_path"],
        help="Optional custom path to vocab_tone.json.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=CONFIG["max_train_samples"],
        help="Optional cap on training samples per domain for local smoke tests.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=CONFIG["max_eval_samples"],
        help="Optional cap on validation/test samples per domain for local smoke tests.",
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
    CONFIG["base_checkpoint"] = args.base_checkpoint
    CONFIG["epochs"] = args.epochs
    CONFIG["lmplz_path"] = args.lmplz_path
    CONFIG["max_steps"] = args.max_steps
    CONFIG["max_train_samples"] = args.max_train_samples
    CONFIG["max_eval_samples"] = args.max_eval_samples
    CONFIG["batch_size"] = args.batch_size
    CONFIG["eval_batch_size"] = args.eval_batch_size
    CONFIG["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    CONFIG["eval_strategy"] = args.eval_strategy
    CONFIG["save_strategy"] = args.save_strategy
    CONFIG["eval_steps"] = args.eval_steps
    CONFIG["save_steps"] = args.save_steps
    CONFIG["save_total_limit"] = args.save_total_limit
    CONFIG["gradient_checkpointing"] = args.gradient_checkpointing
    CONFIG["tone_loss_weight"] = args.tone_loss_weight
    CONFIG["num_branch_layers"] = args.num_branch_layers
    CONFIG["vocab_tone_path"] = args.vocab_tone_path
    CONFIG["push_to_hub"] = args.push_to_hub
    CONFIG["hub_strategy"] = args.hub_strategy
    CONFIG["hub_prefix"] = args.hub_prefix
    CONFIG["hub_model_id"] = args.hub_model_id
    CONFIG["hub_token"] = (
        args.hub_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    CONFIG["augment"] = args.augment
    CONFIG["time_stretch_p"] = args.time_stretch_p
    CONFIG["time_stretch_min_rate"] = args.time_stretch_min_rate
    CONFIG["time_stretch_max_rate"] = args.time_stretch_max_rate
    CONFIG["pitch_shift_p"] = args.pitch_shift_p
    CONFIG["pitch_shift_min_semitones"] = args.pitch_shift_min_semitones
    CONFIG["pitch_shift_max_semitones"] = args.pitch_shift_max_semitones
    CONFIG["gaussian_noise_p"] = args.gaussian_noise_p
    CONFIG["gaussian_noise_min_amplitude"] = args.gaussian_noise_min_amplitude
    CONFIG["gaussian_noise_max_amplitude"] = args.gaussian_noise_max_amplitude

    if CONFIG["push_to_hub"]:
        is_resuming = bool(args.resume_from_checkpoint or args.resume_from_repo)
        base_name = (
            CONFIG["hub_model_id"] if CONFIG["hub_model_id"] else "wav2vec2-large-xlsr"
        )
        if is_resuming and CONFIG["hub_model_id"]:
            CONFIG["hub_model_id"] = base_name
        else:
            run_start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            hub_pfx = str(CONFIG.get("hub_prefix", "")).strip()
            prefix = f"{hub_pfx}-{run_start_time}" if hub_pfx else run_start_time
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
        if "tone_tokens" in df.columns:
            df["tone_tokens"] = df["tone_tokens"].fillna("").astype(str)
        else:
            df["tone_tokens"] = ""
        if "canonical_toned" in df.columns:
            df["canonical_toned"] = df["canonical_toned"].fillna("").astype(str)
        else:
            df["canonical_toned"] = ""
        df.dropna(subset=[audio_col, text_col], inplace=True)

        max_train = CONFIG.get("max_train_samples")
        max_eval = CONFIG.get("max_eval_samples")
        if key.startswith("train_") and max_train is not None and int(max_train) > 0:
            df = df.iloc[: int(max_train)].copy()
            dfs[key] = df
        elif (
            (key.startswith("valid_") or key.startswith("test_"))
            and max_eval is not None
            and int(max_eval) > 0
        ):
            df = df.iloc[: int(max_eval)].copy()
            dfs[key] = df

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
    print(
        "Serializing canonical dual vocabularies (19 CV tokens, 10 Moraic Tone tokens)..."
    )
    vocab_path, vocab_tone_path = save_dual_vocabularies(CONFIG["output_dir"])
    save_dual_vocabularies(folder_model_files)
    if CONFIG.get("vocab_tone_path"):
        custom_vt = str(CONFIG["vocab_tone_path"])
        if os.path.exists(custom_vt) and os.path.abspath(custom_vt) != os.path.abspath(
            str(vocab_tone_path)
        ):
            shutil.copy2(custom_vt, str(vocab_tone_path))
            shutil.copy2(custom_vt, os.path.join(folder_model_files, "vocab_tone.json"))

    print(
        f"Dual vocabularies saved -> CV: {vocab_path} ({len(SEGMENTAL_CV_VOCAB)}), "
        f"Tone: {vocab_tone_path} ({len(MORAIC_TONE_VOCAB)})."
    )

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
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
    setattr(processor, "vocab_tone", dict(MORAIC_TONE_VOCAB))
    processor.save_pretrained(folder_model_files)
    print("Skipping KenLM language model building as requested.")
    return processor


def prepare_datasets(dfs, audio_col, text_col, processor):
    print("Preparing HuggingFace Datasets and interleaving train splits...")

    def df_to_ds(df):
        data_dict = {
            "audio_path": df[audio_col].tolist(),
            "sentence": df[text_col].tolist(),
            "tone_tokens": (
                df["tone_tokens"].fillna("").astype(str).tolist()
                if "tone_tokens" in df.columns
                else [""] * len(df)
            ),
            "canonical_toned": (
                df["canonical_toned"].fillna("").astype(str).tolist()
                if "canonical_toned" in df.columns
                else [""] * len(df)
            ),
        }
        return Dataset.from_dict(data_dict)

    def prepare_batch(batch):
        audio_array, sr = sf.read(batch["audio_path"])
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        batch["input_values"] = processor(audio_array, sampling_rate=sr).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = encode_cv_text(batch["sentence"])
        batch["tone_labels"] = encode_tone_tokens(batch.get("tone_tokens", ""))
        return batch

    train_remove_cols = ["audio_path", "sentence", "tone_tokens", "canonical_toned"]

    # Prepare individual splits
    ds_train_orig = df_to_ds(dfs["train_orig"]).map(
        prepare_batch, remove_columns=train_remove_cols, num_proc=1
    )
    ds_train_bible = df_to_ds(dfs["train_bible"]).map(
        prepare_batch, remove_columns=train_remove_cols, num_proc=1
    )

    ds_valid_orig = df_to_ds(dfs["valid_orig"]).map(
        prepare_batch, remove_columns=train_remove_cols, num_proc=1
    )
    ds_valid_bible = df_to_ds(dfs["valid_bible"]).map(
        prepare_batch, remove_columns=train_remove_cols, num_proc=1
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

    # 2. Merged Validation Set (Original + Bible domains)
    valid_ds = concatenate_datasets([ds_valid_orig, ds_valid_bible])
    valid_ds = valid_ds.filter(
        lambda x: x < MAX_INPUT_LENGTH, input_columns=["input_length"]
    )

    return train_ds, valid_ds, ds_test_orig, ds_test_bible


def compute_dual_metrics(pred, processor, vocab_tone=MORAIC_TONE_VOCAB):
    """
    Computes multi-task evaluation metrics across both branches:
    - `wer` and `cer` on Segmental CV predictions (all samples).
    - `tone_cer` (Tone Error Rate / TER) strictly on tone-annotated samples (`tone_labels != -100`).
    - `reconciled_cer` on tone-annotated samples after DP dual-stream reconciliation.
    """

    def _safe(s: str) -> str:
        s_clean = str(s).strip()
        return s_clean if s_clean else " "

    if isinstance(pred.predictions, (tuple, list)):
        logits_cv = pred.predictions[0]
        logits_tone = pred.predictions[1] if len(pred.predictions) > 1 else None
    else:
        logits_cv = pred.predictions
        logits_tone = None

    if isinstance(pred.label_ids, (tuple, list)):
        labels_cv = np.array(pred.label_ids[0], copy=True)
        labels_tone = (
            np.array(pred.label_ids[1], copy=True) if len(pred.label_ids) > 1 else None
        )
    else:
        labels_cv = np.array(pred.label_ids, copy=True)
        labels_tone = None

    pred_cv_ids = np.argmax(logits_cv, axis=-1)
    pad_id = processor.tokenizer.pad_token_id
    labels_cv[labels_cv == -100] = pad_id

    pred_str = processor.batch_decode(pred_cv_ids)
    label_str = processor.batch_decode(labels_cv, group_tokens=False)

    safe_preds = [_safe(p) for p in pred_str]
    safe_refs = [_safe(r) for r in label_str]
    wer = float(jiwer_wer(safe_refs, safe_preds))
    cer = float(jiwer_cer(safe_refs, safe_preds))

    tone_cer = cer
    reconciled_cer = cer

    if logits_tone is not None and labels_tone is not None:
        valid_mask = (labels_tone != -100).any(axis=-1)
        if np.any(valid_mask):
            hyp_tones: List[str] = []
            ref_tones: List[str] = []
            hyp_recons: List[str] = []
            ref_recons: List[str] = []

            for idx in np.where(valid_mask)[0]:
                pred_t_ids = np.argmax(logits_tone[idx], axis=-1).tolist()
                hyp_t = decode_tone_ids(
                    pred_t_ids,
                    vocab=vocab_tone,
                    collapse_repeats=True,
                    remove_pad=True,
                )
                ref_t_ids = [int(t) for t in labels_tone[idx] if int(t) >= 0]
                ref_t = decode_tone_ids(
                    ref_t_ids,
                    vocab=vocab_tone,
                    collapse_repeats=False,
                    remove_pad=True,
                )

                hyp_tones.append(_safe(hyp_t))
                ref_tones.append(_safe(ref_t))

                hyp_rec = reconcile_dual_stream(pred_str[idx], hyp_t, mode="dp")
                ref_rec = reconcile_dual_stream(label_str[idx], ref_t, mode="dp")
                hyp_recons.append(_safe(hyp_rec))
                ref_recons.append(_safe(ref_rec))

            tone_cer = float(jiwer_cer(ref_tones, hyp_tones))
            reconciled_cer = float(jiwer_cer(ref_recons, hyp_recons))

    return {
        "wer": wer,
        "cer": cer,
        "tone_cer": tone_cer,
        "reconciled_cer": reconciled_cer,
    }


def initialize_model_and_trainer(processor, train_ds, valid_ds, folder_model_files):
    vocab_tone = getattr(processor, "vocab_tone", dict(MORAIC_TONE_VOCAB))

    def compute_metrics(pred):
        return compute_dual_metrics(pred, processor, vocab_tone=vocab_tone)

    print(
        f"Loading base checkpoint into Wav2Vec2ForDualCTC: {CONFIG['base_checkpoint']} "
        f"(num_branch_layers={CONFIG['num_branch_layers']}, tone_loss_weight={CONFIG['tone_loss_weight']})"
    )
    model = Wav2Vec2ForDualCTC.from_pretrained(
        CONFIG["base_checkpoint"],
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        num_branch_layers=int(CONFIG.get("num_branch_layers", 2)),
        tone_vocab_size=len(vocab_tone),
        tone_pad_token_id=int(vocab_tone["[PAD]"]),
        tone_loss_weight=float(CONFIG.get("tone_loss_weight", 1.0)),
        ignore_mismatched_sizes=True,
    )
    setattr(model, "vocab_tone", vocab_tone)

    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=folder_model_files,
        per_device_train_batch_size=CONFIG.get("batch_size", 8),
        per_device_eval_batch_size=CONFIG.get("eval_batch_size", 8),
        gradient_accumulation_steps=CONFIG.get("gradient_accumulation_steps", 2),
        eval_strategy=CONFIG.get("eval_strategy", "steps"),
        save_strategy=CONFIG.get("save_strategy", "steps"),
        eval_steps=CONFIG.get("eval_steps", 200),
        save_steps=CONFIG.get("save_steps", 400),
        num_train_epochs=CONFIG["epochs"],
        fp16=torch.cuda.is_available(),
        learning_rate=3e-4,
        warmup_ratio=0.1,
        save_total_limit=CONFIG.get("save_total_limit", 3),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="tone_cer",
        greater_is_better=False,
        label_names=["labels", "tone_labels"],
        remove_unused_columns=False,
        group_by_length=True,
        length_column_name="input_length",
        report_to="none",
        max_steps=CONFIG["max_steps"],
        gradient_checkpointing=CONFIG.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False}
            if CONFIG.get("gradient_checkpointing", False)
            else None
        ),
        push_to_hub=CONFIG["push_to_hub"],
        hub_strategy=CONFIG.get("hub_strategy", "all_checkpoints"),
        hub_model_id=CONFIG["hub_model_id"],
        hub_token=CONFIG["hub_token"],
        hub_private_repo=True,
    )

    augmenter = None
    if CONFIG.get("augment", True):
        aug_config = AudioAugmentationConfig(
            time_stretch_min_rate=CONFIG["time_stretch_min_rate"],
            time_stretch_max_rate=CONFIG["time_stretch_max_rate"],
            time_stretch_p=CONFIG["time_stretch_p"],
            pitch_shift_min_semitones=CONFIG["pitch_shift_min_semitones"],
            pitch_shift_max_semitones=CONFIG["pitch_shift_max_semitones"],
            pitch_shift_p=CONFIG["pitch_shift_p"],
            gaussian_noise_min_amplitude=CONFIG["gaussian_noise_min_amplitude"],
            gaussian_noise_max_amplitude=CONFIG["gaussian_noise_max_amplitude"],
            gaussian_noise_p=CONFIG["gaussian_noise_p"],
        )
        augmenter = create_audio_augmenter(aug_config)

    data_collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding=True,
        augmenter=augmenter,
        augment=False,
    )

    trainer = CTCTrainer(
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
    print("Training complete. Best combined-validation model saved locally.")

    if CONFIG["push_to_hub"]:
        print("Pushing best combined-validation model to Hugging Face Hub...")
        trainer.push_to_hub(
            commit_message="Best checkpoint on combined validation set (load_best_model_at_end)"
        )
        if CONFIG["hub_model_id"]:
            from huggingface_hub import HfApi

            api = HfApi(token=CONFIG["hub_token"])
            try:
                api.create_tag(
                    repo_id=CONFIG["hub_model_id"],
                    tag="best-combined-valid",
                    tag_message="Best checkpoint on combined validation set (load_best_model_at_end)",
                    repo_type="model",
                    exist_ok=True,
                )
                print(
                    "Tagged best combined-validation model as 'best-combined-valid' on Hub."
                )
            except Exception as e:
                print(f"Warning: Failed to create Hub tag 'best-combined-valid': {e}")


def evaluate_checkpoints_dual(
    folder_model_files, test_orig_ds, test_bible_ds, data_collator, processor
):
    print("\n==========================================")
    print("STARTING DUAL-DOMAIN POST-TRAINING EVALUATION")
    print("==========================================\n")

    # Ensure data collator live augmentation is disabled during post-training evaluation
    if hasattr(data_collator, "augment"):
        setattr(data_collator, "augment", False)

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
    sort_cols = (
        ["tone_cer", "reconciled_cer", "wer", "cer"]
        if "tone_cer" in ranking_orig.columns
        else ["wer", "cer", "median_wer", "median_cer"]
    )
    ranking_orig = ranking_orig.sort_values(
        by=sort_cols,
        ascending=True,
    ).reset_index(drop=True)
    best_ckpt_label = str(ranking_orig.iloc[0]["checkpoint"])
    best_wer = float(ranking_orig.iloc[0]["wer"])
    best_cer = float(ranking_orig.iloc[0]["cer"])
    best_ckpt_path = str(
        ranking_orig[ranking_orig["checkpoint"] == best_ckpt_label].iloc[0]["path"]
    )

    ranking_bible = ranking_bible.sort_values(
        by=["wer", "cer", "median_wer", "median_cer"],
        ascending=True,
    ).reset_index(drop=True)

    print(f"\n==========================================")
    print(f"ORIGINAL TEST SET EVALUATION SUMMARY:")
    if "tone_cer" in ranking_orig.columns:
        best_tcer = float(ranking_orig.iloc[0]["tone_cer"])
        best_rcer = float(ranking_orig.iloc[0]["reconciled_cer"])
        print(
            f"Best Checkpoint: {best_ckpt_label} (TER: {best_tcer:.4f}, Reconciled CER: {best_rcer:.4f}, WER: {best_wer:.4f}, CER: {best_cer:.4f})"
        )
    else:
        print(
            f"Best Checkpoint: {best_ckpt_label} (WER: {best_wer:.4f}, CER: {best_cer:.4f})"
        )
    print(f"==========================================\n")

    return (
        best_ckpt_label,
        best_wer,
        best_cer,
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
    if os.path.abspath(best_ckpt_path) != os.path.abspath(folder_model_files):
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
    else:
        print("Promoted checkpoint is already the root model directory.")

    processor.save_pretrained(folder_model_files)
    save_dual_vocabularies(folder_model_files)
    print(f"Promoted {best_ckpt_label} to final model directory.")


def save_results_summary(
    rows_orig,
    ranking_orig,
    rows_bible,
    ranking_bible,
    best_ckpt_label,
    best_wer,
    best_cer,
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
            f"Best Checkpoint (Orig): {best_ckpt_label} (WER: {best_wer:.4f}, CER: {best_cer:.4f})\n"
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
        best_ckpt_label,
        best_wer,
        best_cer,
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
        best_ckpt_label,
        best_wer,
        best_cer,
        folder_log_files,
    )

    if CONFIG["push_to_hub"] and CONFIG["hub_model_id"]:
        from huggingface_hub import HfApi

        api = HfApi(token=CONFIG["hub_token"])
        print(
            f"Uploading promoted test_orig checkpoint ({best_ckpt_label}) and evaluation logs to Hub..."
        )
        api.upload_folder(
            folder_path=folder_model_files,
            repo_id=CONFIG["hub_model_id"],
            repo_type="model",
            commit_message=f"Promote {best_ckpt_label} (best on orig test: WER {best_wer:.4f}, CER {best_cer:.4f})",
            ignore_patterns=["checkpoint-*/**", "checkpoint-*/*", "checkpoint-*"],
        )
        api.upload_folder(
            folder_path=folder_log_files,
            path_in_repo="eval_logs",
            repo_id=CONFIG["hub_model_id"],
            repo_type="model",
            commit_message=f"Add dual-domain evaluation results and summary for {best_ckpt_label}",
        )
        print("Final model and evaluation logs pushed to Hugging Face Hub.")


if __name__ == "__main__":
    # Prevent OpenMP duplicate initialization crash on macOS conda environments
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Enable CPU fallback for ops missing native MPS implementation
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    main()
