---
id: doc-14
title: Model Training and Evaluation Pipeline
type: guide
created_date: '2026-09-24 17:30'
updated_date: '2026-09-24 17:30'
---
# Wav2Vec2 Training, Dataset Preparation, and Evaluation Guide

This guide provides a comprehensive reference for preparing datasets, training acoustic Wav2Vec2 models, and running disaggregated model evaluations in `digohwelisgi.training`.

---

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [Dataset Preparation (`prepare_csv.py`)](#2-dataset-preparation-prepare_csvpy)
   - [Phonetic Text Normalization](#phonetic-text-normalization)
   - [Duration Filtering & Sanity Checks](#duration-filtering--sanity-checks)
   - [Splitting & Exporting](#splitting--exporting)
   - [CLI Reference](#cli-reference)
3. [Model Training (`train.py`)](#3-model-training-trainpy)
   - [Multi-Domain 6-way Split Handling](#multi-domain-6-way-split-handling)
   - [Live 50/50 Dataset Interleaving](#live-5050-dataset-interleaving)
   - [Vocabulary Generation & CTC Tokenizer](#vocabulary-generation--ctc-tokenizer)
   - [Hyperparameters & Optimization](#hyperparameters--optimization)
   - [Resuming Training & Checkpoint Promotion](#resuming-training--checkpoint-promotion)
   - [CLI Reference](#cli-reference-1)
4. [Evaluation Tools](#4-evaluation-tools)
   - [Local Checkpoint Evaluator (`evaluate_checkpoint.py`)](#local-checkpoint-evaluator-evaluate_checkpointpy)
   - [Hugging Face Revisions Evaluator (`evaluate_revisions.py`)](#hugging-face-revisions-evaluator-evaluate_revisionspy)
   - [Shared Evaluation Framework & Masked Metrics](#shared-evaluation-framework--masked-metrics)
5. [End-to-End Workflow & Recipes](#5-end-to-end-workflow--recipes)

---

## 1. Architectural Overview

The Cherokee ASR model architecture is built on **Hugging Face Wav2Vec2** (`facebook/wav2vec2-large-xlsr-53`) fine-tuned with Connectionist Temporal Classification (CTC) loss for the Cherokee language (`cim`).

```
  +--------------------------------------------------------------------------------+
  |                           TRAINING PIPELINE                                    |
  |                                                                                |
  |  [sentence_audio.csv] ---> [prepare_csv.py] ---> [cim-train.csv]               |
  |  [bible_audio.csv]    ---> [prepare_csv.py] ---> [bible-train.csv]              |
  |                                                               |                |
  |                                                               v                |
  |       +-------------------------------------------------------------+          |
  |       | train.py: Live 50/50 Interleaved Stream (interleave_datasets)|          |
  |       +------------------------------+------------------------------+          |
  |                                      |                                         |
  |                                      v                                         |
  |       +-------------------------------------------------------------+          |
  |       | Wav2Vec2ForCTC (facebook/wav2vec2-large-xlsr-53)            |          |
  |       | - Frozen feature encoder (CNN)                              |          |
  |       | - CTC Loss, group_by_length=True, fp16=True                 |          |
  |       +------------------------------+------------------------------+          |
  |                                      |                                         |
  |                                      v                                         |
  |       +-------------------------------------------------------------+          |
  |       | Post-Training Disaggregated Evaluation (evaluate_checkpoints)|          |
  |       | - Original Test Set (baseline & forgetfulness anchor)       |          |
  |       | - Bible Test Set (out-of-domain generalization)             |          |
  |       +------------------------------+------------------------------+          |
  |                                      |                                         |
  |                                      v                                         |
  |       +-------------------------------------------------------------+          |
  |       | Best Checkpoint Promotion -> output_w2v2/wav2vec2-large-xlsr|          |
  |       +-------------------------------------------------------------+          |
  +--------------------------------------------------------------------------------+
```

---

## 2. Dataset Preparation (`prepare_csv.py`)

The script [`digohwelisgi.training.prepare_csv`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/training/prepare_csv.py) transforms raw metadata CSV files and directories of audio files into shuffled, filtered, duration-bounded Train, Validation, and Test CSV splits.

### Phonetic Text Normalization

Audio transcripts undergo thorough phonetic normalization via [`clean_transcription()`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/training/prepare_csv.py#L18-L53) and [`remove_tones_and_double_vowels()`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/utils/tone_normalization.py):
1. **Semicolon Handling:** Word-final semicolons `;` are stripped; word-medial semicolons are converted to `:` (representing vowel length).
2. **Glottal Stop Standardization:** Unifies glottal variations (`ʔ`, `ʼ`, `‚`) to standard single quote `'`.
3. **Tone Stripping:** Strips tone diacritics and rare tone markers; drops samples with corrupt tone marks.
4. **Long Vowel Conversion:** Replaces doubled vowels (`aa`, `ee`, `ii`, `oo`, `uu`, `vv`) with colon notation (`a:`, `e:`, `i:`, `o:`, `u:`, `v:`).
5. **Sanitization:** Drops rows containing invalid non-Cherokee characters `f` and `b`.
6. **Punctuation & Lowercasing:** Removes brackets, quotes, punctuation, and converts to lowercase.

### Duration Filtering & Sanity Checks

- Reads audio headers with `wave` to compute exact duration and file size without decoding the full waveform.
- Filters out audio samples longer than `--max-duration` (default: 15.0s).
- Drops empty transcripts and missing audio files.

### Splitting & Exporting

- Shuffles remaining valid samples using a configurable random seed (`--seed 42`).
- Splits into three target partitions (`--split 80 10 10`).
- Writes CSV files formatted with standard headers: `path,sentence`.

### CLI Reference

```bash
python -m digohwelisgi.training.prepare_csv \
    --csv training_data/processed/sentence_audio.csv \
    --audio-dir training_data/processed/sentence_audio \
    --text-col phonetic \
    --output-prefix training_data/processed/cim-wav2vec2 \
    --max-duration 15.0 \
    --split 80.0 10.0 10.0 \
    --seed 42
```

#### Command-Line Arguments:
- `--csv`: Path to the input CSV containing audio filenames and transcriptions (default: `training_data/processed/sentence_audio.csv`).
- `--audio-dir`: Root directory containing audio `.wav` files.
- `--text-col`: Column header for phonetic text (default: `phonetic`).
- `--output-prefix`: File path prefix for output splits (generates `{prefix}-train.csv`, `{prefix}-valid.csv`, `{prefix}-test.csv`).
- `--max-duration`: Maximum allowed audio duration in seconds (default: `15.0`).
- `--split`: Three floats summing to 100 representing `Train Valid Test` percentages (default: `80.0 10.0 10.0`).
- `--seed`: Integer random seed for reproducible dataset shuffling (default: `42`).

---

## 3. Model Training (`train.py`)

Implemented in [`digohwelisgi.training.train`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/training/train.py).

### Multi-Domain 6-way Split Handling

To prevent catastrophic forgetting across domains while expanding vocabulary, the trainer accepts 6 separate CSV splits:

| Split Name | Default File Path | Purpose |
| :--- | :--- | :--- |
| `train_orig_csv` | `training_data/processed/cim-wav2vec2-train.csv` | Primary conversational/sentence training set |
| `train_bible_csv` | `training_data/processed/bible-wav2vec2-train.csv` | New Testament audio training set |
| `valid_orig_csv` | `training_data/processed/cim-wav2vec2-valid.csv` | Validation set anchor (used for early stopping & best model selection) |
| `valid_bible_csv` | `training_data/processed/bible-wav2vec2-valid.csv` | In-domain Bible validation split |
| `test_orig_csv` | `training_data/processed/cim-wav2vec2-test.csv` | Held-out conversational benchmark test set |
| `test_bible_csv` | `training_data/processed/bible-wav2vec2-test.csv` | Held-out Bible benchmark test set |

### Live 50/50 Dataset Interleaving

Rather than simply concatenating datasets (which risks early dominance by one domain), `train.py` prepares two separate Hugging Face datasets and combines them into a balanced streaming mixture using `datasets.interleave_datasets`:

```python
train_ds = interleave_datasets(
    [ds_train_orig, ds_train_bible],
    probabilities=[0.5, 0.5],
    seed=42,
    stopping_strategy="all_exhausted",
)
```

### Vocabulary Generation & CTC Tokenizer

The script automatically:
1. Extracts all unique characters across all 6 CSV splits.
2. Formats special CTC tokens: space delimiter `|`, unknown token `[UNK]`, and pad token `[PAD]`.
3. Writes `output_w2v2/vocab.json`.
4. Instantiates `Wav2Vec2CTCTokenizer`, `Wav2Vec2FeatureExtractor`, and `Wav2Vec2Processor`.

### Hyperparameters & Optimization

Key `TrainingArguments` configured in [`initialize_model_and_trainer()`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/training/train.py#L476-L545):

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | Peak learning rate with linear warmup |
| `warmup_ratio` | `0.1` | 10% warmup schedule |
| `per_device_train_batch_size` | `8` | Micro-batch size per device |
| `gradient_accumulation_steps` | `2` | Effective training batch size = 16 |
| `per_device_eval_batch_size` | `8` | Evaluation batch size |
| `group_by_length` | `True` | Groups similar audio lengths to minimize padding overhead |
| `length_column_name` | `"input_length"` | Audio frame length column |
| `fp16` | Auto (`torch.cuda.is_available()`) | Mixed precision on CUDA GPUs |
| `attention_dropout` | `0.1` | Regularization |
| `hidden_dropout` | `0.1` | Regularization |
| `layerdrop` | `0.1` | Stochastic depth regularization for transformer layers |
| `mask_time_prob` | `0.05` | SpecAugment-style time masking probability |
| `eval_steps` / `save_steps` | `100` / `400` | Periodic checkpointing and evaluation |
| `save_total_limit` | `20` | Max checkpoints retained |
| `metric_for_best_model` | `"wer"` | Primary early-stopping metric |

### Resuming Training & Checkpoint Promotion

`train.py` supports multiple resumption workflows via [`resolve_resume_checkpoint()`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/training/train.py#L548-L618):
- **Local Checkpoint:** `--resume-from-checkpoint path/to/checkpoint-800` or `--resume-from-checkpoint latest` (automatically finds the highest numbered step).
- **Hugging Face Hub Revision:** `--resume-from-repo username/model_name --resume-from-revision <commit_hash>`. Downloads the remote snapshot and resumes optimizer/scheduler state if available.

#### Automatic Post-Training Promotion:
Upon completion, [`evaluate_checkpoints_dual()`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/training/train.py#L628-L710) evaluates all saved checkpoints against both test sets, selects the top-performing checkpoint, and promotes its weights into `output_w2v2/wav2vec2-large-xlsr/`.

### CLI Reference

```bash
# Basic local training run
python -m digohwelisgi.training.train \
    --epochs 50 \
    --output-dir output_w2v2

# Resume from latest local checkpoint
python -m digohwelisgi.training.train \
    --resume-from-checkpoint latest

# Train and push checkpoints directly to Hugging Face Hub
python -m digohwelisgi.training.train \
    --epochs 50 \
    --push-to-hub \
    --hub-model-id charliemcvicker/cherokee-wav2vec2 \
    --hub-token "$HF_TOKEN"
```

---

## 4. Evaluation Tools

### Local Checkpoint Evaluator (`evaluate_checkpoint.py`)

[`digohwelisgi.training.evaluate_checkpoint`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/training/evaluate_checkpoint.py) evaluates a single checkpoint (local folder or Hugging Face Hub ID) against a specified test CSV split.

```bash
# Evaluate a local checkpoint
python -m digohwelisgi.training.evaluate_checkpoint \
    --checkpoint output_w2v2/wav2vec2-large-xlsr/checkpoint-1200 \
    --test-csv training_data/processed/cim-wav2vec2-test.csv \
    --audio-dir training_data/processed/sentence_audio

# Evaluate a remote Hugging Face model at a specific commit hash
python -m digohwelisgi.training.evaluate_checkpoint \
    --checkpoint charliemcvicker/length-only-20260702-173608-asr-cherokee \
    --revision 3a1b2c4 \
    --test-csv training_data/processed/cim-wav2vec2-test.csv
```

#### Output Artifacts:
- Prints raw WER/CER and vowel-length-masked WER/CER.
- Exports detailed per-sentence hypothesis vs gold comparison to `data/results/test_inference_results.csv`.

---

### Hugging Face Revisions Evaluator (`evaluate_revisions.py`)

[`digohwelisgi.training.evaluate_revisions`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/training/evaluate_revisions.py) performs automated benchmarking across multiple model revisions listed in a TSV or CSV file.

#### Example `data/results/revisions_to_test.tsv`:
```tsv
name	revision
Baseline-Init	f4a8b1c
Epoch-10	8c2d1e0
Epoch-30	9e3f2a1
Final-Promoted	1a2b3c4
```

#### Running the Multi-Revision Sweep:
```bash
python -m digohwelisgi.training.evaluate_revisions \
    --revisions-csv data/results/revisions_to_test.tsv \
    --checkpoint charliemcvicker/length-only-20260702-173608-asr-cherokee \
    --test-csv training_data/processed/cim-wav2vec2-test.csv \
    --output-csv data/results/revision_scores.csv
```

#### Generated `revision_scores.csv` Format:
```csv
name,revision,greedy_wer,greedy_cer,greedy_masked_wer,greedy_masked_cer
Baseline-Init,f4a8b1c,0.4520,0.1820,0.3810,0.1410
Epoch-10,8c2d1e0,0.3120,0.1140,0.2540,0.0890
Epoch-30,9e3f2a1,0.2210,0.0760,0.1810,0.0580
Final-Promoted,1a2b3c4,0.1980,0.0650,0.1620,0.0490
```

---

### Shared Evaluation Framework & Masked Metrics

Evaluation logic is unified across all tools via [`digohwelisgi.utils.evaluation.run_evaluation`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/utils/evaluation.py#L129-L312) and [`CherokeeASRModel`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/models/asr_model.py#L42-L188).

For each evaluated sample, four distinct metric variations are computed to isolate specific acoustic error categories:

1. **Raw (Unmasked) WER/CER:** Direct string comparison against reference phonetics.
2. **Vowel-Length Masked WER/CER:** Evaluated with [`strip_length()`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/inference/infer.py) (collapses long vowels `V:` to short `V`), measuring transcription accuracy independent of vowel duration ambiguities.
3. **Tone Masked WER/CER:** Evaluated with [`strip_tones()`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/inference/infer.py) (removes pitch/tone contours).
4. **Both Masked WER/CER:** Evaluated with [`strip_both()`](file:///Users/julietmcvicker/code/workshop-digohwelisgi/digohwelisgi/inference/infer.py) (removes both tone and vowel length markers).

---

## 5. End-to-End Workflow & Recipes

### Full Training & Evaluation Pipeline Recipe

```bash
# 1. Activate conda environment
conda activate cherokee-asr

# 2. Prepare Conversational / Sentence Dataset Splits
python -m digohwelisgi.training.prepare_csv \
    --csv training_data/processed/sentence_audio.csv \
    --audio-dir training_data/processed/sentence_audio \
    --text-col phonetic \
    --output-prefix training_data/processed/cim-wav2vec2 \
    --max-duration 15.0 \
    --split 80.0 10.0 10.0

# 3. Launch Fine-Tuning with 6-way Multi-Domain Splits
python -m digohwelisgi.training.train \
    --train-orig-csv training_data/processed/cim-wav2vec2-train.csv \
    --train-bible-csv training_data/processed/bible-wav2vec2-train.csv \
    --valid-orig-csv training_data/processed/cim-wav2vec2-valid.csv \
    --valid-bible-csv training_data/processed/bible-wav2vec2-valid.csv \
    --test-orig-csv training_data/processed/cim-wav2vec2-test.csv \
    --test-bible-csv training_data/processed/bible-wav2vec2-test.csv \
    --output-dir output_w2v2 \
    --epochs 50

# 4. Evaluate Promoted Model on Test Split
python -m digohwelisgi.training.evaluate_checkpoint \
    --checkpoint output_w2v2/wav2vec2-large-xlsr \
    --test-csv training_data/processed/cim-wav2vec2-test.csv \
    --audio-dir training_data/processed/sentence_audio
```
