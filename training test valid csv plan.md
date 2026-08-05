That makes your pipeline remarkably clean and modular. Pre-splitting the Bible corpus into dedicated CSVs ahead of time gives you full control over chapter boundary enforcement, speaker filtering, and token audit before training starts.

Here is how you can update `trainer_w2v2_local.py` to handle loading all six CSVs (`train_orig`, `train_bible`, `valid_orig`, `valid_bible`, `test_orig`, `test_bible`), interleave the training streams live using Hugging Face `datasets`, and evaluate both domain test sets independently.

---

### Updated Training & Evaluation Implementation

#### 1. Config & Argument Parsing Update

Update your configuration dictionary and CLI arguments to accept all six CSV paths:

```python
# Updated CONFIG entries
CONFIG.update({
    "train_orig_csv": "training_data/processed/cim-wav2vec2-train.csv",
    "train_bible_csv": "training_data/processed/bible-wav2vec2-train.csv",
    "valid_orig_csv": "training_data/processed/cim-wav2vec2-valid.csv",
    "valid_bible_csv": "training_data/processed/bible-wav2vec2-valid.csv",
    "test_orig_csv": "training_data/processed/cim-wav2vec2-test.csv",
    "test_bible_csv": "training_data/processed/bible-wav2vec2-test.csv",
})

```

Add the corresponding CLI arguments in `parse_args()`:

```python
parser.add_argument("--train-orig-csv", type=str, default=CONFIG["train_orig_csv"])
parser.add_argument("--train-bible-csv", type=str, default=CONFIG["train_bible_csv"])
parser.add_argument("--valid-orig-csv", type=str, default=CONFIG["valid_orig_csv"])
parser.add_argument("--valid-bible-csv", type=str, default=CONFIG["valid_bible_csv"])
parser.add_argument("--test-orig-csv", type=str, default=CONFIG["test_orig_csv"])
parser.add_argument("--test-bible-csv", type=str, default=CONFIG["test_bible_csv"])

```

---

#### 2. Load and Normalize All 6 CSVs

Update `load_and_prepare_csvs` to process all six files and ensure character vocabulary generation covers all text sources (so enriched phonetic characters or syllabary symbols aren't dropped):

```python
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

    return dfs, audio_col, text_col

```

---

#### 3. Live 50/50 Dataset Interleaving

In `prepare_datasets`, use `interleave_datasets` on the training streams. Keep the validation and test datasets isolated so you can track metric performance across both domains independently.

```python
from datasets import Features, Value, interleave_datasets, concatenate_datasets

def prepare_datasets(dfs, audio_col, text_col, processor):
    print("Preparing HuggingFace Datasets and interleaving train splits...")
    features = Features(
        {"audio": Audio(sampling_rate=TARGET_SAMPLE_RATE), "sentence": Value("string")}
    )

    def df_to_ds(df):
        data_dict = {"audio": df[audio_col].tolist(), "sentence": df[text_col].tolist()}
        return Dataset.from_dict(data_dict, features=features)

    def prepare_batch(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    # Prepare individual splits
    ds_train_orig = df_to_ds(dfs["train_orig"]).map(prepare_batch, remove_columns=["audio", "sentence"], num_proc=1)
    ds_train_bible = df_to_ds(dfs["train_bible"]).map(prepare_batch, remove_columns=["audio", "sentence"], num_proc=1)

    ds_valid_orig = df_to_ds(dfs["valid_orig"]).map(prepare_batch, remove_columns=["audio", "sentence"], num_proc=1)
    ds_valid_bible = df_to_ds(dfs["valid_bible"]).map(prepare_batch, remove_columns=["audio", "sentence"], num_proc=1)

    ds_test_orig = df_to_ds(dfs["test_orig"]).map(prepare_batch, remove_columns=["audio"], num_proc=1)
    ds_test_bible = df_to_ds(dfs["test_bible"]).map(prepare_batch, remove_columns=["audio"], num_proc=1)

    # 1. Live 50/50 Interleaved Train Stream
    train_ds = interleave_datasets(
        [ds_train_orig, ds_train_bible],
        probabilities=[0.5, 0.5],
        seed=42,
        stopping_strategy="all_exhausted"  # Oversamples smaller split to match total step quota
    )

    MAX_INPUT_LENGTH = TARGET_SAMPLE_RATE * 20
    train_ds = train_ds.filter(
        lambda x: x < MAX_INPUT_LENGTH, input_columns=["input_length"]
    )

    # 2. Validation set strategy: Anchor on original domain for early stopping
    # (Or concatenate both if you want combined step evaluation)
    valid_ds = ds_valid_orig 

    return train_ds, valid_ds, ds_test_orig, ds_test_bible

```

---

#### 4. Post-Training Disaggregated Evaluation

Update the evaluation pass to test checkpoints against both the preserved baseline test set and the new Bible test set:

```python
def evaluate_checkpoints_dual(
    folder_model_files, test_orig_ds, test_bible_ds, data_collator, processor
):
    print("\n==========================================")
    print("STARTING DISAGGREGATED POST-TRAINING EVALUATION")
    print("==========================================\n")
    from transcription.utils.evaluation import yield_local_checkpoints, run_evaluation

    # Pass 1: Original Test Set (Baseline Progress & Forgetfulness Anchor)
    print("Evaluating checkpoints on ORIGINAL Test Set...")
    gen_orig = yield_local_checkpoints(folder_model_files, processor_path=folder_model_files)
    rows_orig, ranking_orig = run_evaluation(
        gen_orig, test_orig_ds, data_collator, batch_size=CONFIG.get("eval_batch_size", 16)
    )

    # Pass 2: Bible Test Set (Unseen Chapters/Books)
    print("Evaluating checkpoints on BIBLE Test Set...")
    gen_bible = yield_local_checkpoints(folder_model_files, processor_path=folder_model_files)
    rows_bible, ranking_bible = run_evaluation(
        gen_bible, test_bible_ds, data_collator, batch_size=CONFIG.get("eval_batch_size", 16)
    )

    return ranking_orig, ranking_bible

```

---

### Pipeline Execution Summary

```
                       ┌──────────────────────┐
                       │  Preprocessing Script│
                       └──────────┬───────────┘
                                  │ (Chapter-by-Chapter Split)
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
┌───────────────────┐                           ┌───────────────────┐
│ Original CSVs     │                           │ Bible CSVs        │
│ (Train/Valid/Test)│                           │ (Train/Valid/Test)│
└─────────┬─────────┘                           └─────────┬─────────┘
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ trainer_w2v2_local.py   │
                     ├─────────────────────────┤
                     │ Live 50/50 Interleave   │
                     │ (train_ds)              │
                     └────────────┬────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌─────────────────────────┐                       ┌─────────────────────────┐
│ Eval: Original-Test     │                       │ Eval: Bible-Test        │
│ (Direct Baseline Comparison)                     │ (Generalization Test)   │
└─────────────────────────┘                       └─────────────────────────┘

```

By decoupling pre-processing into a separate chapter-splitting script and handling interleaving inside `datasets`, your training code stays simple and your metrics remain cleanly isolated.