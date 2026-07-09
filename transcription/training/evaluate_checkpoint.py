import os
import argparse
import pandas as pd
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import Dataset, Audio, Features, Value
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from tqdm import tqdm

from transcription.inference.infer import (
    greedy_inference,
    normalize_text,
    strip_tones,
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
        description="Evaluate a Wav2Vec2 checkpoint on a test dataset."
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
        "--checkpoint",
        type=str,
        default=None,
        help="Path or HF repo ID to the model checkpoint.",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default=None,
        help="Path or HF repo ID to the processor (defaults to checkpoint).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face Hub authentication token.",
    )
    parser.add_argument(
        "--revision", type=str, default=None, help="Specific HF commit hash/branch/tag."
    )
    args = parser.parse_args()

    token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    checkpoint_path = args.checkpoint
    revision = args.revision

    if not checkpoint_path:
        best_model_path = "best_model.json"
        if os.path.exists(best_model_path):
            import json

            with open(best_model_path, "r") as f:
                best_model = json.load(f)
            checkpoint_path = best_model["repo"]
            if not revision:
                revision = best_model.get("revision")
            print(
                f"Loaded best model settings from {best_model_path}: {checkpoint_path} @ {revision}"
            )
        else:
            checkpoint_path = "remote_output_w2v2/checkpoint-800"

    processor_path = args.processor or checkpoint_path

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

    from transcription.utils.evaluation import yield_single_checkpoint, run_evaluation

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

    processor = Wav2Vec2Processor.from_pretrained(
        processor_path, token=token, revision=revision
    )

    def prepare_batch(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        return batch

    test_ds_prepared = test_ds.map(
        prepare_batch,
        remove_columns=[c for c in test_ds.column_names if c != "sentence"],
        num_proc=1,
    )

    # Run Inference using unified helper
    generator = yield_single_checkpoint(
        checkpoint_path, processor_path=processor_path, revision=revision, token=token
    )
    rows_by_ckpt, ranking_df = run_evaluation(generator, test_ds_prepared)

    if not ranking_df.empty:
        # Convert rows_by_ckpt["checkpoint"] into the expected results_df format
        rows = rows_by_ckpt["checkpoint"]
        results_df = pd.DataFrame(rows).rename(
            columns={"index": "idx", "hyp_greedy": "greedy"}
        )

        row = ranking_df.iloc[0]
        overall_wer_greedy = row["agg_wer_greedy"]
        overall_cer_greedy = row["agg_cer_greedy"]
        overall_wer_greedy_masked = row["agg_wer_greedy_masked"]
        overall_cer_greedy_masked = row["agg_cer_greedy_masked"]

        print("\n" + "=" * 50)
        print("OVERALL RAW METRICS ON TEST SET (WITH TONES & ORIGINAL TRANSC):")
        print(
            f"Greedy: WER = {overall_wer_greedy:.4f} | CER = {overall_cer_greedy:.4f}"
        )
        print("=" * 50 + "\n")

        print("=" * 50)
        print("OVERALL METRICS ON TEST SET WITH VOWEL LENGTHS MASKED (COLLAPSED):")
        print(
            f"Greedy (Vowel-Length Masked): WER = {overall_wer_greedy_masked:.4f} | CER = {overall_cer_greedy_masked:.4f}"
        )
        print("=" * 50 + "\n")

        # Save output to a file
        os.makedirs("data/results", exist_ok=True)
        results_df.to_csv("data/results/test_inference_results.csv", index=False)
        print("Saved test results to data/results/test_inference_results.csv")

    # Display first few comparisons
    print("\nSample Comparisons:")
    for i in range(min(15, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n[{i}] Gold:   {row['gold']}")
        print(f"    Greedy: {row['greedy']} (WER: {row['wer_greedy']:.2f})")


if __name__ == "__main__":
    main()
