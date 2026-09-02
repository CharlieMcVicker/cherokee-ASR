import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import normalization helpers from our codebase
from transcription.models.asr_model import CherokeeASRModel
from transcription.inference.infer import (
    normalize_text,
    strip_tones,
    TARGET_SAMPLE_RATE,
)


def align_strings(target, hypothesis):
    m, n = len(target), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if target[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # substitution
                    dp[i - 1][j] + 1,  # deletion
                    dp[i][j - 1] + 1,  # insertion
                )

    i, j = m, n
    alignment = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and target[i - 1] == hypothesis[j - 1]:
            alignment.append((target[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append((target[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((target[i - 1], "<del>"))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            alignment.append(("<ins>", hypothesis[j - 1]))
            j -= 1
        else:
            if i > 0 and j > 0:
                alignment.append((target[i - 1], hypothesis[j - 1]))
                i -= 1
                j -= 1
            elif i > 0:
                alignment.append((target[i - 1], "<del>"))
                i -= 1
            else:
                alignment.append(("<ins>", hypothesis[j - 1]))
                j -= 1

    alignment.reverse()
    return alignment


def build_confusion_matrix(alignments, alphabet):
    row_labels = sorted(list(alphabet)) + ["<ins>"]
    col_labels = sorted(list(alphabet)) + ["<del>"]

    matrix = pd.DataFrame(0, index=row_labels, columns=col_labels, dtype=int)

    for align in alignments:
        for t, h in align:
            t_key = t if t in alphabet or t == "<ins>" else "<ins>"
            h_key = h if h in alphabet or h == "<del>" else "<del>"
            if t_key in matrix.index and h_key in matrix.columns:
                matrix.loc[t_key, h_key] += 1

    return matrix


def plot_and_save_matrix(matrix, title, filepath):
    # We normalize each row to show percentages of predicted characters for each true character
    row_sums = matrix.sum(axis=1)
    norm_matrix = matrix.div(row_sums.replace(0, 1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(norm_matrix.values, cmap="Purples")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels([f"'{c}'" if c == " " else c for c in matrix.columns])
    ax.set_yticklabels([f"'{c}'" if c == " " else c for c in matrix.index])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.values[i, j]
            pct = norm_matrix.values[i, j]
            if val > 0:
                text_color = "white" if pct > 50 else "black"
                ax.text(
                    j,
                    i,
                    f"{val}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Character", fontsize=12)
    ax.set_ylabel("True Character (Gold)", fontsize=12)
    fig.tight_layout()
    plt.colorbar(im, ax=ax, label="Percentage (%)")
    plt.savefig(filepath, dpi=300)
    plt.close()


def main():
    asr_model = CherokeeASRModel.from_pretrained_or_best()
    processor = asr_model.processor
    device = asr_model.device
    print(f"Using device: {device}")

    # Determine the alphabet from processor's vocabulary
    vocab = processor.tokenizer.get_vocab()
    alphabet = set()
    for char in vocab.keys():
        if char not in ["[PAD]", "[UNK]", "<s>", "</s>"] and not char.startswith("<"):
            if char == "|":
                alphabet.add(" ")
            else:
                # Decode vocabulary tokens properly to handle multi-byte/unicode chars if any
                decoded = processor.decode([vocab[char]])
                if decoded:
                    alphabet.add(decoded)
                else:
                    alphabet.add(char)

    print(f"Detected alphabet from vocab: {sorted(list(alphabet))}")

    # Load dataset files
    test_csv = "training_data/processed/cim-wav2vec2-test.csv"
    train_csv = "training_data/processed/cim-wav2vec2-train.csv"
    valid_csv = "training_data/processed/cim-wav2vec2-valid.csv"

    df_test = pd.read_csv(test_csv)
    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)

    df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)

    datasets = {"test": df_test, "all": df_all}

    os.makedirs("data/results", exist_ok=True)

    for name, df in datasets.items():
        print(f"\nProcessing {name} dataset ({len(df)} samples)...")

        # Resolve paths
        df["path"] = df["path"].apply(
            lambda p: (
                p
                if os.path.exists(p)
                else os.path.join(
                    "training_data/processed/sentence_audio", os.path.basename(p)
                )
            )
        )
        # Verify files exist
        df = df[df["path"].apply(os.path.exists)]
        print(f"Filtered to {len(df)} existing audio files.")

        df["sentence"] = df["sentence"].apply(normalize_text)
        df.dropna(subset=["path", "sentence"], inplace=True)

        # Run inference
        alignments = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {name}"):
            res = asr_model.transcribe(row["path"], compute_word_confidences=False)
            hyp = res.text
            gold = row["sentence"]

            gold_norm = normalize_text(gold)
            hyp_norm = normalize_text(hyp)

            align = align_strings(gold_norm, hyp_norm)
            alignments.append(align)

        # Build confusion matrix
        matrix = build_confusion_matrix(alignments, alphabet)

        # Save CSV
        csv_path = f"data/results/confusion_matrix_{name}.csv"
        matrix.to_csv(csv_path)
        print(f"Saved confusion matrix to {csv_path}")

        # Plot and save heatmap
        img_path = f"data/results/confusion_matrix_{name}.png"
        plot_and_save_matrix(
            matrix, f"Character Confusion Matrix - {name.capitalize()} Data", img_path
        )
        print(f"Saved heatmap to {img_path}")


if __name__ == "__main__":
    main()
