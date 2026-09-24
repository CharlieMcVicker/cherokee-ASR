#!/usr/bin/env python3
"""
Preprocessing script to split Bible audio transcripts (Mark & Matthew)
into train, validation, and test CSV splits based on chapter boundaries.
"""

import os
import re
import pandas as pd

# Define Chapter allocations (0-indexed or 1-indexed integers matching file names)
# Target ~80% Train, ~10% Valid, ~10% Test per book without leaking chapters.
CHAPTER_SPLITS = {
    "mark.csv": {
        "valid_chapters": [4, 11],  # 41 + 33 = 74 samples (~10.9%)
        "test_chapters": [7, 16],  # 37 + 20 = 57 samples (~8.4%)
        # train_chapters: [1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 15] (547 samples, ~80.7%)
    },
    "matthew.csv": {
        "valid_chapters": [3, 6, 17, 19],  # 17 + 34 + 27 + 30 = 108 samples (~10.1%)
        "test_chapters": [2, 9, 20, 28],  # 23 + 38 + 34 + 20 = 115 samples (~10.7%)
        # train_chapters: remaining 20 chapters (848 samples, ~79.2%)
    },
}

INPUT_DIR = "data/projects/cherokee_new_testament/train_csvs"
OUTPUT_DIR = "data/training/processed"


def extract_chapter(path_str):
    """Extract chapter number from file path, e.g. mark_01_05.wav -> 1"""
    match = re.search(r"_(\d+)_(\d+)\.wav$", str(path_str))
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse chapter from path: {path_str}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_dfs = []
    valid_dfs = []
    test_dfs = []

    for filename, splits in CHAPTER_SPLITS.items():
        file_path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}, skipping.")
            continue

        df = pd.read_csv(file_path)
        df["chapter"] = df["path"].apply(extract_chapter)

        valid_ch = set(splits["valid_chapters"])
        test_ch = set(splits["test_chapters"])

        valid_mask = df["chapter"].isin(valid_ch)
        test_mask = df["chapter"].isin(test_ch)
        train_mask = ~(valid_mask | test_mask)

        df_train = df[train_mask].drop(columns=["chapter"])
        df_valid = df[valid_mask].drop(columns=["chapter"])
        df_test = df[test_mask].drop(columns=["chapter"])

        print(f"Book: {filename}")
        print(
            f"  Total: {len(df)} | Train: {len(df_train)} | Valid: {len(df_valid)} | Test: {len(df_test)}"
        )

        train_dfs.append(df_train)
        valid_dfs.append(df_valid)
        test_dfs.append(df_test)

    full_train = pd.concat(train_dfs, ignore_index=True)
    full_valid = pd.concat(valid_dfs, ignore_index=True)
    full_test = pd.concat(test_dfs, ignore_index=True)

    train_out = os.path.join(OUTPUT_DIR, "bible-wav2vec2-train.csv")
    valid_out = os.path.join(OUTPUT_DIR, "bible-wav2vec2-valid.csv")
    test_out = os.path.join(OUTPUT_DIR, "bible-wav2vec2-test.csv")

    full_train.to_csv(train_out, index=False)
    full_valid.to_csv(valid_out, index=False)
    full_test.to_csv(test_out, index=False)

    print("\n--- Summary ---")
    print(f"Saved {train_out}: {len(full_train)} rows")
    print(f"Saved {valid_out}: {len(full_valid)} rows")
    print(f"Saved {test_out}: {len(full_test)} rows")
    print(
        f"Total Bible Processed: {len(full_train) + len(full_valid) + len(full_test)} rows"
    )


if __name__ == "__main__":
    main()
