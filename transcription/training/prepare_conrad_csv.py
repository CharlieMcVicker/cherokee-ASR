#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Conrad WAVs CSV and append to Wav2Vec2 training dataset.
Converts transcription column to length-only format (normalizing, removing accents, turning colons to doubled vowels).
"""

import os
import csv
from transcription.cherokee.orthography import remove_tones_and_double_vowels
from transcription.training.prepare_csv import clean_transcription


def main():
    conrad_csv = "training_data/processed/conrad-wavs-all.csv"
    train_csv = "training_data/processed/cim-wav2vec2-train.csv"

    if not os.path.exists(conrad_csv):
        print(f"Error: Conrad CSV '{conrad_csv}' not found.")
        return

    if not os.path.exists(train_csv):
        print(f"Error: Target training CSV '{train_csv}' not found.")
        return

    print(f"Reading and processing '{conrad_csv}'...")
    processed_rows = []
    dropped_count = 0
    empty_count = 0
    dropped_f_count = 0
    dropped_b_count = 0

    with open(conrad_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_text = row["sentence"]
            if raw_text is None:
                continue
            if isinstance(raw_text, str):
                # First step: drop wordfinal ';' and replace word medial ';' with ':'
                words = raw_text.split()
                processed_words = []
                for w in words:
                    stripped = w.rstrip(";")
                    processed_words.append(stripped.replace(";", ":"))
                raw_text = " ".join(processed_words)

                raw_text = raw_text.lower()
                # Replace glottal stop characters
                raw_text = raw_text.replace("ɂ", "'").replace("ʔ", "'")
                # Drop specific punctuation/quotation characters
                for char in ["?", ".", "“", "”", "‚", "ʼ"]:
                    raw_text = raw_text.replace(char, "")
            # Normalizing, removing accents, turning colons to doubled vowels
            norm_text, should_drop = remove_tones_and_double_vowels(raw_text)
            if should_drop or norm_text is None:
                dropped_count += 1
                continue

            # Final step: remove all colons (those that weren't placed by vowels, remaining after tone normalizer)
            norm_text = norm_text.replace(":", "")
            # Replace doubled vowels VV with V: for long vowels
            for v in ["a", "e", "i", "o", "u", "v"]:
                norm_text = norm_text.replace(v + v, v + ":")
            cleaned_text = clean_transcription(norm_text)
            if not cleaned_text:
                empty_count += 1
                continue

            # Drop rows containing 'f' or 'b'
            if "f" in cleaned_text:
                dropped_f_count += 1
                continue
            if "b" in cleaned_text:
                dropped_b_count += 1
                continue

            processed_rows.append([row["path"], cleaned_text])

    print(
        f"Processed {len(processed_rows)} rows. (Dropped due to rare marks: {dropped_count}, Empty: {empty_count}, Containing 'f': {dropped_f_count}, Containing 'b': {dropped_b_count})"
    )

    print(f"Appending {len(processed_rows)} rows to '{train_csv}'...")
    with open(train_csv, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(processed_rows)

    print("Done!")


if __name__ == "__main__":
    main()
