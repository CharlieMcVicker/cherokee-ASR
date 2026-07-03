#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Conrad WAVs CSV and append to Wav2Vec2 training dataset.
Converts transcription column to length-only format (normalizing, removing accents, turning colons to doubled vowels).
"""

import os
import csv
from transcription.utils.tone_normalization import remove_tones_and_double_vowels
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

    with open(conrad_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_text = row['sentence']
            if isinstance(raw_text, str):
                raw_text = raw_text.lower()
                # Replace glottal stop characters
                raw_text = raw_text.replace("ɂ", "'").replace("ʔ", "'")
                # Drop specific punctuation/quotation characters
                for char in ["?", ".", "“", "”", "‚", "ʼ"]:
                    raw_text = raw_text.replace(char, "")
            # Normalizing, removing accents, turning colons to doubled vowels
            norm_text, should_drop = remove_tones_and_double_vowels(raw_text)
            if should_drop:
                dropped_count += 1
                continue
            
            cleaned_text = clean_transcription(norm_text)
            if not cleaned_text:
                empty_count += 1
                continue

            processed_rows.append([row['path'], cleaned_text])

    print(f"Processed {len(processed_rows)} rows. (Dropped due to rare marks: {dropped_count}, Empty: {empty_count})")

    print(f"Appending {len(processed_rows)} rows to '{train_csv}'...")
    with open(train_csv, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(processed_rows)

    print("Done!")


if __name__ == "__main__":
    main()
