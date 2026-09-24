#!/usr/bin/env python3
import csv
import os


def main():
    manifest_path = "data/projects/audiofiles-to-transcribe/segmentation_manifest.csv"
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found.")
        return

    print("Reading existing manifest...")
    rows_to_append = []
    total_existing_rows = 0

    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            total_existing_rows += 1
            orig_fn = row["original_filename"]
            seg_path = row["segmented_filepath"]

            # Check if this row is for an MP3 and is a denoised segment entry
            if orig_fn.lower().endswith(".mp3") and "denoised_segments" in seg_path:
                # Create a matching noisy segment entry
                new_row = row.copy()
                new_row["segmented_filepath"] = seg_path.replace(
                    "denoised_segments", "noisy_segments"
                )
                rows_to_append.append(new_row)

    print(f"Read {total_existing_rows} existing rows.")
    print(
        f"Found {len(rows_to_append)} matching denoised MP3 segments to duplicate for noisy segments."
    )

    if not rows_to_append:
        print("No matching rows found to append. Exiting.")
        return

    print("Appending new rows to manifest...")
    with open(manifest_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows_to_append)

    # Verification: check total rows now
    final_row_count = 0
    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for _ in reader:
            final_row_count += 1

    print(f"Successfully appended {len(rows_to_append)} rows.")
    print(
        f"New total row count in manifest: {final_row_count} (Expected: {total_existing_rows + len(rows_to_append)})"
    )


if __name__ == "__main__":
    main()
