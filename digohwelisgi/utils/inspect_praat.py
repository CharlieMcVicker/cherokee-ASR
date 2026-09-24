#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_praat.py

Parses and inspects Praat TextGrid files:
1. Calculates annotation duration statistics (total annotated seconds).
2. Verifies spelling correctness against the target spelling system:
   - Flags characters that should be "spelled away" (d, k, c).
   - Flags occurrences of 's' without a leading 'h'.
"""

import os
import sys
import argparse
from digohwelisgi.utils.praat_export import parse_textgrid


def verify_text(text):
    """
    Checks the transcription text for spelling system violations.
    Returns a list of error dictionaries with the character, index, and error type.
    """
    errors = []
    forbidden_chars = {"d", "g", "c"}

    for i, char in enumerate(text):
        char_lower = char.lower()

        # Rule 1: Check for forbidden "spelled away" characters (d, k, c)
        if char_lower in forbidden_chars:
            errors.append(
                {
                    "index": i,
                    "char": char,
                    "type": "forbidden_char",
                    "message": f"Forbidden character '{char}' (should be spelled away)",
                }
            )

        # Rule 2: Check for 's' without leading 'h'
        elif char_lower == "s":
            if i > 0 and text[i - 1].lower() not in ["h", "t", " "]:
                errors.append(
                    {
                        "index": i,
                        "char": char,
                        "type": "invalid_s",
                        "message": f"Character '{char}' is present without a leading 'h'",
                    }
                )

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Praat TextGrid files and verify spelling."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default="data/processed/praat/Cora Flute.TextGrid",
        help="Path to the TextGrid file to inspect.",
    )
    parser.add_argument(
        "--tier", default="human", help="Name of the tier to inspect (default: human)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading and parsing {args.file_path}...")
    try:
        tiers = parse_textgrid(args.file_path)
    except Exception as e:
        print(f"Error parsing TextGrid: {e}", file=sys.stderr)
        sys.exit(1)

    if args.tier not in tiers:
        available_tiers = list(tiers.keys())
        print(
            f"Error: Tier '{args.tier}' not found in the TextGrid file.",
            file=sys.stderr,
        )
        print(f"Available tiers: {available_tiers}", file=sys.stderr)
        sys.exit(1)

    intervals = tiers[args.tier]

    # Calculate annotation stats
    total_tier_duration = 0.0
    total_annotated_duration = 0.0
    annotated_count = 0
    non_empty_durations = []

    for xmin, xmax, text in intervals:
        duration = xmax - xmin
        total_tier_duration += duration
        if text.strip():
            total_annotated_duration += duration
            annotated_count += 1
            non_empty_durations.append(duration)

    print(f"\n==================================================")
    print(f"Inspection Results for Tier '{args.tier}':")
    print(f"  Total intervals in tier:      {len(intervals)}")
    print(f"  Total annotated intervals:    {annotated_count}")
    print(f"  Total tier duration:          {total_tier_duration:.2f} seconds")
    print(f"  Total human annotation time:  {total_annotated_duration:.2f} seconds")
    if annotated_count > 0:
        print(
            f"  Avg annotation duration:      {sum(non_empty_durations)/annotated_count:.2f} seconds"
        )
        print(f"  Min annotation duration:      {min(non_empty_durations):.2f} seconds")
        print(f"  Max annotation duration:      {max(non_empty_durations):.2f} seconds")
    print(f"==================================================\n")

    print(f"Verifying spelling rules on '{args.tier}' tier...")

    total_violations = 0
    intervals_with_errors = 0

    for idx, (xmin, xmax, text) in enumerate(intervals):
        if not text.strip():
            continue

        errors = verify_text(text)
        if errors:
            intervals_with_errors += 1
            total_violations += len(errors)

            # Print interval info
            print(
                f"\n[Interval {idx + 1}] {xmin:.3f}s - {xmax:.3f}s (Duration: {xmax-xmin:.2f}s)"
            )
            print(f'  Text: "{text}"')
            for err in errors:
                start_idx = max(0, err["index"] - 10)
                end_idx = min(len(text), err["index"] + 11)
                context = text[start_idx:end_idx]
                pointer = " " * (err["index"] - start_idx) + "^"

                print(f"  Violation: {err['message']} at index {err['index']}")
                print(f"    Context: {context}")
                print(f"             {pointer}")

    print("\n" + "=" * 50)
    print(f"Spelling Verification Summary for '{args.tier}':")
    print(f"  Total intervals checked: {len(intervals)}")
    print(f"  Intervals with errors:   {intervals_with_errors}")
    print(f"  Total spelling errors:   {total_violations}")
    print("=" * 50)


if __name__ == "__main__":
    main()
