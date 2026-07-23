import csv
import json

from jiwer import wer as jiwer_wer, cer as jiwer_cer
from transcription.utils.tone_normalization import respell_consonants
import re

# file_path,filename,greedy_transcription,greedy_confidence,word_confidences
with open("data/results/mark_01.csv") as f:
    r = csv.DictReader(
        f,
        fieldnames=[
            "file_path",
            "filename",
            "greedy_transcription",
            "greedy_confidence",
            "word_confidences",
        ],
    )
    transcriptions = list(r)

transcriptions = transcriptions[1:]  # skip header
transcriptions = sorted(transcriptions, key=lambda x: x["filename"])
transcriptions = transcriptions[3:]  # skip first two files that are not in the metadata

cnt_metadata_raw = json.load(open("mark_01_metadata.json"))
cnt_metadata = list(cnt_metadata_raw.values())

aligned = []

index = 0
line_word_idx = 0
for d in transcriptions:
    line = cnt_metadata[index]["phonetic"].lower().replace("-", "")
    line = line.replace("qu", "gw")
    line = respell_consonants(line)
    chars_to_drop = [
        "'",
        "’",
        "‘",
        "“",
        "”",
        ".",
        ",",
        "!",
        "?",
        ";",
        ":",
        "[",
        "]",
        "{",
        "}",
    ]
    for c in chars_to_drop:
        line = line.replace(c, "")

    # put glottals between adjacent vowels
    # line = re.sub(r"([aeiouv])([aeiouv])", "\\1'\\2", line).strip()
    line_words = line.split(" ")
    min_cer = 1.0
    min_idx = line_word_idx
    for i in range(line_word_idx, len(line_words) + 1):
        if i == line_word_idx:
            continue
        trunc_line = " ".join(line_words[line_word_idx:i])
        if not trunc_line:
            continue
        cer: float = jiwer_cer(trunc_line, d["greedy_transcription"])  # type: ignore
        if cer < min_cer:
            min_cer = cer
            min_idx = i
        print(cer, trunc_line)

    aligned_words = " ".join(line_words[line_word_idx:min_idx])

    aligned.append(
        {
            "transcription": d["greedy_transcription"],
            "phonetic": aligned_words,
        }
    )

    print(d["greedy_transcription"])
    print(aligned_words)

    line_word_idx = min_idx
    if line_word_idx >= len(line_words):
        index += 1
        line_word_idx = 0

json.dump(aligned, open("data/results/mark_01_aligned.json", "w"), indent=2)
