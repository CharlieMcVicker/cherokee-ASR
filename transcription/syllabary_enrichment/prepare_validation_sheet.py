"""
Prepare validation data CSV for syllabary enrichment

Validation CSV will have rows
```
split,audio,syllabary,target
```

`split` will mark if the model was trained with this row as train, test, or validation data.
"""

from csv import DictReader, DictWriter

from transcription.utils.tone_normalization import respell_consonants
from transcription.utils.syllabary_map import (
    CHEROKEE_SYLLABARY_MAP,
    syllabary_to_phonetics,
)


def read_sentence_training_data():
    """
    Read splits from training to get list of audios that were in each split
    Get ground truth as well
    """
    base = "training_data/processed/cim-wav2vec2"
    splits = ["train", "test", "valid"]
    sentence_audio_path_prefix = "training_data/processed/sentence_audio/"
    examples = {}
    for split in splits:
        path = f"{base}-{split}.csv"
        with open(path) as f:
            reader = DictReader(f, fieldnames=["path", "sentence"])
            next(reader)
            for row in reader:
                row["split"] = split
                if row["path"].startswith(sentence_audio_path_prefix):
                    audio = row["path"][len(sentence_audio_path_prefix) :]
                    examples[audio] = {
                        "split": split,
                        "target": row["sentence"].replace(":", "").replace(",", ""),
                        "audio_path": row["path"],
                    }
        # read path into CSV
    return examples


def read_sentence_source_data():
    """
    Read original sentence audio sheet to grab syllabary transcripts for each audio file
    """
    with open("training_data/processed/sentence_audio.csv") as f:
        reader = DictReader(
            f,
            fieldnames=[
                "index",
                "sentence_id",
                "audio",
                "syllabary",
                "phonetic",
                "english",
                "source",
                "speaker",
                "notes",
            ],
        )
        next(reader)
        return {row["audio"]: row for row in reader}


def clean_syllabary(syl):
    for drop in ".,?:\"'*‚":
        syl = syl.replace(drop, "")
    return syl


def syllabary_matches_phonetics(syl: str, phonetics: str):
    phon = syllabary_to_phonetics(syl)
    syl_phonetics = respell_consonants(phon)

    # check that consonants match
    vowels = "aeiouv"

    def drop_glottals(s):
        return s.replace("tlh", "lh").replace("h", "").replace("'", "")

    phonetics_no_glottals = drop_glottals(phonetics)
    syl_phonetics_no_glottals = drop_glottals(syl_phonetics)

    cons = phonetics_no_glottals
    cons_syl = syl_phonetics_no_glottals
    for v in vowels:
        cons = cons.replace(v, "")
        cons_syl = cons_syl.replace(v, "")

    if not cons == cons_syl:
        # print("cons don't line up", cons, cons_syl)
        return False

    # check that vowels line up
    vowels_dropped = 0
    syl_vowels = [v for v in syl_phonetics if v in vowels]
    phn_vowels = [v for v in phonetics if v in vowels]
    for i, v in enumerate(phn_vowels):
        if i + vowels_dropped >= len(syl_vowels):
            # print("too many phn vowels", syl_vowels, phn_vowels)
            return False
        elif v == syl_vowels[i + vowels_dropped]:
            continue
        elif (
            i + vowels_dropped + 1 < len(syl_vowels)
            and v == syl_vowels[i + vowels_dropped + 1]
        ):
            vowels_dropped += 1
            continue
        else:
            # vowel doesn't match next or vowel after
            # print("vowels don't line up", syl_vowels, phn_vowels)
            return False

    return len(phn_vowels) + vowels_dropped <= len(syl_vowels)


def add_syllabary_to_examples(examples, sentence_audio_src):
    enriched_examples = []
    for audio_key in examples:
        source = sentence_audio_src.get(audio_key, None)
        if source:
            syllabary = clean_syllabary(source["syllabary"])
            if syllabary_matches_phonetics(syllabary, examples[audio_key]["target"]):
                enriched_examples.append(
                    {**examples[audio_key], "syllabary": syllabary}
                )
    return enriched_examples


def main():
    examples = read_sentence_training_data()
    print(f"Found {len(examples)} CND sentence training examples from manifest")

    sentence_audio_src = read_sentence_source_data()
    print(f"Found {len(sentence_audio_src)} sentences with syllabary from CND export")

    examples_with_syllabary = add_syllabary_to_examples(examples, sentence_audio_src)

    total = len(examples)
    matched = len(examples_with_syllabary)
    pct = (matched / total * 100) if total else 0.0
    print(f"Matched {matched} examples to valid syllabary ({pct:.2f}%)")

    with open("training_data/processed/split_audio_syl_target.csv", "w") as f:
        writer = DictWriter(
            f, fieldnames=["split", "audio_path", "syllabary", "target"]
        )
        writer.writeheader()
        writer.writerows(examples_with_syllabary)


if __name__ == "__main__":
    main()
