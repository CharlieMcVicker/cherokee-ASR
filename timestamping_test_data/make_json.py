import json
from transcription.utils.syllabary_map import cherokee_to_bad_phonetics


def syl_to_bad_phonetic(s):
    for drop in ".!?:,;'\"“”":
        s = s.replace(drop, "")
    return cherokee_to_bad_phonetics(s)


def main():
    json_data = []
    with open("timestamping_test_data/fishing_story.txt") as src:
        chunk = ""
        for line in src:
            line = line.strip()
            if line == "":
                json_data.append(
                    {
                        "line_id": str(len(json_data) + 1),
                        "cherokee_syllabary": chunk.strip(),
                        "raw_phonetic": syl_to_bad_phonetic(chunk.strip()),
                    }
                )
                chunk = ""
            else:
                chunk += " " + line
        json.dump(
            json_data, open("timestamping_test_data/fishing_story.json", "w"), indent=2
        )


if __name__ == "__main__":
    main()
