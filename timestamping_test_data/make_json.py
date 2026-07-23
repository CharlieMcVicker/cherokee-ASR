import json


def cherokee_to_bad_phonetics(text: str) -> str:
    """Translates Cherokee syllabary into literal phonetic transliteration."""
    mapping = {
        "Ꭰ": "a",
        "Ꭱ": "e",
        "Ꭲ": "i",
        "Ꭳ": "o",
        "Ꭴ": "u",
        "Ꭵ": "v",
        "Ꭶ": "ga",
        "Ꭷ": "ka",
        "Ꭸ": "ge",
        "Ꭹ": "gi",
        "Ꭺ": "go",
        "Ꭻ": "gu",
        "Ꭼ": "gv",
        "Ꭽ": "ha",
        "Ꭾ": "he",
        "Ꭿ": "hi",
        "Ꮀ": "ho",
        "Ꮁ": "hu",
        "Ꮂ": "hv",
        "Ꮃ": "la",
        "Ꮄ": "le",
        "Ꮅ": "li",
        "Ꮆ": "lo",
        "Ꮇ": "lu",
        "Ꮈ": "lv",
        "Ꮉ": "ma",
        "Ꮊ": "me",
        "Ꮋ": "mi",
        "Ꮌ": "mo",
        "Ꮍ": "mu",
        "Ꮎ": "na",
        "Ꮏ": "hna",
        "Ꮐ": "nah",
        "Ꮑ": "ne",
        "Ꮒ": "ni",
        "Ꮓ": "no",
        "Ꮔ": "nu",
        "Ꮕ": "nv",
        "Ꮖ": "gwa",
        "Ꮗ": "gwe",
        "Ꮘ": "gwi",
        "Ꮙ": "gwo",
        "Ꮚ": "gwu",
        "Ꮛ": "gwv",
        "Ꮜ": "sa",
        "Ꮝ": "s",
        "Ꮞ": "se",
        "Ꮟ": "si",
        "Ꮠ": "so",
        "Ꮡ": "su",
        "Ꮢ": "sv",
        "Ꮣ": "da",
        "Ꮤ": "ta",
        "Ꮥ": "de",
        "Ꮦ": "te",
        "Ꮧ": "di",
        "Ꮨ": "ti",
        "Ꮩ": "do",
        "Ꮪ": "du",
        "Ꮫ": "dv",
        "Ꮭ": "tla",
        "Ꮬ": "dla",
        "Ꮮ": "tle",
        "Ꮯ": "tli",
        "Ꮰ": "tlo",
        "Ꮱ": "tlu",
        "Ꮲ": "tlv",
        "Ꮳ": "tsa",
        "Ꮴ": "tse",
        "Ꮵ": "tsi",
        "Ꮶ": "tso",
        "Ꮷ": "tsu",
        "Ꮸ": "tsv",
        "Ꮹ": "wa",
        "Ꮺ": "we",
        "Ꮻ": "wi",
        "Ꮼ": "wo",
        "Ꮽ": "wu",
        "Ꮾ": "wv",
        "Ꮿ": "ya",
        "Ᏸ": "ye",
        "Ᏹ": "yi",
        "Ᏺ": "yo",
        "Ᏻ": "yu",
        "Ᏼ": "yv",
    }

    # Translate character by character (preserves spaces, punctuation, unknown symbols)
    return "".join(mapping.get(char, char) for char in text)


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
