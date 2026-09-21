# -*- coding: utf-8 -*-
"""
Unit and integration tests for the code-switching ground truth preparer
(transcription.alignment.arpabet.codeswitched_preparer).

Validates:
1. Script-level token discrimination into Cherokee Syllabary, English, Compound Clitics, and Punctuation.
2. Prevention of double conversion (English words never mutated by Cherokee DG-to-TTH rules).
3. Compound clitic segmentation and unified canonical TTH projection (e.g., JayᎢ, jayᎢ, WellingᏛ, CooksonᎢ).
4. Multi-tier word metadata preservation and lossless serialization.
5. Speaker prefix stripping and preservation.
6. Sample lines and real dialogue from saving-the-voices/gs_mm.txt.
7. Ingestion integration in load_syllabary_transcript and load_interview_transcript with code_switched=True.
"""

from pathlib import Path
import pytest

from transcription.alignment.arpabet import (
    CodeSwitchedLineResult,
    CodeSwitchedToken,
    SyntheticTargetProjector,
    SyntheticTargetProjectorProtocol,
    TokenType,
    classify_token,
    create_groundtruth_for_code_switched_syllabary,
    extract_speaker_prefix,
    get_default_projector,
    prepare_code_switched_token,
    split_compound_clitic,
    strip_boundary_punctuation,
)
from transcription.alignment.ingestion import (
    load_interview_transcript,
    load_syllabary_transcript,
)
from transcription.utils.orthography import Orthography, convert_orthography


@pytest.fixture(scope="module")
def default_projector() -> SyntheticTargetProjectorProtocol:
    return get_default_projector()


# ==============================================================================
# 1. Compound Clitic Segmentation Tests
# ==============================================================================


def test_split_compound_clitic_basic():
    """Validates segmentation of English stems with Cherokee Syllabary clitics."""
    assert split_compound_clitic("JayᎢ") == ("Jay", "Ꭲ")
    assert split_compound_clitic("jayᎢ") == ("jay", "Ꭲ")
    assert split_compound_clitic("WellingᏛ") == ("Welling", "Ꮫ")
    assert split_compound_clitic("CooksonᎢ") == ("Cookson", "Ꭲ")
    assert split_compound_clitic("CooksonᎢᏃ") == ("Cookson", "ᎢᏃ")


def test_split_compound_clitic_with_punctuation():
    """Validates that boundary punctuation does not break compound clitic detection."""
    assert split_compound_clitic("JayᎢ?") == ("Jay", "Ꭲ")
    assert split_compound_clitic("JayᎢ,") == ("Jay", "Ꭲ")
    assert split_compound_clitic("(WellingᏛ)") == ("Welling", "Ꮫ")


def test_split_compound_clitic_negative():
    """Non-compound tokens must return None."""
    assert split_compound_clitic("Jay") is None
    assert split_compound_clitic("Soldier") is None
    assert split_compound_clitic("ᏓᏩᏙ") is None
    assert split_compound_clitic("ᎯᎠ") is None
    assert split_compound_clitic("Ꭲ") is None
    assert split_compound_clitic("") is None
    assert split_compound_clitic("123") is None
    assert split_compound_clitic("...") is None


# ==============================================================================
# 2. Token Classification Tests
# ==============================================================================


def test_classify_token_syllabary():
    """Pure Cherokee Syllabary tokens must be classified as CHEROKEE_SYLLABARY."""
    assert classify_token("ᎯᎠ") == TokenType.CHEROKEE_SYLLABARY
    assert classify_token("ᏓᏩᏙ") == TokenType.CHEROKEE_SYLLABARY
    assert classify_token("ᎣᏏᏍ") == TokenType.CHEROKEE_SYLLABARY
    assert classify_token("ᎭᏛᎩ?") == TokenType.CHEROKEE_SYLLABARY
    assert classify_token("Ꭲ") == TokenType.CHEROKEE_SYLLABARY
    assert classify_token("ᏣᎵ:") == TokenType.CHEROKEE_SYLLABARY


def test_classify_token_english():
    """Pure English tokens must be classified as ENGLISH."""
    assert classify_token("Soldier") == TokenType.ENGLISH
    assert classify_token("Charley") == TokenType.ENGLISH
    assert classify_token("Dry") == TokenType.ENGLISH
    assert classify_token("Creek") == TokenType.ENGLISH
    assert classify_token("yeah") == TokenType.ENGLISH
    assert classify_token("ok") == TokenType.ENGLISH
    assert classify_token("Soldier:") == TokenType.ENGLISH
    assert classify_token("Charley,") == TokenType.ENGLISH
    assert classify_token("McCoy...") == TokenType.ENGLISH
    assert classify_token("(Barber)") == TokenType.ENGLISH


def test_classify_token_compound_clitic():
    """Compound tokens must be classified as COMPOUND_CLITIC."""
    assert classify_token("JayᎢ") == TokenType.COMPOUND_CLITIC
    assert classify_token("jayᎢ") == TokenType.COMPOUND_CLITIC
    assert classify_token("WellingᏛ") == TokenType.COMPOUND_CLITIC
    assert classify_token("CooksonᎢ") == TokenType.COMPOUND_CLITIC
    assert classify_token("JayᎢ?") == TokenType.COMPOUND_CLITIC


def test_classify_token_punctuation():
    """Punctuation, symbols, and empty tokens must be classified as PUNCTUATION."""
    assert classify_token("?") == TokenType.PUNCTUATION
    assert classify_token(",") == TokenType.PUNCTUATION
    assert classify_token("...") == TokenType.PUNCTUATION
    assert classify_token(":") == TokenType.PUNCTUATION
    assert classify_token("-") == TokenType.PUNCTUATION
    assert classify_token("") == TokenType.PUNCTUATION
    assert classify_token("   ") == TokenType.PUNCTUATION


# ==============================================================================
# 3. Double-Conversion Prevention (CRITICAL REQUIREMENT)
# ==============================================================================


def test_prevention_of_double_conversion(default_projector):
    """
    Ensures English tokens are NEVER passed through Cherokee DG-to-TTH consonant mutation.
    In Cherokee DG-to-TTH:
    - 'd' -> 't' (would turn 'Soldier' -> 'soltier', 'Dry' -> 'try')
    - 'j' -> 'ts' (would turn 'Jay' -> 'tsay')
    G2P must receive uncorrupted English words.
    """
    # 1. Test "Soldier": if corrupted to "soltier", G2P would see 'T' instead of 'JH'
    tok_soldier = prepare_code_switched_token("Soldier", projector=default_projector)
    assert tok_soldier.token_type == TokenType.ENGLISH
    assert tok_soldier.english_stem == "Soldier"
    # Canonical TTH must reflect the projected target of 'soldier', not 'soltier'
    expected_soldier = default_projector.project_word("Soldier").projected_tth
    assert tok_soldier.canonical_tth == expected_soldier
    assert "hsow" in tok_soldier.canonical_tth

    # 2. Test "Jay": if corrupted to "tsay", G2P would see 'T S' instead of 'JH'
    tok_jay = prepare_code_switched_token("Jay", projector=default_projector)
    assert tok_jay.token_type == TokenType.ENGLISH
    assert tok_jay.english_stem == "Jay"
    expected_jay = default_projector.project_word("Jay").projected_tth
    assert tok_jay.canonical_tth == expected_jay
    assert tok_jay.canonical_tth == "tse"

    # 3. Test "Dry Creek": ensure 'Dry' retains 'D R AY' phonetics
    tok_dry = prepare_code_switched_token("Dry", projector=default_projector)
    assert tok_dry.english_stem == "Dry"
    expected_dry = default_projector.project_word("Dry").projected_tth
    assert tok_dry.canonical_tth == expected_dry


# ==============================================================================
# 4. Compound Token Segmentation & Projection Tests
# ==============================================================================


def test_prepare_compound_clitic_jay_i(default_projector):
    """JayᎢ -> English 'Jay' (tse) + Cherokee Syllabary 'Ꭲ' (i) -> 'tsei'."""
    tok = prepare_code_switched_token("JayᎢ", projector=default_projector)
    assert tok.token_type == TokenType.COMPOUND_CLITIC
    assert tok.english_stem == "Jay"
    assert tok.syllabary_clitic == "Ꭲ"
    assert tok.canonical_tth == "tsei"
    assert tok.source_display == "JayᎢ"


def test_prepare_compound_clitic_jay_i_lowercase(default_projector):
    """jayᎢ -> English 'jay' (tse) + Cherokee Syllabary 'Ꭲ' (i) -> 'tsei'."""
    tok = prepare_code_switched_token("jayᎢ", projector=default_projector)
    assert tok.token_type == TokenType.COMPOUND_CLITIC
    assert tok.english_stem == "jay"
    assert tok.syllabary_clitic == "Ꭲ"
    assert tok.canonical_tth == "tsei"


def test_prepare_compound_clitic_welling_tv(default_projector):
    """WellingᏛ -> English 'Welling' (wawin) + Cherokee Syllabary 'Ꮫ' (tv) -> 'wawintv'."""
    tok = prepare_code_switched_token("WellingᏛ", projector=default_projector)
    assert tok.token_type == TokenType.COMPOUND_CLITIC
    assert tok.english_stem == "Welling"
    assert tok.syllabary_clitic == "Ꮫ"
    assert tok.canonical_tth == "wawintv"


def test_prepare_compound_clitic_cookson_i(default_projector):
    """CooksonᎢ -> English 'Cookson' (khukhhsan) + Cherokee Syllabary 'Ꭲ' (i) -> 'khukhhsani'."""
    tok = prepare_code_switched_token("CooksonᎢ", projector=default_projector)
    assert tok.token_type == TokenType.COMPOUND_CLITIC
    assert tok.english_stem == "Cookson"
    assert tok.syllabary_clitic == "Ꭲ"
    assert tok.canonical_tth == "khukhhsani"


# ==============================================================================
# 5. Multi-Tier Word Metadata & Serialization Tests
# ==============================================================================


def test_multi_tier_metadata_and_serialization(default_projector):
    """Validates multi-tier metadata properties and round-trip serialization."""
    line = "JayᎢ ᏂᏛᎩᎶᏒ Guy Soldier"
    result = create_groundtruth_for_code_switched_syllabary(
        line, projector=default_projector
    )

    assert isinstance(result, CodeSwitchedLineResult)
    assert len(result.tokens) == 4

    # Multi-tier properties on line result
    assert result.syllabary_tier_tokens == ("JayᎢ", "ᏂᏛᎩᎶᏒ")
    assert result.english_tier_tokens == ("Jay", "Guy", "Soldier")
    assert len(result.reconciled_tier_tokens) == 4
    assert result.unified_tth == " ".join(result.reconciled_tier_tokens)

    # Individual token properties
    tok_compound = result.tokens[0]
    assert tok_compound.syllabary_tier == "JayᎢ"
    assert tok_compound.english_tier == "Jay"
    assert tok_compound.reconciled_tier == "tsei"

    tok_syll = result.tokens[1]
    assert tok_syll.syllabary_tier == "ᏂᏛᎩᎶᏒ"
    assert tok_syll.english_tier is None
    assert tok_syll.reconciled_tier == "nitvkilohsv"

    tok_eng = result.tokens[2]
    assert tok_eng.syllabary_tier is None
    assert tok_eng.english_tier == "Guy"

    # Lossless dictionary round-trip
    d = result.to_dict()
    reconstructed = CodeSwitchedLineResult.from_dict(d)
    assert reconstructed.raw_text == result.raw_text
    assert reconstructed.unified_tth == result.unified_tth
    assert len(reconstructed.tokens) == len(result.tokens)
    for orig_t, rec_t in zip(result.tokens, reconstructed.tokens):
        assert orig_t.raw_token == rec_t.raw_token
        assert orig_t.token_type == rec_t.token_type
        assert orig_t.english_stem == rec_t.english_stem
        assert orig_t.syllabary_clitic == rec_t.syllabary_clitic
        assert orig_t.canonical_tth == rec_t.canonical_tth
        assert orig_t.source_display == rec_t.source_display

    # Lossless JSON round-trip
    json_str = result.to_json()
    from_json_res = CodeSwitchedLineResult.from_json(json_str)
    assert from_json_res.unified_tth == result.unified_tth


# ==============================================================================
# 6. Speaker Prefix Stripping & Preservation Tests
# ==============================================================================


def test_extract_speaker_prefix():
    """Validates speaker prefix extraction from colon-delimited lines."""
    speaker, text = extract_speaker_prefix("Guy Soldier: ᎯᏅ ᎣᏏᏍ ᎭᏛᎩ?")
    assert speaker == "Guy Soldier"
    assert text == "ᎯᏅ ᎣᏏᏍ ᎭᏛᎩ?"

    speaker, text = extract_speaker_prefix("Charley McCoy: Ꮭ ᎤᏟ ᏱᎦ")
    assert speaker == "Charley McCoy"
    assert text == "Ꮭ ᎤᏟ ᏱᎦ"

    speaker, text = extract_speaker_prefix("ᏣᎵ: ᏓᏤᏟᎢᏛ Ꮓ")
    assert speaker == "ᏣᎵ"
    assert text == "ᏓᏤᏟᎢᏛ Ꮓ"

    speaker, text = extract_speaker_prefix("No colon here")
    assert speaker is None
    assert text == "No colon here"


def test_speaker_prefix_stripping_and_preservation(default_projector):
    """
    Validates create_groundtruth_for_code_switched_syllabary with:
    - strip_speaker=True: isolates spoken text from speaker label.
    - strip_speaker=False: preserves speaker label in token sequence.
    """
    raw_line = "Guy Soldier: ᎯᏅ ᎣᏏᏍ ᎭᏛᎩ?"

    # Stripped mode
    stripped_res = create_groundtruth_for_code_switched_syllabary(
        raw_line, projector=default_projector, strip_speaker=True
    )
    assert stripped_res.speaker == "Guy Soldier"
    assert len(stripped_res.tokens) == 3
    assert [t.raw_token for t in stripped_res.tokens] == ["ᎯᏅ", "ᎣᏏᏍ", "ᎭᏛᎩ?"]
    assert "hsowtsa" not in stripped_res.unified_tth  # 'Soldier' omitted from TTH
    assert stripped_res.unified_tth == "hinv ohsihs hatvki"

    # Preserved mode
    preserved_res = create_groundtruth_for_code_switched_syllabary(
        raw_line, projector=default_projector, strip_speaker=False
    )
    assert preserved_res.speaker is None
    assert len(preserved_res.tokens) == 5
    assert preserved_res.tokens[0].raw_token == "Guy"
    assert preserved_res.tokens[1].raw_token == "Soldier:"
    assert "hsowtsa" in preserved_res.unified_tth  # 'Soldier' included in TTH


# ==============================================================================
# 7. Real Sentences from saving-the-voices/gs_mm.txt
# ==============================================================================


def test_gs_mm_line_1(default_projector):
    """Line 1: 'Guy Soldier: ᎯᏅ ᎣᏏᏍ ᎭᏛᎩ?'"""
    res = create_groundtruth_for_code_switched_syllabary(
        "Guy Soldier: ᎯᏅ ᎣᏏᏍ ᎭᏛᎩ?",
        projector=default_projector,
        strip_speaker=True,
    )
    assert res.speaker == "Guy Soldier"
    assert res.unified_tth == "hinv ohsihs hatvki"


def test_gs_mm_line_5_compounds(default_projector):
    """
    Line 5: 'Guy: ᎮᏍᏗᏍ ᏌᎶᎳ ᏱᏍᏕᎵᏍᎨᏍᏗ ᎯᎠ ᎯᏅ Guy Soldier ᏓᏩᏙ JayᎢ ᏂᏛᎩᎶᏒ ᏍᏓᏅᏔᏛ ᎤᎾ jayᎢ ᎨᎲ?'
    Contains pure Syllabary, pure English names, and compound clitics JayᎢ and jayᎢ.
    """
    res = create_groundtruth_for_code_switched_syllabary(
        "Guy: ᎮᏍᏗᏍ ᏌᎶᎳ ᏱᏍᏕᎵᏍᎨᏍᏗ ᎯᎠ ᎯᏅ Guy Soldier ᏓᏩᏙ JayᎢ ᏂᏛᎩᎶᏒ ᏍᏓᏅᏔᏛ ᎤᎾ jayᎢ ᎨᎲ?",
        projector=default_projector,
        strip_speaker=True,
    )
    assert res.speaker == "Guy"

    # Find the compound tokens
    compound_tokens = [
        t for t in res.tokens if t.token_type == TokenType.COMPOUND_CLITIC
    ]
    assert len(compound_tokens) == 2

    tok_cap, tok_low = compound_tokens[0], compound_tokens[1]
    assert tok_cap.english_stem == "Jay"
    assert tok_cap.syllabary_clitic == "Ꭲ"
    assert tok_cap.canonical_tth == "tsei"

    assert tok_low.english_stem == "jay"
    assert tok_low.syllabary_clitic == "Ꭲ"
    assert tok_low.canonical_tth == "tsei"

    # Verify both occurrences of tsei in unified TTH
    assert res.unified_tth.count("tsei") == 2
    # Verify Guy and Soldier were projected without double conversion
    assert "ka" in res.unified_tth
    assert "hsowtsa" in res.unified_tth


def test_gs_mm_line_14_english_names(default_projector):
    """Line 14: 'Charley, Charley McCoy ᏓᏩᏙ'"""
    res = create_groundtruth_for_code_switched_syllabary(
        "Charley, Charley McCoy ᏓᏩᏙ",
        projector=default_projector,
    )
    assert len(res.tokens) == 4
    # First token has comma preserved in raw_token/source_display
    assert res.tokens[0].raw_token == "Charley,"
    assert res.tokens[0].token_type == TokenType.ENGLISH
    assert res.tokens[0].english_stem == "Charley"

    assert res.tokens[1].raw_token == "Charley"
    assert res.tokens[2].raw_token == "McCoy"
    assert res.tokens[3].raw_token == "ᏓᏩᏙ"
    assert res.tokens[3].token_type == TokenType.CHEROKEE_SYLLABARY
    assert res.tokens[3].canonical_tth == "tawato"


def test_gs_mm_line_24_place_names(default_projector):
    """Line 24: 'ᎤᏲᏛ Dry Creek  ᎤᎾ ᏛᏩᏛᏒ ᎠᏯ'"""
    res = create_groundtruth_for_code_switched_syllabary(
        "ᎤᏲᏛ Dry Creek  ᎤᎾ ᏛᏩᏛᏒ ᎠᏯ",
        projector=default_projector,
    )
    tokens_by_type = {t.raw_token: t.token_type for t in res.tokens}
    assert tokens_by_type["Dry"] == TokenType.ENGLISH
    assert tokens_by_type["Creek"] == TokenType.ENGLISH
    assert tokens_by_type["ᎤᏲᏛ"] == TokenType.CHEROKEE_SYLLABARY
    assert tokens_by_type["ᎠᏯ"] == TokenType.CHEROKEE_SYLLABARY


def test_gs_mm_line_40_compound_welling(default_projector):
    """Line 40: 'Mary: WellingᏛ ᏣᏃᏎᎲ ᎠᏩᏕᏅ ᎡᎵᏏ ᎮᏂ ᏧᏪᏅᏒ ᎠᏩᏕᏅ'"""
    res = create_groundtruth_for_code_switched_syllabary(
        "Mary: WellingᏛ ᏣᏃᏎᎲ ᎠᏩᏕᏅ ᎡᎵᏏ ᎮᏂ ᏧᏪᏅᏒ ᎠᏩᏕᏅ",
        projector=default_projector,
        strip_speaker=True,
    )
    assert res.speaker == "Mary"
    welling_tok = res.tokens[0]
    assert welling_tok.token_type == TokenType.COMPOUND_CLITIC
    assert welling_tok.english_stem == "Welling"
    assert welling_tok.syllabary_clitic == "Ꮫ"
    assert welling_tok.canonical_tth == "wawintv"
    assert res.unified_tth.startswith("wawintv")


# ==============================================================================
# 8. Ingestion Pipeline Integration Tests
# ==============================================================================


def test_load_interview_transcript_with_codeswitched_compounds(default_projector):
    """
    Validates load_interview_transcript when code_switched=True properly uses
    create_groundtruth_for_code_switched_syllabary, segments compound clitics,
    and stores code-switched metadata in source_lookup.
    """
    raw_interview = (
        "Guy Soldier: ᎯᎠ Guy Soldier ᏓᏩᏙ JayᎢ ᏂᏛᎩᎶᏒ\n" "Mary: WellingᏛ ᏣᏃᏎᎲ ᎠᏩᏕᏅ"
    )
    chunks, source_lookup = load_interview_transcript(
        raw_interview,
        projector=default_projector,
        code_switched=True,
    )
    assert len(chunks) == 2

    # Turn 1
    t1 = chunks[0]
    assert "tsei" in t1.text
    meta1 = source_lookup["turn_001"]
    assert meta1["speaker"] == "Guy Soldier"
    assert "code_switched" in meta1
    assert any(
        t["token_type"] == TokenType.COMPOUND_CLITIC.value
        for t in meta1["code_switched"]["tokens"]
    )

    # Turn 2
    t2 = chunks[1]
    assert "wawintv" in t2.text
    meta2 = source_lookup["turn_002"]
    assert meta2["speaker"] == "Mary"
    assert "code_switched" in meta2
    assert meta2["code_switched"]["tokens"][0]["english_stem"] == "Welling"


def test_load_syllabary_transcript_with_codeswitched_compounds(default_projector):
    """
    Validates load_syllabary_transcript when code_switched=True properly uses
    create_groundtruth_for_code_switched_syllabary and handles compound tokens.
    """
    raw_lines = [
        "JayᎢ ᏂᏛᎩᎶᏒ",
        "CooksonᎢ ᏗᏓᎾᏅ",
    ]
    chunks, source_lookup = load_syllabary_transcript(
        raw_lines,
        projector=default_projector,
        code_switched=True,
    )
    assert len(chunks) == 2
    assert chunks[0].text.startswith("tsei")
    assert chunks[1].text.startswith("khukhhsani")
    assert "code_switched" in source_lookup["chunk_001"]
    assert "code_switched" in source_lookup["chunk_002"]


def test_codeswitched_token_phonotactic_masks(default_projector):
    """
    Validates that English tokens receive strictly zero syncope and intrusion masks,
    Syllabary tokens receive Cherokee phonotactics, and compound clitics are segmented.
    """
    # 1. Pure English word: Soldier
    soldier_tok = prepare_code_switched_token("Soldier", projector=default_projector)
    assert soldier_tok.token_type == TokenType.ENGLISH
    assert len(soldier_tok.syncope_mask) == len(soldier_tok.canonical_tth)
    assert len(soldier_tok.intrusion_mask) == len(soldier_tok.canonical_tth)
    assert not any(soldier_tok.syncope_mask)
    assert not any(soldier_tok.intrusion_mask)

    # 2. Pure Cherokee Syllabary word: ᎠᏓᎴᏂᏍᎬ (canonical TTH: adalenisgv)
    syll_tok = prepare_code_switched_token("ᎠᏓᎴᏂᏍᎬ", projector=default_projector)
    assert syll_tok.token_type == TokenType.CHEROKEE_SYLLABARY
    assert len(syll_tok.syncope_mask) == len(syll_tok.canonical_tth)
    assert len(syll_tok.intrusion_mask) == len(syll_tok.canonical_tth)
    # Cherokee word has weak vowels (e.g., 'a' at idx 0 or syncopatable sites)
    assert any(syll_tok.syncope_mask)

    # 3. Compound Clitic: JayᎢ (stem Jay -> tse, clitic Ꭲ -> i)
    jay_tok = prepare_code_switched_token("JayᎢ", projector=default_projector)
    assert jay_tok.token_type == TokenType.COMPOUND_CLITIC
    assert jay_tok.canonical_tth == "tsei"
    assert len(jay_tok.syncope_mask) == 4
    # Stem 'tse' (first 3 chars) must be strictly all False for syncope and intrusion
    assert not any(jay_tok.syncope_mask[:3])
    assert not any(jay_tok.intrusion_mask[:3])

    # 4. Serialization roundtrip preserves masks
    tok_dict = jay_tok.to_dict()
    restored_tok = CodeSwitchedToken.from_dict(tok_dict)
    assert restored_tok.syncope_mask == jay_tok.syncope_mask
    assert restored_tok.intrusion_mask == jay_tok.intrusion_mask


def test_codeswitched_preparer_contextual_preaspiration(default_projector):
    """
    Validates that contextual_preaspiration flag properly controls sibilant pre-aspiration
    for Cherokee tokens and clitics within code-switched text.
    """
    line = "Jay ᏍᎩ ᏌᏊ ᎠᏍᎦᏯ"

    res_contextual = create_groundtruth_for_code_switched_syllabary(
        line,
        projector=default_projector,
        contextual_preaspiration=True,
    )
    # Word-initial 's' in ᏍᎩ and ᏌᏊ remains bare 's'; medial in ᎠᏍᎦᏯ receives 'hs'
    assert res_contextual.unified_tth == "tse ski sakwu ahskaya"

    res_unconditional = create_groundtruth_for_code_switched_syllabary(
        line,
        projector=default_projector,
        contextual_preaspiration=False,
    )
    assert res_unconditional.unified_tth == "tse hski hsakwu ahskaya"
