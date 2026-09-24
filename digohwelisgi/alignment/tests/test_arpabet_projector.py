# -*- coding: utf-8 -*-
"""
Unit and integration tests for the runtime synthetic target projector and
code-switched aligner integration (digohwelisgi.cherokee.arpabet.projector).
"""

import json
from pathlib import Path
import tempfile
import pytest

from digohwelisgi.cherokee.codeswitching.types import (
    ArpabetToken,
    CherokeeToken,
    SyntheticCherokeeTarget,
    SyntheticTargetProjectorProtocol,
)
from digohwelisgi.cherokee.codeswitching import (
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_STATIC_DICTIONARY_PATH,
    SyntheticTargetProjector,
    generate_static_dictionary,
    get_default_projector,
    is_english_word,
    load_default_confusion_matrix,
    normalize_code_switched_text,
    project_english_text,
    project_english_word,
)
from digohwelisgi.alignment.ingestion import (
    load_interview_transcript,
    load_syllabary_transcript,
    prepare_alignment_input,
)


@pytest.fixture(scope="module")
def default_matrix():
    return load_default_confusion_matrix(DEFAULT_CONFUSION_MATRIX_PATH)


@pytest.fixture(scope="module")
def default_projector(default_matrix):
    return get_default_projector()


def test_projector_protocol_compliance(default_projector):
    assert isinstance(default_projector, SyntheticTargetProjectorProtocol)


def test_is_english_word():
    # Cherokee syllabary tokens
    assert not is_english_word("ᏣᎳᎩ")
    assert not is_english_word("ᎯᎠ")
    assert not is_english_word("ᎠᎩᏚᎵ")
    assert not is_english_word("ᏣᎳᎩ,")

    # English words
    assert is_english_word("coffee")
    assert is_english_word("hospital")
    assert is_english_word("Doctor")
    assert is_english_word("coffee!")
    assert is_english_word("code-switched")

    # Pure punctuation / numbers
    assert not is_english_word("")
    assert not is_english_word("1234")
    assert not is_english_word("...")


def test_project_word_dynamic_and_static(default_projector):
    # 'coffee' is in static dictionary
    target_coffee = default_projector.project_word("coffee")
    assert isinstance(target_coffee, SyntheticCherokeeTarget)
    assert target_coffee.source_word == "coffee"
    assert target_coffee.projected_tth == "khasi"
    assert target_coffee.confidence_score > 0.0
    assert target_coffee.syllabary == "ᎧᏏ"

    # 'hospital'
    target_hosp = default_projector.project_word("hospital")
    assert isinstance(target_hosp, SyntheticCherokeeTarget)
    assert "h" in target_hosp.projected_tth
    assert "s" in target_hosp.projected_tth or "hs" in target_hosp.projected_tth
    assert target_hosp.confidence_score > 0.0

    # Punctuation handling
    target_punct = default_projector.project_word('"coffee, "')
    assert target_punct.projected_tth == "khasi"


def test_project_word_unknown_loanword(default_projector):
    # Word not present in any precomputed dictionary falls back to dynamic G2P
    target_unusual = default_projector.project_word("microarchitecture")
    assert isinstance(target_unusual, SyntheticCherokeeTarget)
    assert len(target_unusual.projected_tth) > 0
    assert len(target_unusual.arpabet_tokens) > 0
    assert len(target_unusual.cherokee_tokens) > 0
    # Strictly enforces canonical Cherokee consonant inventory (no d, no g)
    assert "d" not in target_unusual.projected_tth
    assert "g" not in target_unusual.projected_tth


def test_project_arpabet_direct(default_projector, default_matrix):
    arp_tokens = (
        ArpabetToken(phone="K"),
        ArpabetToken(phone="AA"),
        ArpabetToken(phone="F"),
        ArpabetToken(phone="IY"),
    )
    target = default_projector.project_arpabet(
        arp_tokens, matrix=default_matrix, source_word="coffee"
    )
    assert target.projected_tth == "khasi"
    assert len(target.cherokee_tokens) == 4
    assert target.cherokee_tokens[0].phone == "kh"
    assert target.cherokee_tokens[1].phone == "a"
    assert target.cherokee_tokens[2].phone == "s"
    assert target.cherokee_tokens[3].phone == "i"


def test_project_english_text(default_projector):
    res = default_projector.project_english_text("coffee hospital")
    assert " " in res
    parts = res.split()
    assert len(parts) == 2
    assert parts[0] == "khasi"

    # Standalone helper
    res_helper = project_english_text("coffee")
    assert res_helper == "khasi"

    target_word_helper = project_english_word("coffee")
    assert target_word_helper.projected_tth == "khasi"


def test_generate_static_dictionary(default_matrix):
    test_words = ["coffee", "tea", "water", "milk"]
    with tempfile.TemporaryDirectory() as tmpdir:
        dict_path = Path(tmpdir) / "test_loanwords.json"
        generated = generate_static_dictionary(
            words=test_words,
            matrix=default_matrix,
            output_path=dict_path,
            full_metadata=True,
        )
        assert dict_path.exists()
        assert len(generated) == 4
        assert "coffee" in generated
        assert generated["coffee"]["projected_tth"] == "khasi"

        # Load into a new projector and verify O(1) lookup
        custom_proj = SyntheticTargetProjector(
            matrix=default_matrix,
            dictionary=dict_path,
        )
        assert len(custom_proj.dictionary) == 4
        t = custom_proj.project_word("coffee")
        assert t.projected_tth == "khasi"
        assert t.confidence_score > 0.0

        # Compact string dictionary test
        compact_path = Path(tmpdir) / "compact_loanwords.json"
        generate_static_dictionary(
            words=test_words,
            matrix=default_matrix,
            output_path=compact_path,
            full_metadata=False,
        )
        compact_proj = SyntheticTargetProjector(
            matrix=default_matrix,
            dictionary=compact_path,
        )
        t_compact = compact_proj.project_word("tea")
        assert len(t_compact.projected_tth) > 0


def test_normalize_code_switched_text(default_projector):
    # Mixed Syllabary and English
    mixed = "ᎯᎠ coffee ᎠᎩᏚᎵ"
    norm = normalize_code_switched_text(mixed, projector=default_projector)
    tokens = norm.split()
    assert len(tokens) == 3
    assert tokens[0] == "hi'a"
    assert tokens[1] == "khasi"
    assert tokens[2] == "akituli"

    # Punctuation handling in mixed sentence
    mixed_punct = "ᎯᎠ, coffee! ᎠᎩᏚᎵ."
    norm_punct = normalize_code_switched_text(mixed_punct, projector=default_projector)
    assert norm_punct == "hi'a khasi akituli"

    # Pure Cherokee Syllabary
    pure_cherokee = "ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ"
    norm_chr = normalize_code_switched_text(pure_cherokee, projector=default_projector)
    assert norm_chr == "tsalaki kawonihihsti"


def test_load_syllabary_transcript_code_switched(default_projector):
    raw_transcript = "ᎯᎠ coffee ᎠᎩᏚᎵ\nᏣᎳᎩ hospital"
    chunks, source_lookup = load_syllabary_transcript(
        raw_transcript,
        projector=default_projector,
        code_switched=True,
    )
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "chunk_001"
    assert chunks[0].text == "hi'a khasi akituli"
    assert "khasi" in source_lookup["chunk_001"]["phonetic"]
    assert source_lookup["chunk_001"]["syllabary"] == "ᎯᎠ coffee ᎠᎩᏚᎵ"

    assert chunks[1].chunk_id == "chunk_002"
    assert chunks[1].text.startswith("tsalaki ")


def test_load_interview_transcript_code_switched(default_projector):
    raw_interview = "Speaker 1: ᎯᎠ coffee ᎠᎩᏚᎵ\n" "Speaker 2: ᎥᎥ, hospital ᏫᏥᎦ"
    chunks, source_lookup = load_interview_transcript(
        raw_interview,
        projector=default_projector,
        code_switched=True,
    )
    assert len(chunks) == 2
    assert source_lookup["turn_001"]["speaker"] == "Speaker 1"
    assert chunks[0].text == "hi'a khasi akituli"

    assert source_lookup["turn_002"]["speaker"] == "Speaker 2"
    assert "v'v" in chunks[1].text
    assert "hahsthitaw" in chunks[1].text


def test_prepare_alignment_input_code_switched(default_projector):
    # Code-switched transcript input
    transcript_input = ["ᎯᎠ coffee ᎠᎩᏚᎵ", "doctor ᎤᏬᏂᎯᏍᏗ"]
    chunks, source_lookup, chunk_norm, emission_norm = prepare_alignment_input(
        transcript=transcript_input,
        projector=default_projector,
        code_switched=True,
    )
    assert len(chunks) == 2
    assert chunks[0].text == "hi'a khasi akituli"
    assert chunk_norm("coffee") == "khasi"
    assert emission_norm("ha-wa") == "hawa"

    # Code-switched chunk list input
    chunk_list_input = [
        {"id": "c1", "text": "ᎯᎠ coffee ᎠᎩᏚᎵ"},
        {"id": "c2", "text": "hospital"},
    ]
    chunks_c, _, chunk_norm_c, _ = prepare_alignment_input(
        chunk_list=chunk_list_input,
        projector=default_projector,
        code_switched=True,
    )
    assert len(chunks_c) == 2
    assert chunks_c[0].text == "hi'a khasi akituli"
    assert chunks_c[1].text == "hahsthitaw"


def test_project_arpabet_multigram_viterbi():
    """Verify 1,2-gram Viterbi projection selects multi-phone cluster targets."""
    from digohwelisgi.core.codeswitching import AcousticConfusionMatrix
    from digohwelisgi.cherokee.codeswitching.types import ArpabetToken

    # Matrix with 1-gram and 2-gram transitions
    probs = {
        "S": {"s": 0.8, "hs": 0.2},
        "T": {"th": 0.7, "t": 0.3},
        "AY": {"ai": 0.8, "a": 0.2},
        "S T": {"hst": 0.9, "st": 0.1},
    }
    matrix = AcousticConfusionMatrix.create(
        model_id="test_multigram",
        probabilities=probs,
    )
    proj = SyntheticTargetProjector(matrix=matrix, dictionary={})

    # Test diphthong 1->2 mapping
    res_ay = proj.project_arpabet([ArpabetToken("AY")], matrix=matrix)
    assert res_ay.projected_tth == "ai"

    # Test 2-gram cluster S T -> hst
    res_st = proj.project_arpabet([ArpabetToken("S"), ArpabetToken("T")], matrix=matrix)
    assert res_st.projected_tth == "hst"
