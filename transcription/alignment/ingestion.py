"""
Ingestion utilities for loading text chunks from Bible metadata and generic JSON sources.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from transcription.cherokee.arpabet.types import SyntheticTargetProjectorProtocol
from transcription.alignment.models import TextChunk
from transcription.cherokee.orthography import (
    Orthography,
    convert_orthography,
)


def normalize_phonetics_for_alignment(
    text: str,
    source: Orthography = Orthography.DG,
    contextual_preaspiration: bool = True,
) -> str:
    """
    Normalizes phonetic Cherokee text for ASR alignment matching.
    """
    if not text:
        return ""
    return convert_orthography(
        text,
        source=source,
        target=Orthography.TTH,
        contextual_preaspiration=contextual_preaspiration,
    )


def normalize_syllabary_for_alignment(
    text: str,
    source: Orthography = Orthography.SYLLABARY,
    target: Orthography = Orthography.TTH,
    contextual_preaspiration: bool = True,
) -> str:
    """
    Normalizes Cherokee Syllabary (or Latin transliteration) for ASR alignment matching.
    """
    if not text:
        return ""
    return convert_orthography(
        text,
        source=source,
        target=target,
        contextual_preaspiration=contextual_preaspiration,
    )


def load_bible_chunks(
    source: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]],
    normalizer: Callable[[str], str] = normalize_phonetics_for_alignment,
) -> Tuple[List[TextChunk], Dict[str, Dict[str, Any]]]:
    """
    Loads Bible verse metadata into TextChunks and a source lookup dictionary.

    Args:
        source: File path to JSON, or a pre-parsed dictionary / list of verse items.
        normalizer: Function to normalize raw phonetic text. Defaults to normalize_phonetics_for_alignment.

    Returns:
        A tuple of (chunks, source_lookup) where:
            chunks: List[TextChunk] with chunk_id and normalized text.
            source_lookup: Dict[str, Dict[str, Any]] mapping chunk_id to source metadata.
    """
    if isinstance(source, (str, Path)):
        if not os.path.exists(str(source)):
            raise FileNotFoundError(f"Bible metadata file not found: {source}")
        with open(str(source), "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = source

    chunks: List[TextChunk] = []
    source_lookup: Dict[str, Dict[str, Any]] = {}

    if isinstance(data, dict):
        for verse_id, verse_info in data.items():
            verse_id_str = str(verse_id)
            if not isinstance(verse_info, dict):
                verse_info = {"text": str(verse_info)}
            raw_text = verse_info.get(
                "phonetic",
                verse_info.get(
                    "raw_phonetic",
                    verse_info.get(
                        "reference_sentence",
                        verse_info.get(
                            "cherokee",
                            verse_info.get(
                                "syllabary",
                                verse_info.get("text", ""),
                            ),
                        ),
                    ),
                ),
            )
            normalized = normalizer(raw_text)
            chunks.append(TextChunk(chunk_id=verse_id_str, text=normalized))
            source_lookup[verse_id_str] = verse_info
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                item = {"text": str(item)}
            verse_id_str = str(
                item.get(
                    "line_id", item.get("verse_id", item.get("id", f"{idx + 1:06d}"))
                )
            )
            raw_text = item.get(
                "phonetic",
                item.get(
                    "raw_phonetic",
                    item.get(
                        "reference_sentence",
                        item.get(
                            "cherokee",
                            item.get(
                                "syllabary",
                                item.get("text", ""),
                            ),
                        ),
                    ),
                ),
            )
            normalized = normalizer(raw_text)
            chunks.append(TextChunk(chunk_id=verse_id_str, text=normalized))
            source_lookup[verse_id_str] = item
    else:
        raise ValueError(
            "Input data must be a dictionary, a list of dictionaries, or a file path."
        )

    return chunks, source_lookup


def load_generic_chunks(
    source: Union[str, Path, List[Dict[str, Any]], Dict[str, Any]],
    normalizer: Optional[Callable[[str], str]] = None,
) -> Tuple[List[TextChunk], Dict[str, Dict[str, Any]]]:
    """
    Loads generic text chunks into TextChunks and a source lookup dictionary.

    Args:
        source: File path to JSON, or a pre-parsed list/dict of chunk items.
        normalizer: Optional function to normalize text. If None, raw text is kept as-is.

    Returns:
        A tuple of (chunks, source_lookup) where:
            chunks: List[TextChunk] with chunk_id and text.
            source_lookup: Dict[str, Dict[str, Any]] mapping chunk_id to source item.
    """
    if isinstance(source, (str, Path)):
        if not os.path.exists(str(source)):
            raise FileNotFoundError(f"Chunk list file not found: {source}")
        with open(str(source), "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = source

    chunks: List[TextChunk] = []
    source_lookup: Dict[str, Dict[str, Any]] = {}

    if isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                item = {"text": str(item)}
            chunk_id = str(
                item.get(
                    "chunk_id",
                    item.get("line_id", item.get("id", f"chunk_{idx + 1:03d}")),
                )
            )
            raw_text = item.get(
                "raw_text",
                item.get("raw_phonetic", item.get("phonetic", item.get("text", ""))),
            )
            text = normalizer(raw_text) if normalizer is not None else raw_text
            chunks.append(TextChunk(chunk_id=chunk_id, text=text))
            source_lookup[chunk_id] = item
    elif isinstance(data, dict):
        for chunk_id, item in data.items():
            chunk_id_str = str(chunk_id)
            if isinstance(item, dict):
                raw_text = item.get(
                    "raw_text",
                    item.get(
                        "raw_phonetic", item.get("phonetic", item.get("text", ""))
                    ),
                )
                info = item
            else:
                raw_text = str(item)
                info = {"text": raw_text}
            text = normalizer(raw_text) if normalizer is not None else raw_text
            chunks.append(TextChunk(chunk_id=chunk_id_str, text=text))
            source_lookup[chunk_id_str] = info
    else:
        raise ValueError(
            "Input data must be a list of dictionaries, a dictionary, or a file path."
        )

    return chunks, source_lookup


def _build_chunk_normalizer(
    base_normalizer: Callable[[str], str] = normalize_phonetics_for_alignment,
    projector: Optional[SyntheticTargetProjectorProtocol] = None,
    code_switched: bool = False,
) -> Callable[[str], str]:
    from transcription.cherokee.codeswitching import (
        create_groundtruth_for_code_switched_syllabary,
        normalize_code_switched_text,
    )

    if code_switched and projector is not None:
        return lambda t: create_groundtruth_for_code_switched_syllabary(
            t, projector=projector
        ).unified_tth
    if projector is not None:
        return lambda t: normalize_code_switched_text(
            t,
            normalizer=base_normalizer,
            projector=projector,
        )
    return base_normalizer


def prepare_alignment_input(
    bible_metadata: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    chunk_list: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]] = None,
    transcript: Optional[
        Union[str, Path, List[str], List[Dict[str, Any]], Dict[str, Any]]
    ] = None,
    projector: Optional[SyntheticTargetProjectorProtocol] = None,
    code_switched: bool = False,
) -> Tuple[
    List[TextChunk],
    Dict[str, Dict[str, Any]],
    Callable[[str], str],
    Callable[[str], str],
]:
    """
    Ingests alignment input from either Bible verse metadata, generic chunk list sources,
    or plain/code-switched syllabary transcripts, resolving the appropriate representation-aware
    text normalizers for chunks and emissions.

    When code_switched=True or a projector is provided, English words in the transcript
    are projected into synthetic Cherokee TTH phonetics using the projector.

    Args:
        bible_metadata: Path to Bible metadata JSON, or dictionary/list of Bible verse metadata items.
        chunk_list: Path to generic chunk list JSON, or list/dictionary of chunk items.
        transcript: Path or in-memory Cherokee Syllabary / code-switched transcript.
        projector: Optional SyntheticTargetProjectorProtocol instance.
        code_switched: Whether to enable code-switched English projection (defaults to False).

    Returns:
        A tuple of (chunks, source_lookup, chunk_normalizer, emissions_normalizer):
            chunks: List[TextChunk] loaded and normalized for alignment.
            source_lookup: Dict[str, Dict[str, Any]] mapping chunk IDs to raw metadata dictionaries.
            chunk_normalizer: Callable[[str], str] normalizer strategy for ground-truth chunks.
            emissions_normalizer: Callable[[str], str] normalizer strategy for ASR token emissions.

    Raises:
        ValueError: If neither or multiple input sources are provided.
    """
    sources_count = sum(x is not None for x in (bible_metadata, chunk_list, transcript))
    if sources_count == 0:
        raise ValueError("Either bible_metadata or chunk_list must be provided.")
    if sources_count > 1:
        if bible_metadata is not None and chunk_list is not None:
            raise ValueError("Cannot provide both bible_metadata and chunk_list.")
        raise ValueError(
            "Cannot provide multiple input sources to prepare_alignment_input."
        )

    active_projector: Optional[SyntheticTargetProjectorProtocol] = projector
    if code_switched and active_projector is None:
        from transcription.cherokee.codeswitching import get_default_projector

        active_projector = get_default_projector()

    if transcript is not None:
        chunks, source_lookup = load_syllabary_transcript(
            transcript,
            normalizer=normalize_syllabary_for_alignment,
            projector=active_projector,
            code_switched=code_switched,
        )
        chunk_norm = _build_chunk_normalizer(
            base_normalizer=normalize_syllabary_for_alignment,
            projector=active_projector,
            code_switched=code_switched,
        )
        return (
            chunks,
            source_lookup,
            chunk_norm,
            normalize_phonetics_for_alignment,
        )

    if bible_metadata is not None:
        chunk_norm = _build_chunk_normalizer(
            base_normalizer=normalize_phonetics_for_alignment,
            projector=active_projector,
        )
        chunks, source_lookup = load_bible_chunks(bible_metadata, normalizer=chunk_norm)
        return (
            chunks,
            source_lookup,
            chunk_norm,
            normalize_phonetics_for_alignment,
        )

    if chunk_list is not None:
        chunk_norm = _build_chunk_normalizer(
            base_normalizer=normalize_phonetics_for_alignment,
            projector=active_projector,
        )
        chunks, source_lookup = load_generic_chunks(chunk_list, normalizer=chunk_norm)
        return (
            chunks,
            source_lookup,
            chunk_norm,
            normalize_phonetics_for_alignment,
        )

    raise ValueError("Either bible_metadata or chunk_list must be provided.")


def load_syllabary_transcript(
    source: Union[
        str,
        Path,
        Sequence[Union[str, Dict[str, Any]]],
        Dict[str, Union[str, Dict[str, Any]]],
    ],
    normalizer: Optional[Callable[[str], str]] = None,
    projector: Optional[SyntheticTargetProjectorProtocol] = None,
    code_switched: bool = False,
    strip_speaker: bool = False,
    contextual_preaspiration: bool = True,
) -> Tuple[List[TextChunk], Dict[str, Dict[str, Any]]]:
    """
    Loads raw Cherokee Syllabary transcripts (lines or chunks) into TextChunks.

    When code_switched=True or a projector is provided, English words in the transcript
    are projected into synthetic Cherokee TTH phonetics using the projector.

    Args:
        source: Raw multiline text string, path to .txt/.json file, list of text lines/chunks,
                or dictionary of chunks.
        normalizer: Function to convert syllabary text into canonical TTH phonetics.
                    Defaults to normalize_syllabary_for_alignment(contextual_preaspiration=...).
        projector: Optional SyntheticTargetProjectorProtocol instance.
        code_switched: Whether to enable code-switched English projection (defaults to False).
        strip_speaker: Whether to strip leading speaker prefixes (e.g. 'Guy Soldier:') from alignment targets.
        contextual_preaspiration: Whether to apply contextual pre-aspiration (suppressing
            leading 'h' before word-initial 's' and affricates) or unconditional 'hs' conversion.

    Returns:
        A tuple of (chunks, source_lookup) where:
            chunks: List[TextChunk] with chunk_id and normalized TTH phonetic text.
            source_lookup: Dict[str, Dict[str, Any]] mapping chunk_id to metadata dictionaries
                           with keys 'syllabary', 'text', 'phonetic', etc.
    """
    from transcription.cherokee.codeswitching import (
        create_groundtruth_for_code_switched_syllabary,
        extract_speaker_prefix,
        get_default_projector,
        normalize_code_switched_text,
    )

    active_projector = projector
    if code_switched and active_projector is None:
        active_projector = get_default_projector()

    resolved_normalizer: Callable[[str], str] = (
        normalizer
        if normalizer is not None
        else (
            lambda s: normalize_syllabary_for_alignment(
                s, contextual_preaspiration=contextual_preaspiration
            )
        )
    )

    if code_switched and active_projector is not None:
        effective_norm: Callable[[str], str] = (
            lambda s: create_groundtruth_for_code_switched_syllabary(
                s,
                projector=active_projector,
                strip_speaker=strip_speaker,
                contextual_preaspiration=contextual_preaspiration,
            ).unified_tth
        )
    elif active_projector is not None:
        effective_norm = lambda s: normalize_code_switched_text(
            s, normalizer=resolved_normalizer, projector=active_projector
        )
    else:
        effective_norm = resolved_normalizer

    def _build_metadata(text_val: str, norm_val: str) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "syllabary": text_val,
            "text": text_val,
            "phonetic": norm_val,
            "raw_line": text_val,
        }
        if code_switched and active_projector is not None:
            cs_res = create_groundtruth_for_code_switched_syllabary(
                text_val,
                projector=active_projector,
                strip_speaker=strip_speaker,
                contextual_preaspiration=contextual_preaspiration,
            )
            meta["code_switched"] = cs_res.to_dict()
            if cs_res.speaker is not None:
                meta["speaker"] = cs_res.speaker
                spoken_display = " ".join(
                    t.source_display for t in cs_res.tokens if t.source_display
                )
                meta["syllabary"] = spoken_display
                meta["text"] = spoken_display
        elif strip_speaker:
            speaker, spoken_text = extract_speaker_prefix(text_val)
            if speaker is not None:
                meta["speaker"] = speaker
                meta["syllabary"] = spoken_text
                meta["text"] = spoken_text
        return meta

    chunks: List[TextChunk] = []
    source_lookup: Dict[str, Dict[str, Any]] = {}

    if isinstance(source, (str, Path)):
        s_str = str(source).strip()
        if os.path.exists(s_str) and os.path.isfile(s_str):
            if s_str.endswith(".json"):
                with open(s_str, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return load_syllabary_transcript(
                    data,
                    normalizer=normalizer,
                    projector=active_projector,
                    code_switched=code_switched,
                    strip_speaker=strip_speaker,
                    contextual_preaspiration=contextual_preaspiration,
                )
            else:
                with open(s_str, "r", encoding="utf-8") as f:
                    content = f.read()
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                for idx, line in enumerate(lines, 1):
                    cid = f"chunk_{idx:03d}"
                    norm = effective_norm(line)
                    chunks.append(TextChunk(chunk_id=cid, text=norm))
                    source_lookup[cid] = _build_metadata(line, norm)
                return chunks, source_lookup
        elif s_str.startswith("{") or s_str.startswith("["):
            try:
                data = json.loads(s_str)
                return load_syllabary_transcript(
                    data,
                    normalizer=normalizer,
                    projector=active_projector,
                    code_switched=code_switched,
                    strip_speaker=strip_speaker,
                    contextual_preaspiration=contextual_preaspiration,
                )
            except Exception:
                pass

        lines = [line.strip() for line in s_str.splitlines() if line.strip()]
        for idx, line in enumerate(lines, 1):
            cid = f"chunk_{idx:03d}"
            norm = effective_norm(line)
            chunks.append(TextChunk(chunk_id=cid, text=norm))
            source_lookup[cid] = _build_metadata(line, norm)
        return chunks, source_lookup

    elif isinstance(source, (list, tuple)):
        for idx, item in enumerate(source, 1):
            if isinstance(item, dict):
                cid = str(
                    item.get(
                        "chunk_id",
                        item.get("line_id", item.get("id", f"chunk_{idx:03d}")),
                    )
                )
                raw_syll = str(
                    item.get(
                        "syllabary",
                        item.get(
                            "cherokee",
                            item.get("text", item.get("raw_text", "")),
                        ),
                    )
                ).strip()
                norm = effective_norm(raw_syll)
                chunks.append(TextChunk(chunk_id=cid, text=norm))
                meta = dict(item)
                meta["syllabary"] = raw_syll
                meta["text"] = raw_syll
                meta["phonetic"] = norm
                if code_switched and active_projector is not None:
                    cs_res = create_groundtruth_for_code_switched_syllabary(
                        raw_syll,
                        projector=active_projector,
                        strip_speaker=strip_speaker,
                        contextual_preaspiration=contextual_preaspiration,
                    )
                    meta["code_switched"] = cs_res.to_dict()
                    if cs_res.speaker is not None:
                        meta["speaker"] = cs_res.speaker
                source_lookup[cid] = meta
            else:
                line_str = str(item).strip()
                cid = f"chunk_{idx:03d}"
                norm = effective_norm(line_str)
                chunks.append(TextChunk(chunk_id=cid, text=norm))
                source_lookup[cid] = _build_metadata(line_str, norm)
        return chunks, source_lookup

    elif isinstance(source, dict):
        for idx, (cid, item) in enumerate(source.items(), 1):
            cid_str = str(cid)
            if isinstance(item, dict):
                raw_syll = str(
                    item.get(
                        "syllabary",
                        item.get(
                            "cherokee",
                            item.get("text", item.get("raw_text", "")),
                        ),
                    )
                ).strip()
                norm = effective_norm(raw_syll)
                chunks.append(TextChunk(chunk_id=cid_str, text=norm))
                meta = dict(item)
                meta["syllabary"] = raw_syll
                meta["text"] = raw_syll
                meta["phonetic"] = norm
                if code_switched and active_projector is not None:
                    cs_res = create_groundtruth_for_code_switched_syllabary(
                        raw_syll,
                        projector=active_projector,
                        strip_speaker=strip_speaker,
                        contextual_preaspiration=contextual_preaspiration,
                    )
                    meta["code_switched"] = cs_res.to_dict()
                    if cs_res.speaker is not None:
                        meta["speaker"] = cs_res.speaker
                source_lookup[cid_str] = meta
            else:
                raw_syll = str(item).strip()
                norm = effective_norm(raw_syll)
                chunks.append(TextChunk(chunk_id=cid_str, text=norm))
                source_lookup[cid_str] = _build_metadata(raw_syll, norm)
        return chunks, source_lookup

    raise ValueError(f"Unsupported transcript source type: {type(source)}")


def load_interview_transcript(
    source: Union[str, Path, List[str]],
    normalizer: Optional[Callable[[str], str]] = None,
    projector: Optional[SyntheticTargetProjectorProtocol] = None,
    code_switched: bool = False,
    contextual_preaspiration: bool = True,
) -> Tuple[List[TextChunk], Dict[str, Dict[str, Any]]]:
    """
    Ingests dialogue and interview transcripts formatted as 'Speaker: Spoken text',
    handling Cherokee Syllabary, English code-switching, speaker labels, and multiline turns.

    Extracts spoken dialogue into normalized TextChunks (stripping speaker prefixes so they
    do not pollute acoustic alignment) while recording speaker identity in source metadata.

    Args:
        source: File path to transcript (.txt), raw multiline string, or list of line strings.
        normalizer: Function to convert syllabary text into canonical TTH phonetics.
                    Defaults to normalize_syllabary_for_alignment(contextual_preaspiration=...).
        projector: Optional SyntheticTargetProjectorProtocol instance.
        code_switched: Whether to enable code-switched English projection (defaults to False).
        contextual_preaspiration: Whether to apply contextual pre-aspiration (suppressing
            leading 'h' before word-initial 's' and affricates) or unconditional 'hs' conversion.

    Returns:
        A tuple of (chunks, source_lookup) where:
            chunks: List[TextChunk] with turn IDs ('turn_001', 'turn_002', ...) and normalized phonetic text.
            source_lookup: Dict[str, Dict[str, Any]] mapping turn ID to metadata dictionary with keys:
                           'speaker', 'syllabary', 'text', 'phonetic', 'raw_line', and 'line_number'.
    """
    from transcription.cherokee.codeswitching import (
        create_groundtruth_for_code_switched_syllabary,
        get_default_projector,
        normalize_code_switched_text,
    )

    active_projector = projector
    if code_switched and active_projector is None:
        active_projector = get_default_projector()

    resolved_normalizer: Callable[[str], str] = (
        normalizer
        if normalizer is not None
        else (
            lambda s: normalize_syllabary_for_alignment(
                s, contextual_preaspiration=contextual_preaspiration
            )
        )
    )

    if code_switched and active_projector is not None:
        effective_norm: Callable[[str], str] = (
            lambda s: create_groundtruth_for_code_switched_syllabary(
                s,
                projector=active_projector,
                contextual_preaspiration=contextual_preaspiration,
            ).unified_tth
        )
    elif active_projector is not None:
        effective_norm = lambda s: normalize_code_switched_text(
            s, normalizer=resolved_normalizer, projector=active_projector
        )
    else:
        effective_norm = resolved_normalizer

    if isinstance(source, (str, Path)):
        s_str = str(source).strip()
        if os.path.exists(s_str) and os.path.isfile(s_str):
            with open(s_str, "r", encoding="utf-8") as f:
                content = f.read()
            raw_lines = content.splitlines()
        else:
            raw_lines = s_str.splitlines()
    elif isinstance(source, list):
        raw_lines = [str(item) for item in source]
    else:
        raise ValueError(
            f"Unsupported interview transcript source type: {type(source)}"
        )

    chunks: List[TextChunk] = []
    source_lookup: Dict[str, Dict[str, Any]] = {}

    current_speaker = "Speaker"
    chunk_idx = 1

    for line_num, line in enumerate(raw_lines, 1):
        stripped = line.strip()
        if not stripped:
            continue

        if ":" in stripped:
            parts = stripped.split(":", 1)
            speaker_candidate = parts[0].strip()
            spoken_text = parts[1].strip()
            if len(speaker_candidate) <= 40 and spoken_text:
                current_speaker = speaker_candidate
                text_to_process = spoken_text
            else:
                text_to_process = stripped
        else:
            text_to_process = stripped

        if not text_to_process:
            continue

        cid = f"turn_{chunk_idx:03d}"
        chunk_idx += 1
        norm_phonetic = effective_norm(text_to_process)

        chunks.append(TextChunk(chunk_id=cid, text=norm_phonetic))
        turn_meta: Dict[str, Any] = {
            "speaker": current_speaker,
            "syllabary": text_to_process,
            "text": text_to_process,
            "phonetic": norm_phonetic,
            "raw_line": stripped,
            "line_number": line_num,
        }
        if code_switched and active_projector is not None:
            turn_meta["code_switched"] = create_groundtruth_for_code_switched_syllabary(
                text_to_process,
                projector=active_projector,
                contextual_preaspiration=contextual_preaspiration,
            ).to_dict()
        source_lookup[cid] = turn_meta

    return chunks, source_lookup


__all__ = [
    "load_bible_chunks",
    "load_generic_chunks",
    "load_interview_transcript",
    "load_syllabary_transcript",
    "prepare_alignment_input",
]
