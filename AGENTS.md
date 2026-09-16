# Agent Instructions

## Python Environment & Setup
- Always use the `conda` virtual environment: `conda activate cherokee-asr` (Python 3.11).
- System tools required: `ffmpeg`, `cmake`.
- Always execute commands from repository root: `export PYTHONPATH=".:${PYTHONPATH}"`.

## Codebase Architecture
- `transcription/alignment`: Ground-truth alignment pipeline (`align-cherokee` CLI, DTW, Needleman-Wunsch, TextGrid export).
- `transcription/models` & `transcription/inference`: `CherokeeASRModel` encapsulation, confidence extraction, single/batch runners.
- `transcription/syllabary_enrichment`: Phonetic syllabary rule merger (syncopation, aspiration, glottal filtering).
- `transcription/audio`: VAD chunking and segmentation (`segment_long_audio`).
- `transcription/training`: Wav2Vec2 fine-tuning, dataset generation, and HF revision evaluation.
- `syllabary_transcriber`: Standalone desktop application (PyWebView + FastAPI + React/TS).
- `docs/`: Modular technical guides for each subsystem.

## Testing & Quality Assurance
- Run tests: `pytest`
- Run static type checker: `pyright transcription`

## Cherokee Orthographies & Phonetic Conventions

Cherokee text in this codebase exists across three primary orthographic representations (`transcription.utils.orthography.Orthography` enum).

| Orthography Enum | Format & Character Set | Key Usages in Codebase |
| :--- | :--- | :--- |
| **`Orthography.SYLLABARY`** | Native Cherokee Unicode glyphs (`U+13A0`–`U+13FF`, `U+AB70`–`U+ABBF`, e.g., `ᎠᏓᎴᏂᏍᎬ ᏱᏍᏛ ᎧᏃᎮᏛ`). | Syllabary transcripts, UI canvas in `Syllabary Transcriber`, immutable structural anchor in syllabary enrichment. |
| **`Orthography.DG`** | Base $d/g$ Latin transliteration (`d, t, g, k, dl, tl, hl, j, ch, qu, gw, hn, hw, hy`). | Raw Bible chapter JSON imports (`"phonetic"` key with hyphen-separated syllables), 1975 Durbin Feeling dictionary metadata, Conrad transcripts. |
| **`Orthography.TTH`** | Canonical $t/th$ acoustic phonetics (`t, th, k, kh, tl, tlh, lh, ts, tsh, nh, wh, yh, s, hs, a, e, i, o, u, v, '`). | **Acoustic model vocabulary (`CherokeeASRModel`)**, training targets (`cim-wav2vec2-*.csv`), CTC logits, ASR emissions, and alignment targets. |

### Strict T/TH Consonant Inventory Rules
In canonical **`T/TH`**, tones and vowel lengths are decoupled/stripped, and the consonant inventory strictly enforces:
- **NO `d` and NO `g`**: Voiced stops are **`t`** and **`k`**; aspirated/voiceless stops are **`th`** and **`kh`**.
- **NO `ch` and NO `j`**: Voiced affricate is **`ts`**; voiceless/aspirated affricate is **`tsh`**.
- **Lateral Consonants**:
  - `dl` (voiced lateral affricate) $\rightarrow$ **`tl`**
  - `tl` (voiceless/aspirated lateral affricate) $\rightarrow$ **`tlh`**
  - `hl` (voiceless lateral fricative) $\rightarrow$ **`lh`**
- **Aspirated Glides & Nasals**: `hn` $\rightarrow$ **`nh`**, `hw` $\rightarrow$ **`wh`**, `hy` $\rightarrow$ **`yh`**.
- **Pre-aspiration**: Respell preconsonantal/postvocalic sibilants with **`hs`** (e.g. `sgw` $\rightarrow$ `hskw`).
- **NO `c`, `q`, `x`, `z`**: Labio-velars use `kw` / `kwh`.

### Pipeline Ingestion & Emission Mapping
- **Bible Ingestion (`transcription.alignment.ingestion:load_bible_chunks`)**:
  - Bible on disk contains `SYLLABARY` (`"cherokee"`) and hyphenated `DG` (`"phonetic"`: e.g. `A-da-le-ni-s-gv yi-s-dv ka-no-he-dv, Tsi-sa Ga-lo-ne-dv`).
  - Ingestion strips hyphens and converts `DG -> TTH` (`convert_orthography(source=DG, target=TTH)`), yielding `adalenisgv yihstv khanohetv, tsisa kalonetv`.
- **Acoustic Emissions (`CherokeeASRModel`)**:
  - Emits tokens strictly in `TTH` (`athaleniskv`, `hahswanko`, `atil`).
- **Syllabary Enrichment (`transcription.syllabary_enrichment`)**:
  - Anchor: `SYLLABARY` (`ᎠᏓᎴᏂᏍᎬ`).
  - Acoustic Observation: `TTH` emissions (`athaleniskv`).
  - Output: Reconciled `TTH` phonetic representation (`athaleniskv`).
- **Desktop Transcriber (`syllabary_transcriber`)**:
  - Backend transcribes audio to `TTH` $\rightarrow$ frontend renders native `SYLLABARY` glyphs.

## Documentation Maintenance & Freshness
- **Keep Documentation Synchronized**: When adding new modules, refactoring subsystem architecture, changing CLI tools, or modifying testing procedures, immediately update this `AGENTS.md` and the corresponding guide in `docs/`.
- **Prune Obsolete Instructions**: Actively remove superseded workflows, deprecated flags, or outdated setup instructions to prevent agent confusion.
