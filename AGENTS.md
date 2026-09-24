# Agent Instructions

## Python Environment & Setup
- Always use the `conda` virtual environment: `conda activate cherokee-asr` (Python 3.11).
- System tools required: `ffmpeg`, `cmake`.
- Always execute commands from repository root: `export PYTHONPATH=".:${PYTHONPATH}"`.

## 4-Tier Codebase Architecture

The codebase is modularized into four distinct architectural layers:

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 4: Applications & Entrypoints (digohwelisgi.apps)     │
│   • CLI (align-cherokee)                                    │
│   • Desktop Transcriber (syllabary_transcriber)             │
├─────────────────────────────────────────────────────────────┤
│ Tier 3: Domain Pipelines (digohwelisgi.pipelines)          │
│   • scripture: Continuous chapter alignment & verse slicing │
│   • dialogue: Code-switched interview alignment             │
│   • enrichment: Phonetic syllabary enrichment & alignment   │
├─────────────────────────────────────────────────────────────┤
│ Tier 2: Cherokee Domain (digohwelisgi.cherokee)            │
│   • orthography: Syllabary, DG, TTH conversion & tables     │
│   • phonotactics: Intrusions, syncope, surface constraints  │
│   • distance: Phonological confusion cost metrics           │
│   • codeswitching: Synthetic loanword target projection     │
│   • enrichment: Syllable reconciliation engine              │
│   • models: CherokeeASRModel factory & weights              │
├─────────────────────────────────────────────────────────────┤
│ Tier 1: Core Engine (digohwelisgi.core)                    │
│   • audio: Segmenting & Silero VAD soft-masking             │
│   • models: ModelOutput currency, inference & ASRModel      │
│   • alignment: DP (DTW, Needleman-Wunsch) & CTC trellis     │
│   • exporters: Praat TextGrid & Manifest serialization      │
└─────────────────────────────────────────────────────────────┘
```

### Module Paths
- **`digohwelisgi.core`**: Language-agnostic foundational engine.
  - `digohwelisgi.core.audio`: `AudioChunk`, `segment_long_audio`, `SileroVADDetector`, `mask_non_speech_logits`.
  - `digohwelisgi.core.models`: `ModelOutput` (universal currency with `.npz` caching), standalone inference (`infer_emissions`, `infer_emissions_batch`), and `ASRModel`.
  - `digohwelisgi.core.alignment`: Pure domain models (`TextChunk`, `TokenEmission`, `WordInterval`, `AlignedChunk`, `AlignmentOutput`), DP aligners (`NeedlemanWunschWordAligner`, `SlidingWindowDTWAligner`), CTC segmentation aligner (`CTCSegmentationAligner`), and generic distance protocols (`DistanceMetric`, `DefaultCERDistanceMetric`).
  - `digohwelisgi.core.exporters`: `TextGridBuilder`, `export_textgrid`, `export_manifest`, `export_debug_json`.
- **`digohwelisgi.cherokee`**: Cherokee phonetic, phonotactic, and linguistic domain logic.
  - `digohwelisgi.cherokee.orthography`: `Orthography` enum, `convert_orthography`, syllabary lookup dictionaries, and tone stripping.
  - `digohwelisgi.cherokee.phonotactics`: Surface phonotactic rules, transition masks, syncope/intrusion masks, and `prepare_cherokee_text`.
  - `digohwelisgi.cherokee.distance`: `PhonologicalConfusionCostMetric`, `ConfusionMatrixCostMetric`.
  - `digohwelisgi.cherokee.codeswitching`: `SyntheticTargetProjector`, `CodeSwitchedPreparer`, compound clitic segmentation.
  - `digohwelisgi.cherokee.enrichment`: `SyllableAlignmentEngine`, `reconcile_phonetics`, `reconcile_alignment_words`.
  - `digohwelisgi.cherokee.models`: `CherokeeASRModel` loader.
- **`digohwelisgi.pipelines`**: End-to-end domain orchestration pipelines.
  - `digohwelisgi.pipelines.scripture`: `ScripturePipeline`, `align_chapter`.
  - `digohwelisgi.pipelines.dialogue`: `DialogueAlignmentPipeline`, `align_dialogue`.
  - `digohwelisgi.pipelines.enrichment`: `EnrichmentPipeline`.
- **`digohwelisgi.apps`**: High-level application drivers and CLI interfaces.
  - `digohwelisgi.apps.cli`: `align-cherokee` CLI entrypoint.
  - `syllabary_transcriber`: Standalone desktop application (PyWebView + FastAPI + React/TS).
- **`digohwelisgi.evaluation`** & **`digohwelisgi.training`**: Wav2Vec2 fine-tuning, dataset split preparation, manifold analysis, and checkpoint evaluation.

## Testing & Quality Assurance
- Run tests: `pytest`
- Run static type checker: `pyright digohwelisgi`

## Cherokee Orthographies & Phonetic Conventions

Cherokee text in this codebase exists across three primary orthographic representations (`digohwelisgi.cherokee.orthography.Orthography` enum).

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
- **Bible Ingestion (`digohwelisgi.pipelines.scripture` / `load_bible_chunks`)**:
  - Bible on disk contains `SYLLABARY` (`"cherokee"`) and hyphenated `DG` (`"phonetic"`: e.g. `A-da-le-ni-s-gv yi-s-dv ka-no-he-dv, Tsi-sa Ga-lo-ne-dv`).
  - Ingestion strips hyphens and converts `DG -> TTH` (`convert_orthography(source=DG, target=TTH)`), yielding `adalenisgv yihstv khanohetv, tsisa kalonetv`.
- **Acoustic Emissions (`digohwelisgi.core.models` / `CherokeeASRModel`)**:
  - Emits tokens strictly in `TTH` (`athaleniskv`, `hahswanko`, `atil`).
- **Code-Switched Ingestion & Loanword Projection (`digohwelisgi.cherokee.codeswitching:SyntheticTargetProjector`)**:
  - Input: Mixed Cherokee Syllabary and English code-switched transcripts (e.g., `ᎯᎠ coffee ᎠᎩᏚᎵ`).
  - Projector: O(1) static memoized dictionary lookup (`data/arpabet_alignment/dictionaries/english_loanwords_tth.json`, 3,662+ words) with fallback dynamic G2P (`g2p_en`) mapped via calibrated acoustic confusion matrix argmax substitutions.
  - Output: Reconciled canonical `TTH` phonetic targets (`hi'a khasi akituli`) aligning seamlessly against Cherokee ASR emissions.
- **Code-Switched Ground Truth Preparation & Compound Clitic Segmentation (`digohwelisgi.cherokee.codeswitching:create_groundtruth_for_code_switched_syllabary`)**:
  - Script-level token discrimination: Pure Cherokee Syllabary (`TokenType.CHEROKEE_SYLLABARY`), Pure English (`TokenType.ENGLISH`), Compound Latin Stem + Syllabary Clitic (`TokenType.COMPOUND_CLITIC`, e.g. `JayᎢ` -> `Jay` + `Ꭲ`, `WellingᏛ` -> `Welling` + `Ꮫ`), and Punctuation (`TokenType.PUNCTUATION`).
  - Strict isolation: English tokens pass strictly through `SyntheticTargetProjector` (zero Cherokee DG-to-TTH consonant mutation), eliminating double conversion corruption (`Soldier` -> `hsowtsa`, `Jay` -> `tse`).
  - Compound clitic projection: English stem projected via confusion matrix, syllabary clitic directly normalized, fused into unified canonical TTH (`JayᎢ` -> `tsei`, `WellingᏛ` -> `wawintv`).
  - Preserves multi-tier word metadata (Syllabary, English, Reconciled) and speaker labels (`extract_speaker_prefix`).
- **Syllabary Enrichment (`digohwelisgi.cherokee.enrichment` / `digohwelisgi.pipelines.enrichment`)**:
  - Anchor: `SYLLABARY` (`ᎠᏓᎴᏂᏍᎬ`).
  - Acoustic Observation: `TTH` emissions (`athaleniskv`).
  - Output: Reconciled `TTH` phonetic representation (`athaleniskv`).
- **Desktop Transcriber (`syllabary_transcriber`)**:
  - Backend transcribes audio to `TTH` $\rightarrow$ frontend renders native `SYLLABARY` glyphs.

## Documentation Maintenance & Freshness
- **Keep Documentation Synchronized**: When adding new modules, refactoring subsystem architecture, changing CLI tools, or modifying testing procedures, immediately update this `AGENTS.md` and the corresponding guide in `backlog/docs/` (managed via `backlog doc` CLI).
- **Prune Obsolete Instructions**: Actively remove superseded workflows, deprecated flags, or outdated setup instructions to prevent agent confusion.
