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

## Linguistic & Domain Conventions
- Ensure clear separation between Cherokee Syllabary (Unicode), Latin phonetics, and phonetic alignment tokens.
- Keep domain transformations pure (see `Types & Maps` rules) within `alignment/` and `syllabary_enrichment/`.

## Documentation Maintenance & Freshness
- **Keep Documentation Synchronized**: When adding new modules, refactoring subsystem architecture, changing CLI tools, or modifying testing procedures, immediately update this `AGENTS.md` and the corresponding guide in `docs/`.
- **Prune Obsolete Instructions**: Actively remove superseded workflows, deprecated flags, or outdated setup instructions to prevent agent confusion.
