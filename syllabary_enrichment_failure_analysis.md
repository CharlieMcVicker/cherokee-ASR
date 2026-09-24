# Syllabary ASR Phonetic Reconciliation Debug & Failure Analysis Plan

## 1. Overview & Problem Statement

The goal of phonetic reconciliation is to enrich ground-truth Cherokee Syllabary (ᏪᎳᏍᏗ) transliterations with acoustic features emitted by the fine-tuned Wav2Vec2 ASR model—specifically **vowel syncopation (deletion)** and **aspiration / glottal activity (`h`, `'`, and laryngeal $t \rightarrow th / k \rightarrow kh$ toggles)**.

In our first end-to-end evaluation run across 1,466 ground-truth sentence audio files:
- **Baseline Raw ASR CER (`emitted_text` vs Target GT)**: **0.44%** (Overall) / **2.25%** (Test split)
- **Reconciled Phonetics CER (`reconciled_phonetics` vs Target GT)**: **9.73%** (Overall) / **9.89%** (Test split)
- **Relative Improvement ($\Delta$ CER)**: **-2,110%**

This indicates a severe degradation introduced by the current reconciliation process.

---

## 2. Key Hypotheses: What Might Be Going Wrong?

### Hypothesis A: Target Notation Schema Discrepancy (t/th vs d/g vs t/k)
- **The Issue**: `CHEROKEE_SYLLABARY_MAP` historically mapped Cherokee Syllabary characters (e.g. Ꭶ, Ꮣ, Ꭸ, Ꮥ) to base transliterations using **`g/d`** voicing (`ga`, `da`, `ge`, `de`). However, our target ground-truth dataset was pre-normalized into the **`t/th` and `k/kh`** acoustic scheme (`ka`, `ta`, `kha`, `tha`, `nikhv`, `uwesdanelidoho`).
- **Impact**: The base transliteration engine produces `d` and `g` instead of `t` and `k`. If ASR emits `t` or `k`, or if the target ground-truth expects `t` / `k`, starting from `d` or `g` forces artificial errors on almost every single stop consonant in the Cherokee language.

### Hypothesis B: Syllabary Character-to-ASR Alignment Drift
- **The Issue**: `align_character_syllable()` aligns syllabary characters (Unicode Cherokee) against raw emitted ASR phonetic text (Latin alphabet) using character-level Levenshtein edit distance.
- **Impact**: Because Cherokee syllabary characters are single 16-bit Unicode characters while ASR output is multi-character Latin text (e.g., Syllabary `Ꮝ` vs ASR `s`, Syllabary `Ꮖ` vs ASR `gwa`), aligning character-by-character without converting syllabary to Latin transliteration first creates distorted aligned pairs, pairing wrong syllabary slots with incorrect ASR slices.

### Hypothesis C: Over-aggressive Vowel Syncopation
- **The Issue**: If an ASR window is slightly truncated or misaligned, `emitted_clean` for a syllable may omit a vowel character. `_enrich_single_syllable()` instantly truncates the final vowel of the base syllable.
- **Impact**: Legitimate vowels in ground-truth target text get deleted whenever ASR misses a vowel or misaligns by even one character offset.

### Hypothesis D: Space & Word Boundary Destruction
- **The Issue**: Word boundary spaces are stripped or improperly inserted during syllable concatenation, causing word boundaries to shift or disappear, inflating Character Error Rate (CER).

---

## 3. Immediate Testing & Inspection Framework

To see the exact outputs at every step and isolate the root cause, we will build a step-by-step diagnostic CLI tool: `transcription/syllabary_enrichment/inspect_pipeline.py`.

```
┌─────────────────────────────────────────────────────────┐
│              Step 1: Input Syllabary & Base             │
│        Syllabary: ᏂᎬ ᎤᏪᏍᏓᏁᎵᏙᎰ ᎠᏗᎭ                     │
│        Base Trans: nigv uwesdanelidoho adiha            │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│              Step 2: Raw ASR Emitted Text               │
│        Emitted: nikhv uwehstanelitoho atiha             │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│        Step 3: Character / Syllable Aligned Pairs       │
│        [("Ꮒ", "nikh"), ("Ꭼ", "v"), (" ", " "), ...]     │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│             Step 4: Syllable-by-Syllable Rules          │
│        Input: base="nigv", emitted="nikhv"              │
│        Action: 'g' -> 'kh' (Laryngeal Toggle)           │
│        Output: "nikhv"                                  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│            Step 5: Target vs Reconciled Diff            │
│        Target:     nikhv uwehstanelitoho atiha          │
│        Reconciled: nikhv uwestanelitoho atiha           │
│        Diff:       -weh +wes                            │
└─────────────────────────────────────────────────────────┘
```

### Planned Diagnostic Commands

1. **Detailed Single-Record Visualizer**:
   ```bash
   conda run -n cherokee-asr python -m transcription.pipelines.enrichment.pipeline --record-id "Sentence_for_entry_1368_01.wav"
   ```
2. **Top-N Error Breakdown & Diff Analysis**:
   ```bash
   conda run -n cherokee-asr python -m transcription.pipelines.enrichment.pipeline --top-errors 20
   ```

---

## 4. Next Actions for Immediate Execution

1. Build `inspect_pipeline.py` script to visually render inputs, intermediate aligned slices, rule decisions, target ground-truth, and diffs for any sample.
2. Fix **Hypothesis A** (Syllabary transliteration scheme alignment with `t/th` and `k/kh` target conventions).
3. Fix **Hypothesis B** (Transliterating syllabary to base Latin before running Levenshtein DTW character alignment).
