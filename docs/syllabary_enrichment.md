# Cherokee Syllabary Phonetic Enrichment Documentation

This guide provides a comprehensive technical reference for the **Cherokee Syllabary Phonetic Enrichment and Reconciliation Pipeline** in `transcription.syllabary_enrichment`.

---

## Table of Contents

1. [Core Concepts](#1-core-concepts)
2. [Phonetic Rules Engine](#2-phonetic-rules-engine)
3. [Alignment Engine](#3-alignment-engine)
4. [Batch Inference & Disk Caching](#4-batch-inference--disk-caching)
5. [Evaluation Framework](#5-evaluation-framework)
6. [Step-by-Step Diagnostic CLI](#6-step-by-step-diagnostic-cli)
7. [Programmatic Python Usage](#7-programmatic-python-usage)

---

## 1. Core Concepts

### The Reconciliation Problem

Standard written Cherokee utilizes the 85-character **Cherokee Syllabary** (invented by Sequoyah). While syllabary text provides canonical morphological structures and word boundaries, it does not explicitly capture several spoken phonetic phenomena:

- **Vowel Syncopation / Deletion:** Weak vowels frequently drop in rapid speech (e.g., *adalenisgv* $\rightarrow$ *athaleniskv*).
- **Consonant Aspiration & Laryngeal Shifts:** Unvoiced stops and sonorants alternate between plain and aspirated forms ($t \rightarrow th$, $k \rightarrow kh$, $l \rightarrow lh$, $n \rightarrow nh$, $w \rightarrow wh$, $y \rightarrow yh$).
- **Pre-aspiration & Post-vocalic Aspiration:** Inherent breath sounds ($h-$ and $-h$) omitted in standard orthography.
- **Glottal Stops:** Acoustic glottal stops ($'$) that occur phonemically or as sandhi effects.

Conversely, acoustic Automatic Speech Recognition (ASR) acoustic models output fine-grained phonetic transcriptions reflecting audio emissions, but can suffer from character substitutions, misheard consonants, or dropped syllables.

```
       +------------------------------------+
       | Ground-Truth Cherokee Syllabary    |  (Immutable Structural Anchor)
       | (e.g., "ᎠᏓᎴᏂᏍᎬ" -> "adalenisgv")    |
       +-----------------+------------------+
                         |
                         v
       +-----------------+------------------+
       |   Fine-Grained Dynamic             |
       |   Programming Alignment            |  (transcription.syllabary_enrichment.alignment_engine)
       +-----------------+------------------+
                         ^
                         |
       +-----------------+------------------+
       | Raw ASR Acoustic Emissions         |  (Acoustic Evidence)
       | (e.g., "athaleniskv")              |
       +-----------------+------------------+
                         |
                         v
       +-----------------+------------------+
       |   Phonetic Rule Merger Engine      |  (transcription.syllabary_enrichment.enrich_syllabary)
       +-----------------+------------------+
                         |
                         v
       +-----------------+------------------+
       | Reconciled Phonetic Transcription  |  (e.g., "adalenisgv" + "athaleniskv" -> "adaleniskv")
       +------------------------------------+
```

### The Solution: Cherokee Syllabary as Immutable Anchor

The reconciliation engine treats the **Cherokee Syllabary as the immutable structural ground truth**. The pipeline:
1. Translates the syllabary into its base phonetic representation using [`CHEROKEE_SYLLABARY_MAP`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/utils/syllabary_map.py#L104-L106).
2. Uses dynamic programming (Needleman-Wunsch / DTW block alignment) to align each syllabary character directly against the corresponding ASR emitted acoustic time window.
3. Applies phonological merge rules syllable-by-syllable, enriching the base transliteration with verified acoustic features (aspiration, syncopation, glottal stops) while rejecting hallucinated or out-of-order ASR errors.

---

## 2. Phonetic Rules Engine

The phonetic reconciliation logic is implemented in [`transcription.syllabary_enrichment.enrich_syllabary`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/syllabary_enrichment/enrich_syllabary.py).

### Syllabary Character Map

Centralized transliterations are defined in [`transcription.utils.syllabary_map`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/utils/syllabary_map.py):
- Maps all 85 syllabary characters (Unicode `U+13A0`--`U+13F5` and `U+AB70`--`U+ABBF`).
- Reflects unified phonetic respellings (e.g., `Ꮏ` $\rightarrow$ `nha` instead of `hna`).
- Provides reverse lookup [`phonetics_to_syllabary()`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/utils/syllabary_map.py#L152-L214) with pre-aspiration and cluster fallback handling (`hska` $\rightarrow$ `ᏍᎦ`, `thv` $\rightarrow$ `Ꮫ`).

### Phonetic Rules Overview

Each aligned pair `(syllabary_char, emitted_slice)` is evaluated by [`_enrich_single_syllable(base, emitted)`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/syllabary_enrichment/enrich_syllabary.py#L87-L178):

#### 1. Rule 1: Vowel Syncopation / Deletion
When ASR emits a consonant without a vowel for a CV (consonant-vowel) syllable, the vowel is dropped:
- **Base `da`** aligned to ASR `t` $\rightarrow$ `d` (vowel dropped).
- **Base `sv`** aligned to ASR `s` $\rightarrow$ `s` (vowel dropped).
- **Pure Vowel `a`** aligned to empty ASR $\rightarrow$ `""` (vowel elided).
- **Post-vocalic `h` cleanup:** If a syllable ends in `-h` but the vowel is syncopated, trailing post-vocalic `h` is stripped.

#### 2. Rule 2: Aspiration Transfer & Laryngeal / Digraph Shifts
ASR-detected aspiration shifts are systematically mapped onto base consonants:

| Base Consonant Prefix | ASR Indicator Trigger | Enriched Output Prefix | Example Syllabary |
| :--- | :--- | :--- | :--- |
| `t`, `d` | `th` | `th` | `Ꮣ` (*da*) $\rightarrow$ *tha* |
| `k`, `g` | `kh` | `kh` | `Ꭶ` (*ka*) $\rightarrow$ *kha* |
| `tl`, `l` | `lh` | `lh` | `Ꮭ` (*tla*) $\rightarrow$ *lha*, `Ꮃ` (*la*) $\rightarrow$ *lha* |
| `n`, `hn` | `nh` | `nh` | `Ꮎ` (*na*) $\rightarrow$ *nha* |
| `w`, `hw` | `wh` | `wh` | `Ꮹ` (*wa*) $\rightarrow$ *wha* |
| `y`, `hy` | `yh` | `yh` | `Ꮿ` (*ya*) $\rightarrow$ *yha* |
| `r`, `hr` | `rh` | `rh` | `Ꮈ` (*lv*) $\rightarrow$ *rhv* |
| `s` | `sh` | `sh` | `Ꮜ` (*sa*) $\rightarrow$ *sha* |
| `c` | `ch` | `ch` | `Ꮳ` (*tsa*) $\rightarrow$ *cha* |
| `ts` | `tsh` | `tsh` | `Ꮵ` (*tsi*) $\rightarrow$ *tshi* |

- **Pre-aspiration (`h-`):** If the emitted slice starts with `h` and the base does not, `h` is prepended (e.g., base `ka` + ASR `hka` $\rightarrow$ `hka`).
- **Post-vocalic aspiration (`-h`):** If the emitted slice ends with `h` following a vowel, `h` is appended (e.g., base `a` + ASR `ah` $\rightarrow$ `ah`).

#### 3. Rule 3: Glottal Stop Handling (`'`)
- **Standalone Onset Glottal Stops Filtered:** Standalone onset glottal stops from ASR are dropped unless accompanied by another onset consonant (e.g., ASR `'a` for base `ya` drops `'` to produce `ya`, preventing misheard glottal substitutions).
- **Medial / Coda Glottal Stops Preserved:** Glottal stops occurring at the end or inside a syllable (e.g., `da'` or `a'`) are preserved.

---

## 3. Alignment Engine

Implemented in [`transcription.syllabary_enrichment.alignment_engine`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/syllabary_enrichment/alignment_engine.py).

### Data Structures

```python
@dataclass
class SyllableAlignment:
    syllabary_char: str     # Original Cherokee character (e.g. "Ꭰ")
    base_phonetic: str      # Base un-enriched transliteration (e.g. "a")
    emitted_text: str       # Aligned ASR character slice (e.g. "a")
    syl_start_idx: int      # Character offset in syllabary string
    syl_end_idx: int
    emitted_start_idx: int  # Character offset in emitted string
    emitted_end_idx: int
```

### Dynamic Programming Algorithm

[`align_character_syllable_detailed(syllabary_text, emitted_text)`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/syllabary_enrichment/alignment_engine.py#L101-L226) maps $M$ syllabary units (characters + whitespace + punctuation) to $N$ emitted ASR characters using a customized 2D DP cost matrix:

1. **Deletion Option:** Syllable mapped to empty slice (cost $1.5$, or $0.5$ for whitespace/punctuation).
2. **Expansion Option:** Syllable mapped to $k$ emitted characters ($1 \le k \le \text{len}(\text{base}) + 3$).
3. **Phonetic Cost Function:**
   - Exact match: cost $0.0$.
   - Vowel-to-vowel substitution (`a`, `e`, `i`, `o`, `u`, `v`): cost $0.5$.
   - Consonant group substitution (`d`/`t`/`th` or `g`/`k`/`kh`): cost $0.3$.
   - Other substitutions: cost $1.0$.
   - Insertions / deletions inside slice: cost $0.8$.
4. **Fallback Mechanism:** If no valid DP path exists, [`_proportional_fallback()`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/syllabary_enrichment/alignment_engine.py#L256-L278) distributes character boundaries proportionally across the emitted text.

---

## 4. Batch Inference & Disk Caching

Implemented in [`transcription.syllabary_enrichment.batch_inference_aligner`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/syllabary_enrichment/batch_inference_aligner.py).

### Pipeline Execution

```bash
python -m transcription.syllabary_enrichment.batch_inference_aligner \
    data/manifests/test_manifest.json \
    --output-cache data/results/aligned_manifest_cache.json \
    --batch-size 16 \
    --num-workers 4
```

### Features:
- Automatically loads the best model checkpoint configuration via [`get_best_model_config()`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/utils/model_utils.py#L22-L37).
- Uses multi-worker CPU CTC decoding pools for high throughput.
- Persists aligned pairs to JSON cache; subsequent runs load instantly unless `--force-recompute` is provided.

---

## 5. Evaluation Framework

The evaluation script [`transcription.syllabary_enrichment.evaluate_reconciliation`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/syllabary_enrichment/evaluate_reconciliation.py) measures phonetic reconciliation performance against ground truth phonetics.

### Metrics Computed

- **Raw CER:** Character Error Rate between raw ASR `emitted_text` and `target_phonetics`.
- **Reconciled CER:** Character Error Rate between `reconciled_phonetics` and `target_phonetics`.
- **Delta CER ($\Delta\text{CER}$):** Relative percentage error reduction:
  $$\Delta\text{CER} = \frac{\text{Raw CER} - \text{Reconciled CER}}{\text{Raw CER}} \times 100\%$$

### CLI Usage

```bash
python -m transcription.syllabary_enrichment.evaluate_reconciliation \
    training_data/processed/split_audio_syl_target.csv \
    --output-cache data/results/aligned_cache.json \
    --output-artifact data/results/eval_results.json
```

### Summary Output Example

```
================================================================================
PHONETIC RECONCILIATION EVALUATION SUMMARY
================================================================================
Split           | Count    | Raw CER    | Reconciled CER  | Δ CER (%) 
--------------------------------------------------------------------------------
train           | 1250     | 0.1420     | 0.0812          | +42.82%   
validation      | 150      | 0.1510     | 0.0865          | +42.72%   
test            | 150      | 0.1485     | 0.0840          | +43.43%   
overall         | 1550     | 0.1435     | 0.0820          | +42.86%   
================================================================================
```

---

## 6. Step-by-Step Diagnostic CLI

The visual inspector [`transcription.syllabary_enrichment.inspect_pipeline`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/syllabary_enrichment/inspect_pipeline.py) provides formatted step-by-step traces of individual audio records.

### CLI Options

```bash
# Inspect a specific record by ID or audio filename substring
python -m transcription.syllabary_enrichment.inspect_pipeline \
    --manifest training_data/processed/split_audio_syl_target.csv \
    --output-cache data/results/aligned_cache.json \
    --record-id "audio_sample_0042"

# Inspect the top 5 records with highest error rate
python -m transcription.syllabary_enrichment.inspect_pipeline \
    --manifest training_data/processed/split_audio_syl_target.csv \
    --output-cache data/results/aligned_cache.json \
    --top-errors 5
```

### Sample Inspection Output

```
================================================================================
 PIPELINE INSPECTION: audio_sample_0042.wav (Split: test)
================================================================================

--- STEP 1: Syllabary Input & Base Transliteration ---
  Syllabary Input:       ᎠᏓᎴᏂᏍᎬ
  Base Transliteration:  adalenisgv

--- STEP 2: ASR Emitted Text ---
  ASR Emitted Text:      athaleniskv

--- STEP 3: Character / Syllable Aligned Slices ---
  Aligned Slices: ('Ꭰ'[a] -> 'a'), ('Ꮣ'[da] -> 'tha'), ('Ꮄ'[le] -> 'le'), ('Ꮒ'[ni] -> 'ni'), ('Ꮝ'[s] -> 's'), ('Ꭼ'[kv] -> 'kv')

--- STEP 4: Syllable Rule Action & Output ---
  Char: Ꭰ   | Base: a     | ASR: a      | Output: a      | Action: Base unchanged (a)
  Char: Ꮣ   | Base: da    | ASR: tha    | Output: tha    | Action: Enriched (da + ASR 'tha' -> tha)
  Char: Ꮄ   | Base: le    | ASR: le     | Output: le     | Action: Base unchanged (le)
  Char: Ꮒ   | Base: ni    | ASR: ni     | Output: ni     | Action: Base unchanged (ni)
  Char: Ꮝ   | Base: s     | ASR: s      | Output: s      | Action: Base unchanged (s)
  Char: Ꭼ   | Base: kv    | ASR: kv     | Output: kv     | Action: Base unchanged (kv)

--- STEP 5: Target vs Reconciled String Diff & CER ---
  Target Phonetics:      athaleniskv
  Reconciled Phonetics:  athaleniskv
  Inline Diff:           athaleniskv
  Raw ASR CER:           0.0000
  Reconciled CER:        0.0000
================================================================================
```

---

## 7. Programmatic Python Usage

### Example 1: Reconciling Syllabary with ASR Emissions

```python
from transcription.syllabary_enrichment.alignment_engine import (
    align_character_syllable,
    get_base_transliteration,
)
from transcription.syllabary_enrichment.enrich_syllabary import reconcile_phonetics

syllabary_text = "ᎠᏓᎴᏂᏍᎬ"
emitted_text = "athaleniskv"

# 1. Generate base transliteration
base_trans = get_base_transliteration(syllabary_text)
print(f"Base Transliteration: {base_trans}")  # "adalenisgv"

# 2. Align syllabary characters to ASR emitted text
aligned_pairs = align_character_syllable(syllabary_text, emitted_text)
print(f"Aligned Pairs: {aligned_pairs}")

# 3. Merge phonetic features
reconciled = reconcile_phonetics(
    syllabary_text=syllabary_text,
    base_transliteration=base_trans,
    emitted_text=emitted_text,
    aligned_pairs=aligned_pairs,
)
print(f"Reconciled Phonetics: {reconciled}")  # "athaleniskv"
```

### Example 2: Inspecting Detailed Alignments with Indices

```python
from transcription.syllabary_enrichment.alignment_engine import (
    align_character_syllable_detailed,
)

syllabary_text = "ᎣᏏᏲ"
emitted_text = "osiyo"

detailed = align_character_syllable_detailed(syllabary_text, emitted_text)
for align in detailed:
    print(
        f"Char: '{align.syllabary_char}' [{align.base_phonetic}] "
        f"-> Emitted: '{align.emitted_text}' "
        f"(Syllabary indices: {align.syl_start_idx}:{align.syl_end_idx}, "
        f"ASR indices: {align.emitted_start_idx}:{align.emitted_end_idx})"
    )
```

### Example 3: Converting Phonetic Transcriptions Back to Syllabary

```python
from transcription.utils.syllabary_map import phonetics_to_syllabary

# Supports aspiration clusters and tone-stripped inputs
phonetic_input = "osiyo thaleniskv"
syllabary_output = phonetics_to_syllabary(phonetic_input)
print(f"Converted Syllabary: {syllabary_output}")  # "ᎣᏏᏲ ᏔᎴᏂᏍᎬ"
```
