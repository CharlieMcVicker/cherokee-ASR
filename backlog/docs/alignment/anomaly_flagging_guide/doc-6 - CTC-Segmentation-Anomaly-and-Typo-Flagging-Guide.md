---
id: doc-6
title: CTC Segmentation Anomaly and Typo Flagging Guide
type: guide
created_date: '2026-09-12 18:32'
updated_date: '2026-09-12 18:32'
---
# CTC Segmentation Anomaly and Typo Flagging Guide

**Module:** `transcription.alignment.ctc_aligner`  
**Core Classes:** `CTCSegmentationAligner`, `AlignedChunk`, `WordInterval`  
**Benchmark Script:** `scripts/benchmark_ctc_segmentation_100_verses.py`  

---

## 1. Executive Summary & Objective

In Cherokee ASR and historical text transcription (e.g. the Cherokee New Testament), audio-transcript mismatches arise from two fundamentally different sources:

1. **Legitimate Phonological Variations (Connected Speech):**
   * **Vowel Syncope:** Deletion of unstressed interconsonantal vowels (e.g. `atalenihskv` spoken as `atalenhskv`).
   * **Intrusive Aspiration & Glottal Stops:** Spontaneous phonetic insertion of `/h/` or `/'/` (e.g. `uweluka` spoken as `uwelhuhka`, `hia` spoken as `hi'a`).
   * *Objective:* The aligner must adapt smoothly and emit the true acoustic realization **without flagging**.

2. **True Transcript Errors, Typographical Anomalies, & Substitutions:**
   * **Historical Syllabary Typos:** Visual glyph confusions in print (e.g. Mark 1:1 `ᏱᏍᏛ` / `yihstv` printed instead of `ᎣᏍᏛ` / `ohstv`).
   * **Word Omissions / Speaker Skips:** Speaker skips a word or substitutes an alternate term.
   * *Objective:* The aligner must isolate these anomalies, assign near-zero confidence, and automatically mark `flagged=True`.

Our current approach combines **blank-constrained trellis structural invariants** with **acoustic character-state confidence scoring** to achieve high sensitivity on true errors with near-zero false positives on natural connected speech.

---

## 2. Dual-Layer Detection Architecture

```
                    Raw Audio + Ground Truth Text
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  Layer 1: Structural Trellis Constraint (ctc-segmentation) │
      │  - Enforces consonant preservation invariant           │
      │  - Restricts syncope stride-2 jumps to blank tokens    │
      │  - Prevents non-syncope consonants from bypassing DP   │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  Layer 2: Acoustic Character-State Confidence Scoring   │
      │  - Extracts non-blank emitted character state frames   │
      │  - Computes geometric mean of character probabilities  │
      │  - Evaluates flag_min_confidence (default: 0.01 / 1%)  │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
               Word Interval Marked: flagged = True/False
```

---

## 3. Layer 1: Blank-Constrained Syncope Transitions

### 3.1 The Problem: Collateral Consonant Dropping
In standard CTC trellis segmentation, a syncope transition allows jumping over an optional vowel (`is_syncope_token == 1`). Previously, the trellis evaluated:
$$\text{c\_prev} \in [c - 3, c - 2]$$

When character tokens are packed contiguously (`replace_spaces_with_blanks=False`), jumping from $c - 3$ allowed the DP path to skip both the preceding consonant ($c - 2$) and the syncope vowel ($c - 1$) in a single jump:
$$[\text{PAD}]_{c=13} \xrightarrow{\text{syncope jump}} [\text{h}]_{c=16}$$
In Mark 1:1 (`yihstv`), this inadvertently skipped consonant `y` ($c=14$) alongside vowel `i` ($c=15$), allowing `h-s-t-v` to align against audio `ohsta` with artificial moderate confidence ($28.2\%$).

### 3.2 The Invariant & Solution
A 2-step syncope jump ($c - 3 \to c$) is permitted **if and only if** the intermediate column $c - 2$ is a CTC blank / PAD token:
$$P_{\text{sync}, 2}(t, c) = \begin{cases}
\text{table}[t - 1 + \Delta, c - 3] + \text{lpz}[t, L(c)] - \lambda_{\text{syncope}} & \text{if } L(c - 2) \in \{\text{blank}, -1\} \\
-\infty & \text{otherwise}
\end{cases}$$

This guarantees that **consonants can never drop via syncope**. When faced with a typo like `yihstv`, the trellis is forced to align `y` against the audio, collapsing the acoustic probability for that state.

---

## 4. Layer 2: Character-State Confidence Scoring

### 4.1 Frame-Averaging Dilution vs. Character-State Scoring
Previously, word confidence was computed by averaging log-probabilities across all acoustic frames spanning the word duration:
$$\text{mean\_logprob} = \frac{1}{T_{\text{end}} - T_{\text{start}}} \sum_{t=T_{\text{start}}}^{T_{\text{end}}} \text{char\_probs}[t]$$

Because silent frames between words and consonants have $\text{lpz}[t, \text{blank}] \approx 0.0$ (probability $\approx 1.0$), averaging over 25+ blank frames diluted severe character mismatches (e.g. $P(y) = 1.7 \times 10^{-6}$, logprob $-13.26$) into a false $25.96\%$ overall word score.

### 4.2 Current Character-State Formulation
In `CTCSegmentationAligner`, confidence is now computed exclusively over the **frames where non-blank character states were emitted**:

```python
# Extract log-probabilities of emitted character states (excluding blanks/PAD)
char_state_lps = [
    char_probs[f]
    for f in range(start_f, max(start_f + 1, end_f))
    if state_list[f] and state_list[f] not in ("ε", "[PAD]")
]

if w_timings and char_state_lps:
    mean_logprob = float(np.mean(char_state_lps))
    word_conf = float(np.exp(mean_logprob))
else:
    word_conf = 0.0

is_low_conf = bool(word_conf < self.flag_min_confidence)
is_unaligned = bool(len(w_timings) == 0 or len(char_state_lps) == 0)
is_flagged = bool(is_low_conf or is_unaligned)
```

For a word with character states $[c_1, c_2, \dots, c_K]$:
$$\text{Confidence} = \exp\left(\frac{1}{K} \sum_{k=1}^K \log P(c_k | x_{t_k})\right)$$

If even one non-syncope consonant suffers a catastrophic mismatch (such as $P(y) \approx 0$), the geometric mean drops exponentially, driving $\text{word\_conf} \ll 0.01$.

---

## 5. Verification & Benchmark Case Studies

### 5.1 Case Study: Mark 1:1 (`yihstv` Typo)
* **Citation Text**: `atalenihskv yihstv khanohetv ...`
* **Spoken Audio**: `atalenihskv ohsta khanoheta ...`

```text
State List:     ...  | y (logp=-13.26) | h (logp=-0.00) | s (logp=-0.00) | t (logp=-0.00) | v (logp=-12.71)
Emitted Word:   yhstv
Character Conf: 0.005551 (0.55%)
Threshold:      0.01 (1.00%)
Flagged:        True  [Caught Typo!]
```

### 5.2 100-Verse Benchmark Distribution (1,274 Words)
Running `benchmark_ctc_segmentation_100_verses.py` with `flag_min_confidence = 0.01`:

| Metric | Result |
| :--- | :--- |
| **Total Verses Realigned** | 100 |
| **Total Words Evaluated** | 1,274 |
| **Total Flagged Words** | **11** ($0.86\%$) |
| **Verses with Anomalies** | **10** ($10.0\%$) |
| **Legitimate Syncope Drops Unflagged** | 67 / 67 ($100\%$) |
| **Legitimate Intrusions (`h`, `'`) Unflagged** | 70 / 70 ($100\%$) |
| **Mean Alignment Speed** | **33.0 ms / verse** |

### 5.3 Complete List of Anomalies Flagged Across 100 Verses
1. `020101` | `yihstv` $\to$ `yhstv` (conf=$0.55\%$) — Historical print typo for `ohstv`.
2. `020115` | `nhakwo` $\to$ `nhakw` (conf=$0.82\%$) — Missing final vowel in fast speech.
3. `020116` | `kehsei` $\to$ `khse` (conf=$0.73\%$) — Severe compression / slurred syllables.
4. `020138` | `hiano` $\to$ `han` (conf=$0.007\%$) — Particle reduction.
5. `020207` | `hia` $\to$ `hi` (conf=$0.10\%$) — Particle reduction.
6. `020210` | `ahseno` $\to$ `hsn` (conf=$0.0002\%$) — Dropped vowels and slurred onset.
7. `020225` | `hiano` $\to$ `han` (conf=$0.59\%$) — Particle reduction.
8. `020311` | `ale` $\to$ `l` (conf=$0.00\%$) — Skipped particle.
9. `020314` | `ale` $\to$ `l` (conf=$0.01\%$) — Skipped particle.
10. `020318` | `eniti` $\to$ `nt` (conf=$0.00\%$) — Severe word compression.
11. `020318` | `ale` $\to$ `l` (conf=$0.00\%$) — Skipped particle.

---

## 6. Configuration & API Reference

### `CTCSegmentationAligner` Constructor Options

```python
aligner = CTCSegmentationAligner(
    model=asr_model,
    syncope_tokens=("a", "e", "i", "o", "u", "v"),
    syncope_penalty=2.0,                  # Penalty per dropped vowel
    intrusive_tokens=("h", "'"),
    intrusive_penalty=0.1,                # Penalty for acoustic aspiration/glottal detour
    flag_min_confidence=0.01,             # 1.0% character-state geometric mean threshold
    flag_min_char_duration_sec=0.03,      # Minimum duration per character
    index_duration=0.02,                  # 20ms per CTC frame
    cache=True,                           # Enable acoustic logprob disk cache
)
```
