
# Architecture & Implementation Plan: Syllabary + ASR Phonetic Reconciliation

## Project Context

The goal of this project is to combine noisy **Automatic Speech Recognition (ASR)** outputs with **ground-truth Cherokee Syllabary** to generate accurate surface-level phonetic representations of spoken Cherokee.

While the traditional Cherokee syllabary (ᏪᎳᏍᏗ) explicitly encodes vowel quality (when vowels are present) and consonant place/manner of articulation, it does not explicitly encode fine-grained phonological phenomena such as **pre-aspiration ($h$)**, **laryngeal contrasts ($t/th$ series)**, or **vowel deletion/syncopation**. The ASR model predicts phonetics directly from audio, capturing acoustic laryngeal and aspiration cues, but it is subject to acoustic noise and misrecognitions.

By using the **syllabary as an immutable structural anchor** and enriching it with **acoustic features injected from the aligned ASR transcript**, we aim to synthesize ground-truth quality phonetics $(\text{Audio} + \text{Syllabary} \rightarrow \text{Target Phonetics})$.

---

## Current Status & Completed Work

1. **Target Phonetic Scheme Defined:** Utilizes a $t/th$ notation system ($t$ = unaspirated/voiced, $th$ = aspirated/unvoiced) along with explicit pre-aspiration ($h$) and glottal stop ($'$) representations.
2. **Evaluation Strategy Filter:** Filtered the CND dataset to ensure syllabary and target transcript compatibility (dropping rows with mismatched vowel quality).
3. **Dataset Manifest Creation:** Created the foundational dataset manifest containing:
* Dataset split (`train`, `validation`, `test`)
* Source syllabary (`cherokee_syllabary` and base syllabic transliteration `text`)
* Audio file references and timing boundaries (`start`, `end`)
* Target ground-truth phonetic transcripts



---

## Implementation Work Items for Agent

```
               [ Data Manifest ]
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  Phase 1: Batch Inference & Alignment Pipeline │
└──────────────────────┬──────────────────────┘
                       │ (Outputs: emitted_text + word/syllable alignments)
                       ▼
┌─────────────────────────────────────────────┐
│  Phase 2: Phonetic Rule Merger Engine       │
└──────────────────────┬──────────────────────┘
                       │ (Outputs: reconciled_phonetics)
                       ▼
┌─────────────────────────────────────────────┐
│  Phase 3: Benchmarking & Test Framework     │
└─────────────────────────────────────────────┘

```

---

### Phase 1: Batch Inference & Forced Alignment Pipeline

**Objective:** Run batch inference using the current best ASR checkpoint across all manifest audio files and output aligned ASR text with syllable/word time-steps.

#### Requirements:

1. **Module Name:** `batch_inference_aligner.py`
2. **Inputs:**
* Path to data manifest JSON / JSONL file.
* Path to the best-performing ASR model checkpoint.


3. **Behavior:**
* Load audio segments specified by `start` and `end` timestamps.
* Execute batch inference to produce raw acoustic phonetic predictions (`emitted_text`).
* Run the pre-existing forced aligner to map predicted ASR phonetic tokens to the corresponding syllabary sequence.


4. **Outputs:**
* Updated manifest containing added fields:
* `emitted_text`: Raw ASR phonetic output ($t/th$ system).
* `aligned_pairs`: Array of aligned `(syllabary_segment, asr_segment)` tuples.





---

### Phase 2: Phonetic Rule Merger Engine

**Objective:** Implement the core reconciliation logic that enriches the syllabary skeleton with ASR acoustic features.

#### Requirements:

1. **Module Name:** `enrich_syllabary.py`
2. **Function Signature:**
```python
def reconcile_phonetics(syllabary_text: str, base_transliteration: str, emitted_text: str, aligned_pairs: list) -> str:
    """
    Merges ASR laryngeal/aspiration features onto the syllabary frame.
    """

```


3. **Core Reconciliation Rules:**
* **Rule A (Vowel Quality Anchor):** Vowel quality ($a, e, i, o, u, v$) is dictated strictly by the syllabary slot. Ignore ASR vowel substitutions.
* **Rule B (Consonant Place/Manner Anchor):** Base consonant place and manner are dictated by the syllabary slot. Block intrusive glides (e.g., ASR `/kwo/` on syllabary Ꭺ `/go/` drops `/w/`).
* **Rule C (Laryngeal Feature Injection):** Toggle $t \rightarrow th$ or $k \rightarrow kh$ if the aligned ASR segment indicates an aspirated/unvoiced phone.
* **Rule D (Pre-aspiration & Glottal Injection):** Inject pre-aspirated $h$ (especially in $s$-clusters, e.g., `s-gv` + `hskv` $\rightarrow$ `hskv`) and glottal stops ($'$) from the ASR prediction.
* **Rule E (Vowel Syncopation):** Drop the vowel in a syllabary $CV$ slot if the aligned ASR window indicates complete vocalic deletion.



---

### Phase 3: Orchestration, Testing & Benchmarking Framework

**Objective:** Orchestrate the pipeline from end to end and compute performance metrics split by dataset partition (`train`, `validation`, `test`).

#### Requirements:

1. **Module Name:** `evaluate_reconciliation.py`
2. **Metrics to Compute:**
* **Baseline Raw CER:** Character Error Rate between `emitted_text` (Raw ASR) and Target Ground-Truth Phonetics.
* **Reconciled CER:** Character Error Rate between `reconciled_phonetics` ($\text{Syllabary} + \text{ASR}$) and Target Ground-Truth Phonetics.
* **Relative Improvement:** Percentage reduction in error rate:

$$\Delta \text{CER} = \frac{\text{CER}_{\text{raw}} - \text{CER}_{\text{reconciled}}}{\text{CER}_{\text{raw}}} \times 100\%$$




3. **Reporting:**
* Print a formatted summary table broken down by split (`train`, `valid`, `test`, `overall`).
* Save a detailed evaluation artifact `eval_results.json` containing per-line predictions, target strings, and individual CER scores for error analysis.



---