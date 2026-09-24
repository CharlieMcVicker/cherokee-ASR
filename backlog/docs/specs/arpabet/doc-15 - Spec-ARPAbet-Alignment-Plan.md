---
id: doc-15
title: 'Spec: ARPAbet Alignment Plan'
type: specification
created_date: '2026-09-24 17:30'
updated_date: '2026-09-24 17:30'
---
### Phase 1: Data Preparation & Segmentation

* **Source Selection:** Download LibriSpeech (`dev-clean` or `test-clean`) containing clear audio alongside sentence-level transcripts.
* **Word Alignment & Slicing:**
* Run the English corpus through Montreal Forced Aligner (MFA) or a pretrained English CTC aligner to extract word-level timestamps (`start_time`, `end_time`).
* Slice the full audio files into individual single-word `.wav` clips to prevent Cherokee language-model context from compounding across word boundaries.


* **Metadata Export:** Build a manifest mapping each word clip to its target orthographic text:
```json
{"clip_id": "0001", "audio_path": "words/0001.wav", "word": "coffee"}

```



---

### Phase 2: Feature Extraction & Inference

* **English G2P:**
* Process the ground-truth text with `g2p_en`.
* Strip numeric stress markers (e.g., `AA1` $\to$ `AA`) to yield standardized ARPAbet token sequences.


* **Cherokee ASR Inference:**
* Pass each sliced single-word audio clip through the Cherokee ASR model.
* Extract greedy character/phonetic outputs (e.g., Syllabary or Latin Cherokee phonetics such as `ko-wi`).
* Tokenize Cherokee output into discrete phone/syllable units.



---

### Phase 3: Alignment & Statistical Matrix Formulation

Because ARPAbet tokens and Cherokee phonetic tokens share no common alphabet, standard string Levenshtein edit distance requires **Iterative Expectation-Maximization (EM)** or **Dynamic Time Warping (DTW)** over substitution probabilities.

1. **Initial Seed Mapping:** Define a soft acoustic initialization matrix (e.g., ARPAbet labials mapping to Cherokee `w` or `k` with slightly higher initial weights than unrelated sounds).
2. **Alignment Traceback:** Run dynamic programming (Needleman-Wunsch / Levenshtein alignment) using the cost matrix:
* Cost of substitution: $-\log P(\text{Cherokee Phone} \mid \text{ARPAbet Token})$.
* Insertion cost: Epenthetic Cherokee vowels ($\epsilon \to V$).
* Deletion cost: Dropped English sounds (e.g., coda consonants $\to \epsilon$).


3. **Frequency Aggregation:**
* Accumulate alignment counts across all dataset pairs:

$$\text{Count}(\text{ARPAbet}_i \to \text{Cherokee}_j)$$


* Re-estimate the conditional probabilities $P(\text{Cherokee}_j \mid \text{ARPAbet}_i)$.
* Iterate 3–5 cycles until alignments stabilize.


4. **Outlier Removal:** Prune mappings occurring below a set frequency threshold (e.g., $< 5\%$ probability given the source English phone).

---

### Phase 4: Production Pipeline Execution

Once the statistical map is built, the runtime pipeline runs deterministically without needing the English ASR or audio alignment tools:

```
[Spoken English Word in Code-Switched Input]
                      │
                      ▼
            [g2p_en Text Analysis]
                      │
             (ARPAbet Phonemes)
                      │
                      ▼
       [Statistical Cherokee Mapping Matrix]
       (Applies phonotactics & epenthesis rules)
                      │
                      ▼
     [Target Cherokee Phonetic Ground Truth]
                      │
                      ▼
  [CTC / Levenshtein Alignment against Cherokee ASR]

```

* **Step 1:** Extract the English segment from the code-switched text.
* **Step 2:** Convert to ARPAbet tokens via `g2p_en`.
* **Step 3:** Query the translation dictionary for the argmax Cherokee phonetic equivalents.
* **Step 4:** Emit the synthetic Cherokee phonetic string to serve as the ground truth target for aligning the Cherokee ASR model's predictions.
