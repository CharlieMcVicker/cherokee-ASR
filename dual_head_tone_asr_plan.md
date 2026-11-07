# Dual-Head Multi-Task Wav2Vec2 Cherokee ASR Implementation Plan

## 1. Executive Summary & Design Decisions

### Core Architecture & Strategy
* **Split-Top ("Y-Network") Model**: Initialized from pretrained wav2vec2 / XLS-R.
  * **Two Segmental Streams**: Both streams are segmental—one is a Consonant-Vowel (CV) model predicting base phonemic quality, and the other is a Moraic Tone model predicting discrete tone numbers per mora alongside onset separators.
  * **Shared Trunk**: Transformer layers 1–10 capture shared acoustic and phonetic grounding.
  * **Branch A (CV / Segmental Head)**: Cloned layers 11A–12A + Linear CTC head, outputting base consonants, vowels (`a, e, i, o, u, v`), and word boundaries (`|`). Trained on all data (CV-only Bible + tone-annotated orig).
  * **Branch B (Moraic Tone Head)**: Cloned layers 11B–12B + Linear CTC head, outputting moraic tones, onset separators (`C`), and word boundaries (`|`). Trained with masked loss on the 3-hour tone subset.
  * **Equal Depth & Bidirectional Context**: Both heads operate non-causally with identical backpropagation depth, avoiding layer warmup lag or final-layer pitch erasure.

### Head B Ground Truth Representation
* **Moraic Tone Tokens**: Tone numbers (1=low-fall, 2=low, 3=high, 4=superhigh) indexed per mora:
  * **Short Vowels**: Strictly marked as Mora 1 (e.g., short low $\rightarrow$ `2_1`, short high $\rightarrow$ `3_1`).
  * **Long Vowels & Contours**: Split into Mora 1 and Mora 2:
    * Long low (level): `2_1 2_2`
    * Long high (level): `3_1 3_2`
    * Low-fall (`21`): `2_1 1_2`
    * Rising (`23`): `2_1 3_2`
    * Falling (`32`): `3_1 2_2`
    * Superhigh (`34`): `3_1 4_2`
* **Onset / Consonant Separator**: `C` (one `C` per contiguous sequence of one or more consonants/glottal stops). Prevents adjacent short vowels with identical tones from merging. Omitted for word-initial vowel onsets.
* **Word Boundary**: `|` (word delimiter token).

### Training & Optimization Decisions
* **Batch Sampling**: 50/50 interleaved batches between CV-only (Bible) and tone-annotated (`orig`) datasets, ensuring steady, non-starving gradients to both branches.
* **Loss Weighting**: Masked CTC loss for Head B ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CTC\_cv}} + \lambda \cdot \mathcal{L}_{\text{CTC\_tone}}$, with $\lambda \approx 1.0$).
* **Early Stopping & Best Model Metric**: **Tone-only CER / TER** on `valid_orig` to prioritize tone branch convergence.
* **Inference Reconciliation**: Implement and compare both:
  1. *Fast Moraic Slicing*: Head A vowel emission intervals $[t_{\text{start}}, t_{\text{end}}]$ sample Head B's emissions directly.
  2. *Sequence DP alignment*: Needleman-Wunsch alignment using `C` and `|` boundary anchors.

---

## 2. Model Architecture Diagram

```
                              ┌────────────────────────────────────────┐
                              │  Raw 16kHz Cherokee Speech Waveform   │
                              └──────────────────┬─────────────────────┘
                                                 │
                                                 ▼
                              ┌────────────────────────────────────────┐
                              │  Wav2Vec2 Feature Extractor (7 Conv1D) │
                              └──────────────────┬─────────────────────┘
                                                 │
                                                 ▼
                              ┌────────────────────────────────────────┐
                              │ Shared Transformer Trunk (Layers 1-10) │
                              └──────────────────┬─────────────────────┘
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
                     ▼                                                       ▼
        ┌─────────────────────────┐                             ┌─────────────────────────┐
        │  Branch A (Layers 11-12)│                             │  Branch B (Layers 11-12)│
        │      CV (Base) Head     │                             │    Moraic Tone Head     │
        └────────────┬────────────┘                             └────────────┬────────────┘
                     │                                                       │
                     ▼                                                       ▼
        ┌─────────────────────────┐                             ┌─────────────────────────┐
        │       Linear CTC        │                             │       Linear CTC        │
        │ (Consonants + Vowels)   │                             │   (C, Moraic Tones, |)  │
        └────────────┬────────────┘                             └────────────┬────────────┘
                     │                                                       │
                     ▼                                                       ▼
             $\mathcal{L}_{\text{CTC\_cv}}$ (All samples)            $\mathcal{L}_{\text{CTC\_tone}}$ (Masked on 3h subset)
```

---

## 3. Comprehensive Spelling & Orthographic Normalization Pipeline

To ensure acoustic stability, eliminate dialectal and transcription artifacts, and prevent ground-truth ambiguity between Head A and Head B, the entire dataset and inference pipeline strictly enforces **Standard 2 (Contextual Phonotactic TTH)** alongside six canonical spelling rules.

### 3.1 Sibilant Pre-aspiration Standard (Standard 2: Contextual Phonotactic TTH)
* **Word-Initial `/s/`**: Strictly bare `s-` across all words, regardless of position in sentence (`sakwu`, `sookwili`, `skwaalheehsti`, `sunaalee`, `svvhi`).
  * *Linguistic Rationale*: Acoustically, pre-aspiration is vowel-offset breathiness; utterance onset cannot produce pre-aspiration before voicing begins. Cherokee surface phonotactics forbids initial `*hs-` clusters (*HH constraint).
* **Postvocalic / Medial `/s/`**: Respell as `hs` when preceded by non-`h`, non-`t`, non-space (`ohsiyo`, `ahskaya`, `tsihskwa`, `uweehstaaneliitoohoo`).
* **S-Clusters**:
  * Word-initial clusters remain bare: `sk-`, `st-`, `skw-` (`skwaalheehsti`, `skoohi`).
  * Medial clusters receive pre-aspiration: `-hsk-`, `-hst-`, `-hskw-` (`tsihskwa`, `ahskaya`).
* **Affricates**: Always plain `ts` and aspirated `tsh` (never `ths` or `hts`).

### 3.2 Glottal Stop Canonicalization
* Raw datasets contain multiple glottal variants: ASCII apostrophe `'` (`U+0027`), right quote `’` (`U+2019`), modifier letter `ʼ` (`U+02BC`), IPA glottal `ʔ` (`U+0294`), low-9 quote `‚` (`U+201A`), and question mark `?` (legacy ASCII).
* **Rule**: Unify all representations strictly to canonical ASCII apostrophe `'` (`U+0027`). Strip illegal post-consonantal glottal stops (`*C'`) per Cherokee phonotactic surface constraints.

### 3.3 Inter-vocalic Hiatus Resolution (Epenthetic Glottal Insertion)
* Cherokee phonotactics strictly prohibits vowel hiatus ($*VV$ across distinct syllables without an onset consonant).
* **Rule**: Insert an epenthetic glottal stop `'` between adjacent vowels across morpheme or syllable boundaries:
  $$\text{V}_1 \text{V}_2 \longrightarrow \text{V}_1' \text{V}_2 \quad (\text{e.g., } \text{ᎢᎾᎨᎢ} \rightarrow \text{inake'i})$$
  *(Note: Does not apply to long vowel pairs representing moras within the same syllable nucleus).*

### 3.4 Consonant Cascade & Rewrite Ordering
To prevent destructive double-conversions, consonant rewrites must follow strict priority ordering:
1. `t(?!s) -> th` (must run before `d -> t`)
2. `d -> t`
3. `k -> kh` (must run before `g -> k`)
4. `g -> k`
5. `j -> ts`
6. `ch -> tsh`
7. Voiceless Sonorant Inversions: `hn -> nh`, `hl -> lh`, `hw -> wh`, `hy -> yh` (acoustically voiceless sonorants $[n̥, l̥, w̥, j̥]$ where voiceless airflow precedes oral release).
8. Sibilant-lateral cluster: `sl(?=[aeiouv]) -> slh` (while preconsonantal lateral is `hsl`).
9. Contextual pre-aspiration: `re.sub(r"([^ht\s])s", r"\1hs", text)`.

### 3.5 Vowel Quantity & Mora Factoring
* **Head A (Segmental CV)**: Represents base phonemic quality only (`a, e, i, o, u, v`). Strips colons (`:`) and collapses all doubled vowels (`aa -> a`, `ee -> e`, etc.).
  * *Design Win*: Head A is entirely decoupled from vowel duration and speech rate variance.
* **Head B (Moraic Tone)**: Fully absorbs vowel quantity:
  * Short vowel $\rightarrow$ 1 mora token (`2_1`)
  * Long vowel $\rightarrow$ 2 mora tokens (`2_1 2_2` for level low, `2_1 3_2` for rising, etc.)
  * Contours are naturally factored across Mora 1 and Mora 2.

### 3.6 Punctuation, Diacritic & Non-Cherokee Filtering
* Clean formatting marks: asterisks `*` (dictionary headword bolding), brackets, parentheses, semicolons `;` (legacy vowel-length/pitch marker).
* Drop or flag foreign English loanword consonants not present in the Cherokee inventory (`b`, `f`, `p`, `x`, `z`) in native splits.

---

## 4. Implementation Phases & Testable Milestones

Each phase is strictly divided into **Implementation Tasks**, **Unit Tests**, and an **Integration Gate** that must pass before advancing.

```
┌────────────────────────────────────────────────────────┐
│ Phase 1: Orthography & Normalization Engine (Tier 2)   │
└───────────────────────────┬────────────────────────────┘
                            │ (Unit + Regression Tests Pass)
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Moraic Tone Target Generator & Reconciler    │
└───────────────────────────┬────────────────────────────┘
                            │ (100% Round-trip on Feeling Dataset)
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: Dual-Stream Dataset & Vocab Serialization    │
└───────────────────────────┬────────────────────────────┘
                            │ (Dataset Integrity Audit Pass)
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 4: Model Architecture (Wav2Vec2ForDualCTC)       │
└───────────────────────────┬────────────────────────────┘
                            │ (Synthetic Batch Gradient & Shape Pass)
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 5: Training Pipeline & Evaluation Harness       │
└───────────────────────────┬────────────────────────────┘
                            │ (50-Step Dry Run + Validation Check)
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 6: ASR Model Integration & Benchmark Evaluation │
└────────────────────────────────────────────────────────┘
```

---

### Phase 1: Orthography & Spelling Normalization Engine

#### Implementation Tasks
* [ ] **T1.1**: Fix contextual pre-aspiration regex in `digohwelisgi/cherokee/orthography/tones.py`:
  ```python
  s = re.sub(r"([^ht\s])s", r"\1hs", s)
  ```
* [ ] **T1.2**: Implement `standardize_glottal_stops(text: str) -> str` in `tones.py` unifying `['’ʼʔ‚?]` $\rightarrow$ `'` and stripping post-consonantal violations (`*C'`).
* [ ] **T1.3**: Implement `resolve_vowel_hiatus(text: str) -> str` in `tones.py` inserting `'` between adjacent vowels across syllables.
* [ ] **T1.4**: Update `respell_consonants(text: str) -> str` to enforce the strict priority rewrite order (stops, sonorants, sibilants, affricates).
* [ ] **T1.5**: Update `convert_orthography` in `orthography.py` to propagate Standard 2 contextual sibilant pre-aspiration by default.

#### Distinct Unit Tests (`digohwelisgi/cherokee/orthography/tests/test_tones_normalization.py`)
* `test_sibilant_word_initial_vs_medial()`:
  * `"sakwu sookwili"` $\rightarrow$ `"sakwhu sokwhili"` (assert zero leading `hs-`).
  * `"skwaalheehsti"` $\rightarrow$ `"skwalhehsti"`, `"tsihskwa"` $\rightarrow$ `"tsihskwa"`.
* `test_glottal_stop_variants()`:
  * Assert `'`, `’`, `ʼ`, `ʔ`, `‚`, `?` all map to `'`.
* `test_vowel_hiatus_insertion()`:
  * `"inakei"` $\rightarrow$ `"inake'i"`.
* `test_consonant_rewrite_cascades()`:
  * Verify `d` becomes `t` and does NOT cascade to `th`.
  * Verify `g` becomes `k` and does NOT cascade to `kh`.
  * Verify `hn, hl, hw, hy` become `nh, lh, wh, yh`.
  * Verify `sl` before vowels becomes `slh`.

#### Integration Gate 1
* Run `pytest digohwelisgi/cherokee/orthography/ tests/` ensuring all existing 390+ unit tests and new normalization tests pass with 0 failures and 0 Pyright type errors.

---

### Phase 2: Moraic Tone Target Generator & Dual-Stream Sequence Reconciliation

#### Implementation Tasks
* [ ] **T2.1**: Implement `to_moraic_tone_tokens(raw_phonetic: str) -> Tuple[str, str]` in `digohwelisgi/cherokee/orthography/tones.py`:
  * Parses phonetic Cherokee with tones (e.g. `21`, `32`, `22`, `:`) and consonants.
  * Emits `cv_text`: clean segmental stream with single phonemic vowels (`a, e, i, o, u, v`), zero colons, and zero numeric tones.
  * Emits `tone_tokens`: space-separated sequence of `C`, moraic tone tokens (`1_1` to `4_2`), and word delimiters `|`.
  * Correctly omits initial `C` when a word begins with a vowel.
* [ ] **T2.2**: Implement `reconcile_dual_stream(cv_text: str, tone_tokens: str, mode: str = "dp") -> str`:
  * Mode `"fast"`: Deterministic regex/slot matching projecting mora count and tone numbers onto CV vowel slots.
  * Mode `"dp"`: Needleman-Wunsch sequence alignment using `C` and `|` boundary anchors to handle local insertions/deletions.
* [ ] **T2.3**: Implement `moraic_tokens_to_pitch_contour(tone_tokens: str) -> List[Tuple[float, float]]` utility for downstream Praat and visual inspection.

#### Distinct Unit Tests (`digohwelisgi/cherokee/orthography/tests/test_moraic_generator.py`)
* `test_short_vowel_mora_mapping()`:
  * Assert short low $\rightarrow$ `2_1`, short high $\rightarrow$ `3_1`.
* `test_long_vowel_and_contour_mora_mapping()`:
  * Level low `22` $\rightarrow$ `2_1 2_2`.
  * Level high `33` $\rightarrow$ `3_1 3_2`.
  * Low-fall `21` $\rightarrow$ `2_1 1_2`.
  * Rising `23` $\rightarrow$ `2_1 3_2`.
  * Falling `32` $\rightarrow$ `3_1 2_2`.
  * Superhigh `34` $\rightarrow$ `3_1 4_2`.
* `test_onset_separator_rules()`:
  * Words with consonant cluster: `skwa` $\rightarrow$ `C` followed by vowel moras.
  * Words starting with vowel: `ama` $\rightarrow$ no leading `C` (`2_1 C 2_1`).
  * Glottal stop onsets: `'a` $\rightarrow$ `C` followed by vowel moras.
* `test_reconciliation_roundtrip()`:
  * Feed 50 curated Feeling dictionary citations through `to_moraic_tone_tokens` $\rightarrow$ `reconcile_dual_stream` $\rightarrow$ assert 100% exact match against input.

#### Integration Gate 2
* Run an offline verification script processing all 1,482 sentences in `data/training/processed/sentence_audio.csv`:
  * Assert 0 exceptions raised.
  * Assert 100% successful round-trip reconstruction between `cv_text` + `tone_tokens` and source Feeling phonetics.

---

### Phase 3: Dual-Stream Dataset Preparation & Vocabulary Serialization

#### Lightweight Label Sync Strategy (No Audio Re-packaging)
* **Static Remote Audio**: The 462 MB dataset archive (`sentence_audio` and `split_audio`) is already hosted on Hugging Face Hub (`charliemcvicker/cherokee-asr-data`) and remains unchanged. We strictly avoid re-packaging or re-uploading audio files.
* **Fast Metadata & Label Updates**: Only the CSV split files (`chr-wav2vec2-*.csv`, `bible-wav2vec2-*.csv`) and vocabulary manifests (`vocab.json`, `vocab_tone.json`) are updated.
* **Git-Tracked Splits**: The regenerated CSVs and vocabulary files are committed directly into the Git repository branch (`feature/dual-head-tone-asr`).
* **Safe Remote Fetching**: `scripts/fetch_training_data.py` is updated with a `--preserve-local-csvs` flag (or post-fetch `git checkout` step) so remote extraction of `cherokee_asr_data.tar.gz` unpacks the audio without overwriting updated local CSV labels.

#### Implementation Tasks
* [ ] **T3.1**: Update `digohwelisgi/cherokee/training/prepare_csv.py`:
  * Ingest `sentence_audio.csv`, run Standard 2 normalization, and generate `cv_text` and `tone_tokens` columns.
  * Regenerate split CSVs: `chr-wav2vec2-train.csv`, `chr-wav2vec2-valid.csv`, `chr-wav2vec2-test.csv`, and `split_audio_syl_target.csv`.
* [ ] **T3.2**: Update Bible split preparation:
  * Ensure Bible CSVs (`bible-wav2vec2-train.csv`, etc.) populate `cv_text` with Standard 2 normalization and set `tone_tokens=""` (or masked sentinel).
* [ ] **T3.3**: Build and serialize dual tokenizers:
  * Segmental `vocab.json`: 18 tokens (`[PAD]`, `[UNK]`, `|`, `'`, `a`, `e`, `i`, `o`, `u`, `v`, `h`, `k`, `l`, `m`, `n`, `s`, `t`, `w`, `y`).
  * Moraic Tone `vocab_tone.json`: 16 tokens (`[PAD]`, `[UNK]`, `|`, `C`, `1_1`, `1_2`, `2_1`, `2_2`, `3_1`, `3_2`, `4_1`, `4_2`).
* [ ] **T3.4**: Update dataset loading in `train.py`:
  * Tokenize `cv_text` to `labels` using `vocab.json`.
  * Tokenize `tone_tokens` to `tone_labels` using `vocab_tone.json`.
  * For samples with empty tone tokens (Bible), set `tone_labels = [-100]`.
* [ ] **T3.5**: Update `scripts/fetch_training_data.py`:
  * Add `--preserve-local-csvs` flag preventing archive extraction from clobbering updated local CSV splits.
  * Add lightweight push helper `scripts/upload_labels_to_hf.py` to optionally mirror updated CSVs and vocabularies to HF Hub in seconds without touching the audio tarball.

#### Distinct Unit Tests (`digohwelisgi/cherokee/training/tests/test_dual_dataset_pipeline.py`)
* `test_vocab_tokenizers_bidirectional()`:
  * Encode and decode arbitrary CV sentences and Tone token sequences; assert exact recovery.
* `test_bible_sample_masking()`:
  * Verify that Bible rows yield `tone_labels = [-100]`.
* `test_dataset_split_sibilant_integrity()`:
  * Scan generated CSVs; assert 0 words start with `hs-` across all splits.

#### Integration Gate 3
* **Dataset Audit Script**: Run automated verification across all generated CSV files on disk:
  * Confirm 0 colons `:` in `cv_text`.
  * Confirm 0 sentence-medial words start with `hs-`.
  * Confirm 100% of non-empty `tone_tokens` contain strictly valid tokens from `vocab_tone.json`.

---

### Phase 4: Model Architecture (`Wav2Vec2ForDualCTC`) & Masked Collator

#### Implementation Tasks
* [ ] **T4.1**: Implement `Wav2Vec2ForDualCTC(Wav2Vec2PreTrainedModel)` in `digohwelisgi/cherokee/models/dual_head.py`:
  * Initialize shared feature extractor and transformer trunk (Layers 1–10).
  * Branch A: Transformer Layers 11A–12A + `lm_head = nn.Linear(hidden_size, cv_vocab_size)`.
  * Branch B: Cloned Transformer Layers 11B–12B + `tone_lm_head = nn.Linear(hidden_size, tone_vocab_size)`.
  * Forward pass:
    * Compute shared representations $H_{10}$.
    * Compute logits $Z_{\text{cv}} = \text{Head}_A(H_{10})$ and $Z_{\text{tone}} = \text{Head}_B(H_{10})$.
    * Compute CTC loss $\mathcal{L}_{\text{cv}}$ on all samples.
    * Compute masked CTC loss $\mathcal{L}_{\text{tone}}$ strictly on rows where `(tone_labels != -100).any()`.
    * Total loss: $\mathcal{L} = \mathcal{L}_{\text{cv}} + \lambda \mathcal{L}_{\text{tone}}$.
* [ ] **T4.2**: Implement `DualDataCollatorCTCWithPadding`:
  * Dynamically pads audio features, `labels` (padded with `-100`), and `tone_labels` (padded with `-100`).
* [ ] **T4.3**: Integrate `AudioAugmenter` in the collator (pitch shift $\pm 1.5$ semitones, time stretch $0.9$–$1.1$, gaussian noise).

#### Distinct Unit Tests (`digohwelisgi/cherokee/models/tests/test_dual_head_model.py`)
* `test_dual_head_forward_backward_synthetic()`:
  * Construct synthetic batch of 4 samples (2 with tone labels, 2 masked with `-100`).
  * Run forward pass; assert valid loss scalar.
  * Run backward pass; assert non-zero gradients on Trunk, Branch A, and Branch B.
  * Verify that masked samples contribute 0 loss to Branch B.
* `test_dual_collator_padding_and_shapes()`:
  * Pass variable length input sequences; assert tensor shapes and `-100` padding alignment.

#### Integration Gate 4
* Execute a standalone script instantiating `Wav2Vec2ForDualCTC` from the local `prebible-toneless-asr` checkpoint:
  * Run 5 training steps with AdamW optimizer on a dummy 4-sample batch.
  * Verify forward/backward execution, loss decay, and memory footprint on device (MPS/CUDA).

---

### Phase 5: Multi-Task Training Pipeline & Evaluation Harness

#### Implementation Tasks
* [ ] **T5.1**: Implement Interleaved Balanced DataLoader in `train.py`:
  * Alternates 50% Bible batches (CV-only) and 50% dictionary batches (CV + Tone) to prevent tone gradient starvation.
* [ ] **T5.2**: Implement multi-task metric computation:
  * Branch A: `wer` and `cer` on greedy decoded CV emissions.
  * Branch B: Tone Error Rate (`TER` / `tone_cer`) computed strictly on tone-annotated evaluation rows (`valid_orig`).
  * Reconciled metrics: `reconciled_cer` evaluating merged segmental + tone output.
* [ ] **T5.3**: Configure Trainer:
  * Set `metric_for_best_model="tone_cer"`, `greater_is_better=False`.
  * Support checkpoint saving with dual tokenizer artifacts.
* [ ] **T5.4**: Update `evaluate_local_checkpoints.py` to evaluate dual-head checkpoints across clean and noisy test sets.

#### Distinct Unit Tests (`digohwelisgi/cherokee/training/tests/test_metrics_and_eval.py`)
* `test_compute_dual_metrics()`:
  * Feed mock logits and targets; assert accurate computation of CV CER, Tone TER, and Reconciled CER.
* `test_interleaved_sampler()`:
  * Verify 50/50 batch distribution between datasets of unequal lengths.

#### Integration Gate 5
* **50-Step End-to-End Dry Run**:
  * Execute `train.py` with `--max_train_samples 20 --max_eval_samples 10 --max_steps 50`.
  * Assert TensorBoard logs both `loss_cv` and `loss_tone`.
  * Verify checkpoint saving produces a valid directory loadable by Hugging Face / PyTorch.

---

### Phase 6: Core ASR Model Integration & Benchmark Evaluation

#### Implementation Tasks
* [ ] **T6.1**: Update `ModelOutput` currency in `digohwelisgi/core/models/models.py`:
  * Support optional `lpz_tone` and `tone_tokens` decoding alongside base `lpz`.
  * Add `.decode_reconciled()` method returning fully tone-annotated Cherokee phonetics.
* [ ] **T6.2**: Update `CherokeeASRModel` in `digohwelisgi/cherokee/models/loader.py`:
  * Detect dual-head architecture; run joint inference on audio chunks; cache dual emissions.
* [ ] **T6.3**: Update `syllabary_enrichment` and `dialogue` pipelines:
  * Wire dual-head reconciled emissions into syllable reconciliation and Praat TextGrid exporters.
* [ ] **T6.4**: Run comprehensive evaluation benchmark on `chr-wav2vec2-test.csv` (185 samples):
  * Measure Base CER, Tone TER, and Reconciled CER comparing Fast Slicing vs DP alignment.

#### Distinct Unit Tests (`digohwelisgi/cherokee/tests/test_dual_inference_e2e.py`)
* `test_model_output_reconciled_projection()`:
  * Verify `ModelOutput.decode_reconciled()` produces valid tone-annotated text.
* `test_textgrid_exporter_with_tones()`:
  * Verify exported Praat TextGrid contains a dedicated Moraic Tone tier alongside Syllabary and TTH tiers.

#### Integration Gate 6
* **Final Benchmark Verification**:
  * Run `evaluate_local_checkpoints.py` on the trained model.
  * Verify Reconciled CER $\le$ Base CER + 1.5% while achieving Tone TER $\le 8.0\%$ on the 185 test samples.

---

## 5. Verification & Quality Assurance Summary

| Phase | Milestone Name | Key Deliverables | Primary Unit Test Suite | Integration Verification Gate |
| :---: | :--- | :--- | :--- | :--- |
| **1** | Orthography & Spelling Engine | Standard 2 pre-aspiration, glottal unification, hiatus resolution, cascade order | `test_tones_normalization.py` | Full existing 390+ test suite passes with 0 Pyright errors |
| **2** | Moraic Generator & Reconciler | `to_moraic_tone_tokens`, `reconcile_dual_stream` | `test_moraic_generator.py` | 100% roundtrip reconstruction across 1,482 Feeling sentences |
| **3** | Dual Dataset & Tokenizers | Normalizer script, dual CSV splits, `vocab.json` & `vocab_tone.json` | `test_dual_dataset_pipeline.py` | Dataset audit: 0 colons, 0 initial `hs-`, 100% valid vocab IDs |
| **4** | Model Architecture & Collator | `Wav2Vec2ForDualCTC`, dual padding collator, masked loss | `test_dual_head_model.py` | Synthetic forward/backward gradient verification & memory check |
| **5** | Training Pipeline & Harness | Interleaved DataLoader, multi-task metrics, early stopping | `test_metrics_and_eval.py` | 50-step dry run with TensorBoard metric and checkpoint logging |
| **6** | Model Integration & Benchmark | `ModelOutput` dual projection, `CherokeeASRModel`, TextGrid tier | `test_dual_inference_e2e.py` | Full test-split benchmark: Tone TER $\le 8.0\%$, Reconciled CER verified |

---

## 6. Vast.ai Remote Training Roadmap & Pre-Rental Verification Gates

To ensure zero wasted spend and avoid debugging basic pipeline or shape errors on metered GPU hardware, this section maps out **everything that must be in place before renting a GPU instance**, how remote data fetching operates without re-uploading audio, and how remote execution is managed.

### 6.1 Everything Standing in Our Way Before GPU Rental

The following 6 sequential prerequisites must be complete and passing before launching an instance on Vast.ai:

```
┌────────────────────────────────────────────────────────┐
│ 1. Phase 1 & 2 Normalization & Moraic Engine (Local)  │
│    - Standard 2 /hs/ regex fix in tones.py             │
│    - Glottal & hiatus resolvers passing unit tests     │
│    - to_moraic_tone_tokens 100% roundtrip on Feeling   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 2. Local CSV & Vocabulary Regeneration (Phase 3)       │
│    - Regenerated CSVs with cv_text & tone_tokens       │
│    - vocab.json (18 tokens) & vocab_tone.json (16 toks)│
│    - 0 colons, 0 leading hs- on disk verified          │
│    - Committed to Git branch feature/dual-head-tone-asr│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 3. Dual-Head Architecture & Masked Collator (Phase 4)  │
│    - Wav2Vec2ForDualCTC implemented in models/         │
│    - DualDataCollatorCTCWithPadding with live masking  │
│    - Synthetic batch backward pass verifies gradients  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 4. Training Script Wiring (Phase 5)                    │
│    - train.py updated with --vocab-tone-path & dual loss│
│    - compute_metrics evaluates wer, cer, and tone_cer  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 5. The Pre-Rental Gate: 5-Step Local Smoke Test (MPS)  │
│    - Run 5 steps on Mac with tiny sample batch         │
│    - Asserts 0 CUDA/MPS dimension bugs, valid loss,   │
│      optimizer step, and checkpoint write-out          │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 6. Fast Remote Data Strategy & Docker Environment      │
│    - fetch_training_data.py --preserve-local-csvs ready│
│    - Remote instance setup script / Docker image ready │
│    - Hugging Face write token & Vast.ai balance funded │
└────────────────────────────────────────────────────────┘
```

---

### 6.2 The Pre-Rental Gate: 5-Step Local Smoke Test (MPS / Mac)

Before spending any cloud GPU funds, run this local verification command from the repository root on Apple Silicon (MPS) or CPU:

```bash
conda activate cherokee-asr
python3 -m digohwelisgi.cherokee.training.train \
  --train-orig-csv data/training/processed/chr-wav2vec2-train.csv \
  --train-bible-csv data/training/processed/bible-wav2vec2-train.csv \
  --valid-orig-csv data/training/processed/chr-wav2vec2-valid.csv \
  --valid-bible-csv data/training/processed/bible-wav2vec2-valid.csv \
  --audio-dir data/training/processed/sentence_audio \
  --output-dir output_smoke_test \
  --max-train-samples 16 \
  --max-eval-samples 8 \
  --max-steps 5 \
  --tone-loss-weight 1.0
```

#### Pass Criteria for Smoke Test:
1. Both `vocab.json` and `vocab_tone.json` load without key errors.
2. `DataCollatorCTCWithPadding` successfully collates mixed Bible and dictionary batches, padding Bible rows with `-100`.
3. `Wav2Vec2ForDualCTC` forward pass computes both $\mathcal{L}_{\text{cv}}$ and $\mathcal{L}_{\text{tone}}$, with scalar total loss.
4. Backward pass updates model weights with zero gradient shape mismatch errors.
5. Checkpoint write-out saves model weights, dual tokenizer configs, and evaluation metrics cleanly to `output_smoke_test/`.

---

### 6.3 Remote Instance Selection on Vast.ai

Once the Smoke Test passes, select and rent an instance following the guidelines from `doc-2`:

1. **Hardware Requirements**:
   * **GPU**: Minimum **24 GB VRAM** (RTX 3090, RTX 4090, A5000, or A6000) to support concurrent dual-head CTC forward-backward graphs and batch size $\ge 8$.
   * **Disk Space**: Allocate **at least 50 GB** (to store 462 MB audio, unpacked data, Hugging Face pretrained transformer cache, and multiple ~4.5 GB training checkpoints).
2. **Docker Image**:
   * Set Docker Image template to our dedicated prebaked image: **`charliemcvicker/asr:latest`** (built from the repository's root `Dockerfile`).
   * *Why this is essential*: It contains CUDA 12.1, all Python dependencies, custom `ctc-segmentation`, pre-copies the codebase, and has the base `facebook/wav2vec2-large-xlsr-53` model pre-downloaded in its cache. Remote instances start instantly with zero model download wait time.
   * *Local Build & Push Command* (run whenever architecture/dependencies change):
     ```bash
     docker build -t charliemcvicker/asr:latest .
     docker push charliemcvicker/asr:latest
     ```
3. **Connectivity**:
   * Select **SSH** launch mode. Note down `[IP]` and `[PORT]`.

---

### 6.4 Remote Startup & Data Fetching (Zero Audio Re-uploading)

Inside the remote Vast.ai SSH session:

```bash
# 1. Clone repository branch with updated code and CSV labels
cd /workspace
git clone -b feature/dual-head-tone-asr https://github.com/CharlieMcVicker/workshop-transcription.git repo
cd /workspace/repo

# 2. Export Hugging Face token for data fetching and backup pushes
export HF_TOKEN="hf_your_token_here"
export PYTHONPATH="/workspace/repo:${PYTHONPATH}"

# 3. Fetch 462 MB remote audio without clobbering updated local CSV labels
python3 scripts/fetch_training_data.py --preserve-local-csvs
```

*Verification Step*: Verify dataset files exist and local CSVs were not clobbering:
```bash
head -n 2 data/training/processed/chr-wav2vec2-train.csv
# Must show 'cv_text' and 'tone_tokens' columns, zero colons, and zero initial 'hs-'
```

---

### 6.5 Launching Full Training with Automated Hub Backups

To protect against instance preemption or spot bid eviction, run training under `tmux` with automated Hugging Face Hub checkpoint backups:

```bash
tmux new -s training

python3 -m digohwelisgi.cherokee.training.train \
  --train-orig-csv data/training/processed/chr-wav2vec2-train.csv \
  --train-bible-csv data/training/processed/bible-wav2vec2-train.csv \
  --valid-orig-csv data/training/processed/chr-wav2vec2-valid.csv \
  --valid-bible-csv data/training/processed/bible-wav2vec2-valid.csv \
  --test-orig-csv data/training/processed/chr-wav2vec2-test.csv \
  --test-bible-csv data/training/processed/bible-wav2vec2-test.csv \
  --audio-dir data/training/processed/sentence_audio \
  --output-dir /workspace/output_dual_head \
  --tone-loss-weight 1.0 \
  --push-to-hub \
  --hub-model-id "charliemcvicker/cherokee-wav2vec2-dual-head" \
  --hub-token "$HF_TOKEN" \
  --epochs 30
```

---

### 6.6 Retrieving Checkpoints & Results (Fail-safe)

If the instance finishes or is paused, pull checkpoints locally using resuming `rsync`:

```bash
rsync -avzP -e 'ssh -p [PORT]' root@[IP]:/workspace/output_dual_head/ ./output_dual_head/
```
*(Checkpoints pushed to Hugging Face Hub serve as an immediate zero-loss backup if the instance is unexpectedly terminated).*
