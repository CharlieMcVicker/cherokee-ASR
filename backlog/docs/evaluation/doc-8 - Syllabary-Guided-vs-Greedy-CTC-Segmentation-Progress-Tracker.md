---
id: doc-8
title: Syllabary-Guided vs Greedy CTC Segmentation Progress Tracker
type: guide
created_date: '2026-09-16 17:09'
updated_date: '2026-09-16 17:14'
---
# Syllabary-Guided vs Greedy CTC Segmentation Progress Tracker

Tracking comparative accuracy benchmarks, 3-way error breakdowns, and phonotactic alignment progression across the Cherokee Syllabary dataset (`split_audio_syl_target.csv`, 1,389 samples) evaluating Greedy ASR inference vs. Syllabary-Guided CTC Segmentation (Relative Contrastive Acoustic Gating, zero tuned hyperparameters).

---

## 1. Latest Evaluation Benchmark (Post-Deaffrication & Laryngeal Reform)

```
========================================================================================
        EVALUATION RESULTS SUMMARY: GREEDY VS SYLLABARY-GUIDED CTC SEGMENTATION         
     (ctc-segmentation PR #7 Relative Contrastive Gating - Zero Tuned Hyperparams)      
========================================================================================
Split    | Condition | Greedy CER | Guided CER | Δ CER    | Greedy WER | Guided WER | Δ WER   
----------------------------------------------------------------------------------------
test     | Clean  |      2.02% |      2.32% |   +0.30% |     14.29% |     14.92% |   +0.63%
test     | Noisy  |      4.25% |      2.34% |   -1.91% |     27.10% |     16.18% |  -10.92% 🏆
----------------------------------------------------------------------------------------
valid    | Clean  |      2.16% |      2.28% |   +0.12% |     15.26% |     14.52% |   -0.74% 🏆
valid    | Noisy  |      5.50% |      2.57% |   -2.93% |     32.35% |     18.38% |  -13.97% 🏆
----------------------------------------------------------------------------------------
train    | Clean  |      0.04% |      1.89% |   +1.85% |      0.23% |     12.21% |  +11.98%
train    | Noisy  |      2.11% |      1.77% |   -0.34% |     12.32% |     13.15% |   +0.83% 🏆
----------------------------------------------------------------------------------------
all      | Clean  |      0.42% |      1.97% |   +1.55% |      3.05% |     12.69% |   +9.64%
all      | Noisy  |      2.64% |      1.90% |   -0.74% |     15.72% |     13.96% |   -1.76% 🏆
----------------------------------------------------------------------------------------
========================================================================================
```

---

## 2. 3-Way Comparative Benchmark Dynamics (TASK-341)

Across 1,389 evaluation samples, Syllabary-Guided CTC segmentation provides robust resistance to acoustic noise and out-of-domain degradation compared to Greedy ASR:

| Evaluation Split | Condition | Exact Match (Both) | Exact Match (Guided Only) | Exact Match (Greedy Only) | Guided CER Wins | Greedy CER Wins | Equal CER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **test** (130) | Clean | 58 (44.6%) | 15 (11.5%) | 27 (20.8%) | 25 (19.2%) | 32 (24.6%) | 73 (56.2%) |
| **test** (130) | Noisy (18dB) | 34 (26.2%) | **31 (23.8%)** | 16 (12.3%) | **59 (45.4%)** | 19 (14.6%) | 52 (40.0%) |
| **valid** (141) | Clean | 58 (41.1%) | 21 (14.9%) | 21 (14.9%) | 31 (22.0%) | 34 (24.1%) | 76 (53.9%) |
| **valid** (141) | Noisy (18dB) | 28 (19.9%) | **36 (25.5%)** | 10 (7.1%) | **71 (50.4%)** | 19 (13.5%) | 51 (36.2%) |
| **train** (1118) | Clean | 689 (61.6%) | 3 (0.3%) | 422 (37.7%) | 5 (0.4%) | 423 (37.8%) | 690 (61.7%) |
| **train** (1118) | Noisy (18dB) | 500 (44.7%) | **162 (14.5%)** | 257 (23.0%) | **251 (22.5%)** | 295 (26.4%) | 572 (51.2%) |
| **all** (1389) | Clean | 805 (58.0%) | 39 (2.8%) | 470 (33.8%) | 61 (4.4%) | 489 (35.2%) | 839 (60.4%) |
| **all** (1389) | Noisy (18dB) | 562 (40.5%) | **229 (16.5%)** | 283 (20.4%) | **381 (27.4%)** | 333 (24.0%) | 675 (48.6%) |

---

## 3. Iteration History & Milestones

| Stage | Clean Test CER | Clean Valid CER | Clean All CER | Noisy Test CER | Noisy Valid CER | Noisy All CER | Key Structural Interventions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0. Initial Baseline** | 7.81% | 6.58% | 7.16% | 7.85% | 7.20% | 7.21% | Baseline heuristic penalties. Terminal vowels deleted before pad; spurious `h` intrusions everywhere. |
| **1. *HH Constraint & Stride 4** (`TASK-337`, `TASK-340`) | 3.24% | 3.00% | 2.99% | 3.44% | 3.40% | 3.00% | Blocked intrusive `h` before `hs`; licensed pre-consonantal `th`, `kh`, `lh`, `nh`, `wh`, `yh`; onset-only syncope. Missing `h` dropped 82%. |
| **2. Lateral Deaffrication** (`TASK-331`) | **2.32%** | **2.28%** | **1.97%** | **2.34%** | **2.57%** | **1.90%** | Licensed optional stop `t` syncope deletion in `tl` and `tlh` clusters. Lateral errors dropped 89.5% (497 -> 52). Guided decisively beats Greedy on noisy audio. |
| **3. Full Comparative Categorization** (`TASK-341`) | **2.32%** | **2.28%** | **1.97%** | **2.34%** | **2.57%** | **1.90%** | Comprehensive 3-way error categorization across all 1,389 clean & noisy samples. Pinpointed laryngeal stride mutex and initial sibilant normalization as primary remaining targets. |

---

## 4. Revised Error Taxonomy (TASK-341)

| Error Category | Clean Events (All) | Clean Events (Test+Valid) | Noisy Events (All) | Noisy Events (Test+Valid) | Primary Linguistic & Structural Cause | Status / Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`LARYNGEAL_STRIDE_VOWEL_INTRUSION`** | 210 | 40 | 93 | 20 | Syllable vowel retained alongside intrusive `h` in trellis (e.g. `akhahth` vs `akhth`, `kalhih` vs `kalh`, `uwhvh` vs `uwh`). | 🎯 Priority #1: Add trellis mutex between vowel retention and syncopated aspiration. |
| **`VOWEL_UNDER_SYNCOPE_MEDIAL`** | 135 | 21 | 122 | 19 | Speaker syncopated medial vowel, but aligner emitted base template vowel (e.g. `tsunahsti` vs `tsunhsti`, `amalhvhti` vs `amalhti`). | 🎯 Priority #4: Tune syncope gating before voiceless stops. |
| **`INITIAL_SIBILANT_HS_VS_S`** | 93 | 12 | 93 | 12 | Template maps initial Syllabary Ꮝ to `hs-`, but GT transcript uses `s-` (e.g. `hstaya` vs `staya`, `hsinvta` vs `sinvta`). | 🎯 Priority #2: Canonicalize word-initial `Ꮝ` normalizer to allow `s-`/`hs-` equivalence. |
| **`SONORANT_PREASPIRATION`** | 87 | 33 | 194 | 56 | Pre-aspiration / devoicing mismatch on sonorants `nh, wh, yh, lh` (e.g. `uwhtohti` vs `uwtohti`, `tuninhti` vs `tuninti`). | 🎯 Priority #3: License dynamic pre-aspiration across sonorants (`n, w, y, l`). |
| **`VOWEL_OVER_SYNCOPE_MEDIAL`** | 52 | 16 | 74 | 28 | Speaker pronounced medial vowel, but aligner syncopated (e.g. `totalv` -> `ttalv`, `analhihv` -> `analhv`). | In progress. |
| **`GLOTTAL_STOP_HIATUS`** | 48 | 12 | 53 | 15 | Presence/absence of hiatus glottal stop apostrophe `'` in GT vs Syllabary (e.g. `kalvla'ti` vs `kalvlati`). | In progress (orthographic variant). |
| **`STOP_AFFRICATE_ASPIRATION`** | 42 | 11 | 114 | 26 | Voicing/aspiration mismatch on stops & affricates (`th/t`, `kh/k`, `tsh/ts`). | In progress. |
| **`VOWEL_OVER_SYNCOPE_FINAL`** | 25 | 9 | 22 | 12 | Final vowel dropped by aligner before boundary. | Stable. |
| **`RESIDUAL_SUBSTITUTION`** | 15 | 11 | 29 | 11 | Single-character phoneme substitutions. | Low residual count. |
| **`VOWEL_UNDER_SYNCOPE_INITIAL`** | 8 | 1 | 8 | 1 | Initial onset vowel retained from template when speaker syncopated. | Low residual count. |
| **`VOWEL_UNDER_SYNCOPE_FINAL`** | 4 | 0 | 4 | 0 | Word-final vowel retained from template when speaker syncopated. | Low residual count. |
| **`MEDIAL_SIBILANT_PREASPIRATION`** | 0 | 0 | 3 | 2 | Medial preconsonantal sibilant aspiration `hs` vs `s`. | Stable. |
| **`ISOLATED_LARYNGEAL_H`** | 1 | 0 | 1 | 0 | Intervocalic / isolated stray `h`. | Stable. |

---

## 5. Priority Next Interventions

1. **Laryngeal Stride Mutex Constraint (`LARYNGEAL_STRIDE_VOWEL_INTRUSION`, 210 clean events)**:
   - In `prepare_cherokee_text`, enforce a strict mutual exclusion constraint during syllabic trellis expansion: when a syllable contains both an intrusive `h` and an optional syncope vowel (e.g., `ka` -> `k`, `kh`, `kah`), it must branch into either `[Consonant + h + NextConsonant]` (aspirated syncope) or `[Consonant + Vowel]` (full syllable), preventing dual emission of `h` and `V` (`k-h-a-h-th`).
2. **Word-Initial Sibilant Normalizer (`INITIAL_SIBILANT_HS_VS_S`, 93 events)**:
   - Standardize Syllabary word-initial `Ꮝ` handling in phonetic normalizers: allow flexible `s-` / `hs-` branch in trellis alignment or normalize evaluation references.
3. **Dynamic Sonorant Pre-Aspiration Licensing (`SONORANT_PREASPIRATION`, 87 clean / 194 noisy events)**:
   - Broaden trellis phonotactics for pre-vocalic and intervocalic sonorants (`n, w, y, l`) to license optional `nh, wh, yh, lh` paths dynamically.
4. **Medial Vocalic Syncope Weighting (`VOWEL_UNDER_SYNCOPE_MEDIAL`, 135 clean events)**:
   - Calibrate relative contrastive gating or phonotactic penalty for medial short vowels (`a, i, v`) preceding voiceless consonants.

---

## 6. Evaluation & Analysis Execution

To reproduce the benchmark and full comparative error analysis:
```bash
# 1. Rescore Greedy vs Syllabary-Guided across all 1,389 samples (~3s via LPZ cache):
python scripts/rescore_syllabary_dataset.py

# 2. Run 3-way comparative error categorization and generate markdown breakdown:
python scripts/analyze_comparative_errors.py
```
