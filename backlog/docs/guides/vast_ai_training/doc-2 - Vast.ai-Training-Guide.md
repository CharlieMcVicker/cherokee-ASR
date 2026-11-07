---
id: doc-2
title: Vast.ai Training Guide
type: guide
created_date: '2026-06-30 23:22'
updated_date: '2026-11-07 15:10'
---
# Vast.ai Training Guide for Cherokee Wav2Vec2 ASR

This guide outlines the standard operating procedure for running Wav2Vec2 fine-tuning on a remote GPU instance using **vast.ai** and our dedicated Docker image (`charliemcvicker/asr:latest`).

---

## 1. Our Custom Docker Image (`charliemcvicker/asr:latest`) 🐳

We train using our project's custom Docker image, built from the root [Dockerfile](file:///Users/julietmcvicker/code/workshop-transcription/Dockerfile). 

### What is Prebaked in the Image:
1. **Base Runtime**: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` with CUDA 12.1.
2. **Audio & CTC Libraries**: Custom `ctc-segmentation`, `torchaudio`, `audiomentations`, `soundfile`, `librosa`, `jiwer`, `evaluate`, `transformers`.
3. **Pre-cached Foundation Model**: `facebook/wav2vec2-large-xlsr-53` is pre-downloaded into the image cache during build (`RUN python3 -c "from transformers import Wav2Vec2ForCTC; Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-xlsr-53')"`). This eliminates remote cold-start download delays on Vast.ai.
4. **Codebase Packages**: `digohwelisgi` package and `scripts/` are baked into `/workspace` with `PYTHONPATH="/workspace:${PYTHONPATH}"`.

### Building & Pushing the Image:
When codebase architecture, model definitions (`Wav2Vec2ForDualCTC`), or dependencies change, rebuild and push the image locally:
```bash
# 1. Build the image locally
docker build -t charliemcvicker/asr:latest .

# 2. Push to Docker Hub
docker push charliemcvicker/asr:latest
```

---

## 2. Vast.ai Template Configuration & Instance Selection

1. **Sign in to Vast.ai**: Go to [Vast.ai Console](https://vast.ai/).
2. **Template Configuration**:
   * **Docker Image**: `charliemcvicker/asr:latest`
   * **Disk Allocation**: **At least 50 GB** (accommodates remote audio downloads, Hugging Face transformers cache, and intermediate ~4.5 GB model checkpoints).
   * **Launch Mode**: Select **Use SSH** (recommended).
3. **GPU Hardware Recommendation**:
   * Choose an instance with **at least 24 GB of VRAM** (e.g., RTX 3090, RTX 4090, RTX A5000, or A6000) to ensure large batch sizes ($\ge 8$), stability during dual-head CTC loss calculation, and fast execution.
4. **Launch the Instance** and note down connection details:
   * **IP Address**: `[IP]`
   * **Port**: `[PORT]`

---

## 3. Remote Data Fetching (Zero Audio Re-uploading) 🚀

The raw audio files (462 MB) are hosted on Hugging Face Hub (`charliemcvicker/cherokee-asr-data`). You do **not** need to re-package or re-upload audio when labels change.

Inside your remote Vast.ai SSH session:

```bash
# 1. Connect via SSH
ssh -p [PORT] root@[IP]

# 2. Navigate to workspace
cd /workspace

# 3. Pull latest git branch containing updated split CSVs and tokenizers
git clone -b feature/dual-head-tone-asr https://github.com/CharlieMcVicker/workshop-transcription.git repo
cd /workspace/repo

# 4. Set environment
export HF_TOKEN="hf_your_hugging_face_token_here"
export PYTHONPATH="/workspace/repo:${PYTHONPATH}"

# 5. Fetch 462 MB remote audio without clobbering updated local CSV splits
python3 scripts/fetch_training_data.py --preserve-local-csvs
```

The script downloads and unpacks:
- `data/training/processed/sentence_audio/` (1,482 files)
- `data/projects/cherokee_new_testament/split_audio/`

*(Note: `--preserve-local-csvs` ensures the newly generated git-tracked CSV labels on disk are not overwritten by the older CSVs in the remote audio archive).*

---

## 4. Remote Environment Sanity Check

Once connected, run quick verification checks inside the container:

1. **Verify GPU Status**:
   ```bash
   nvidia-smi
   ```
2. **Verify CUDA availability in PyTorch**:
   ```bash
   python3 -c "import torch; print('CUDA active:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
   ```
3. **Verify Pre-cached Foundation Model**:
   ```bash
   python3 -c "from transformers import Wav2Vec2ForCTC; m = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-xlsr-53'); print('Model loaded successfully!')"
   ```

---

## 5. Cost-Saving Pre-Flight / Smoke Test (10 Steps)

Before starting an overnight 30-epoch run, run a 10-step sanity test to verify tensor dimensions, forward/backward passes, dual loss calculation, and checkpoint writing:

```bash
python3 -m digohwelisgi.cherokee.training.train \
  --train-orig-csv data/training/processed/chr-wav2vec2-train.csv \
  --train-bible-csv data/training/processed/bible-wav2vec2-train.csv \
  --valid-orig-csv data/training/processed/chr-wav2vec2-valid.csv \
  --valid-bible-csv data/training/processed/bible-wav2vec2-valid.csv \
  --audio-dir data/training/processed/sentence_audio \
  --output-dir /workspace/output_smoke_test \
  --max-steps 10 \
  --max-train-samples 16 \
  --max-eval-samples 8
```

---

## 6. Full Training Run with Hugging Face Continuous Backups 🛡️

Run training under `tmux` so that disconnecting SSH does not interrupt training. Continuous checkpoint uploads to Hugging Face protect against instance spot-preemption:

```bash
# Start a detachable tmux session
tmux new -s training

# Run full training
python3 -m digohwelisgi.cherokee.training.train \
  --train-orig-csv data/training/processed/chr-wav2vec2-train.csv \
  --train-bible-csv data/training/processed/bible-wav2vec2-train.csv \
  --valid-orig-csv data/training/processed/chr-wav2vec2-valid.csv \
  --valid-bible-csv data/training/processed/bible-wav2vec2-valid.csv \
  --test-orig-csv data/training/processed/chr-wav2vec2-test.csv \
  --test-bible-csv data/training/processed/bible-wav2vec2-test.csv \
  --audio-dir data/training/processed/sentence_audio \
  --output-dir /workspace/output_w2v2 \
  --push-to-hub \
  --hub-model-id "charliemcvicker/cherokee-wav2vec2-dual-head" \
  --hub-token "$HF_TOKEN" \
  --epochs 30
```
*Detach from tmux*: Press `Ctrl+B`, then `D`.
*Reattach later*: Run `tmux attach -t training`.

---

## 7. Instance Lifecycle Management

* **Stopped Instance**: If your bid is outpriced, or you stop the instance manually, the machine is paused. **Your data inside `/workspace` is preserved**. You only pay a minimal disk fee (~$0.005/GB/month). You can restart it later and resume.
* **Destroyed Instance**: Clicking the trash icon deletes the container and disk permanently. Ensure your best checkpoint has pushed to Hugging Face or has been downloaded locally first.

---

## 8. Retrieving Checkpoints and Results Locally 📥

To download the checkpoints to your local machine (run **from your local terminal**, not inside SSH):

### Using `rsync` (Supports Resuming):
```bash
rsync -avzP -e 'ssh -p [PORT]' root@[IP]:/workspace/output_w2v2/ ./output_w2v2/
```
*(Checkpoints pushed to Hugging Face Hub also serve as an immediate zero-loss backup).*
