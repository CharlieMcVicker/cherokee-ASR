# Use a PyTorch CUDA development image as base
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies & audio utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    transformers==4.44.2 \
    datasets==2.21.0 \
    accelerate==0.34.2 \
    torchaudio \
    jiwer==3.0.4 \
    evaluate==0.4.2 \
    soundfile \
    librosa==0.10.2.post1


# Set default working directory
WORKDIR /workspace

# Pre-download the base XLS-R model checkpoint to cache it in the image
RUN python3 -c "from transformers import Wav2Vec2ForCTC; Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-xlsr-53')"

# Copy the package source and data directories directly into the container
COPY transcription /workspace/transcription
COPY data /workspace/data

# Ensure Python knows where to find the transcription package modules
ENV PYTHONPATH="/workspace:${PYTHONPATH}"




