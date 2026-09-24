FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
  
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
  
# Install runtime system dependencies & audio utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
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

WORKDIR /workspace

# Pre-cache model checkpoint (optional: omit if using vast.ai volume storage)
RUN python3 -c "from transformers import Wav2Vec2ForCTC; Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-xlsr-53')"

COPY data/training /workspace/data/training
COPY data/projects/cherokee_new_testament /workspace/data/projects/cherokee_new_testament

COPY digohwelisgi /workspace/digohwelisgi
ENV PYTHONPATH="/workspace:${PYTHONPATH}"


