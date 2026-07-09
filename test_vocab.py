import torch
import numpy as np
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("charliemcvicker/asr-cherokee")
vocab = processor.tokenizer.get_vocab()
print(vocab)
