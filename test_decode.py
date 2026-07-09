import torch
import numpy as np
import json
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("charliemcvicker/asr-cherokee")
pad_id = processor.tokenizer.pad_token_id
if pad_id is None:
    pad_id = processor.tokenizer.vocab.get("[PAD]", 0)

print(f"pad_id={pad_id}")

# Fake probs
pred_ids = [10, 10, pad_id, 11, pad_id, 12, 12, pad_id]
token_probs = [0.9, 0.8, 0.1, 0.95, 0.1, 0.7, 0.8, 0.1]

chars = []
char_probs = []
prev_id = -1
for i, token_id in enumerate(pred_ids):
    if token_id != pad_id and token_id != prev_id:
        token_str = processor.decode([token_id])
        if token_str:
            chars.append(token_str)
            char_probs.append(float(token_probs[i]))
    prev_id = token_id

words = []
current_word = ""
current_word_probs = []
for char, prob in zip(chars, char_probs):
    if char == " ":
        if current_word:
            words.append(
                {"word": current_word, "confidence": float(np.mean(current_word_probs))}
            )
            current_word = ""
            current_word_probs = []
    else:
        current_word += char
        current_word_probs.append(prob)
if current_word:
    words.append(
        {"word": current_word, "confidence": float(np.mean(current_word_probs))}
    )

out = {
    "chars": chars,
    "char_probs": char_probs,
    "words": words,
    "vocab": {k: v for k, v in list(processor.tokenizer.get_vocab().items())[:10]},
}

with open("test_vocab_out.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
