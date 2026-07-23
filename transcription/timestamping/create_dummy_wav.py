# -*- coding: utf-8 -*-
import os
from pydub.generators import Sine

dummy_wav = "data/dummy_test.wav"
os.makedirs("data", exist_ok=True)
audio = Sine(440).to_audio_segment(duration=5000)
audio.export(dummy_wav, format="wav")
print(f"Created {dummy_wav}")
