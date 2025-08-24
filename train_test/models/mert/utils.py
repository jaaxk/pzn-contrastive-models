import os
import soundfile as sf
import numpy as np

def mert_load_wavs(wav_paths):
    waveforms = []
    for wav_path in wav_paths:
        if not os.path.exists(wav_path):
            print(f"Skipping {wav_path} because it does not exist")
            waveform = np.zeros(24000*29)
        else:
            waveform, sr = sf.read(wav_path, dtype='float32')
            waveform = waveform.squeeze()
            waveforms.append(waveform)
    return waveforms