import torch
from collections import OrderedDict
import os
import soundfile as sf
import numpy as np


def load_encoder_checkpoint(checkpoint_path: str, output_dim: int) -> OrderedDict:
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if "pytorch-lightning_version" in state_dict.keys():
        new_state_dict = OrderedDict(
            {
                k.replace("model.encoder.", ""): v
                for k, v in state_dict["state_dict"].items()
                if "model.encoder." in k
            }
        )
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "encoder." in k:
                new_state_dict[k.replace("encoder.", "")] = v

    new_state_dict["fc.weight"] = torch.zeros(output_dim, 512)
    new_state_dict["fc.bias"] = torch.zeros(output_dim)
    return new_state_dict


def load_finetuner_checkpoint(checkpoint_path: str) -> OrderedDict:
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if "pytorch-lightning_version" in state_dict.keys():
        state_dict = OrderedDict(
            {
                k.replace("model.", ""): v
                for k, v in state_dict["state_dict"].items()
                if "model." in k
            }
        )
    return state_dict

def clmr_load_wavs(wav_paths, target_sr=22050, target_len=661500):
        """
        wav_paths: a list of paths to wav files
        Returns a torch.FloatTensor of shape [batch_size, 1, target_len]
        target length for 30s at 22.05 kHz is 661500
        """
       
        batch_tensors = []
        for wav_path in wav_paths:
            if not os.path.exists(wav_path):
                print(f"Skipping {wav_path} because it does not exist")
                waveform_np = np.zeros(target_len, dtype=np.float32)
            else:
                waveform_np, sr = sf.read(wav_path, dtype='float32')
                if waveform_np.ndim > 1:
                    waveform_np = np.mean(waveform_np, axis=-1)
                if sr != target_sr:
                    # Assume prior resampling via ffmpeg; if mismatch, pad/trim only.
                    pass
                if waveform_np.shape[0] >= target_len:
                    waveform_np = waveform_np[:target_len]
                else:
                    pad_width = target_len - waveform_np.shape[0]
                    waveform_np = np.pad(waveform_np, (0, pad_width), mode='constant')

            tensor = torch.from_numpy(waveform_np).float().unsqueeze(0)  # [1, L]
            batch_tensors.append(tensor)

        if len(batch_tensors) == 0:
            return torch.zeros(0, 1, target_len, dtype=torch.float32)

        batch = torch.stack(batch_tensors, dim=0)  # [B, 1, L]
        return batch