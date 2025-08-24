import requests
import os
import tqdm
import subprocess
import time
import soundfile as sf
import numpy as np
import torch

def download_and_preprocess_previews(preview_urls, track_ids, cluster_ids, output_dir, sample_rate):
    wav_paths = []
    cluster_ids_list = []
    track_ids_list = []
    for url, track_id, cluster_id in tqdm.tqdm(zip(preview_urls, track_ids, cluster_ids)):
        if not str(url).startswith('http'):
            print(f"Skipping {url} because it is not a valid URL")
            continue
        wav_path = os.path.join(output_dir, f'{track_id}.wav')
        if not os.path.exists(wav_path):
            retries = 0
            while not os.path.exists(os.path.join(output_dir, f"{track_id}.mp3")):
                response = requests.get(url)
                if response.status_code == 200:
                    with open(os.path.join(output_dir, f"{track_id}.mp3"), "wb") as f:
                        f.write(response.content)
                retries += 1
                if retries > 3:
                    print(f"Failed to download preview from {url}")
                    break
        
            #convert mp3 to wav, mono, 24khz
            cmd = [
                "ffmpeg",
                "-i", str(os.path.join(output_dir, f"{track_id}.mp3")),  # Input file
                "-ar", str(sample_rate),  # Sample rate
                "-ac", "1",  # Mono audio
                "-f", "wav",  # Output format
                "-y",  # Overwrite output file if it exists
                "-loglevel", "warning",  # Only show warnings/errors
                wav_path
            ] #  "-t", "15",  # Limit to first 15 seconds
            subprocess.run(cmd)
            #remove mp3 file
            if os.path.exists(os.path.join(output_dir, f"{track_id}.mp3")):
                os.remove(os.path.join(output_dir, f"{track_id}.mp3"))
        if os.path.exists(os.path.join(output_dir, f'{track_id}.wav')):
            wav_paths.append(wav_path)
            track_ids_list.append(track_id)
            cluster_ids_list.append(cluster_id)

    return wav_paths, cluster_ids_list, track_ids_list


def clmr_load_wavs(wav_paths, target_sr=22050, target_len=59049):
        """
        wav_paths: a list of paths to wav files
        Returns a torch.FloatTensor of shape [batch_size, 1, target_len]
        at 22.05 kHz, padded/truncated as needed.
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