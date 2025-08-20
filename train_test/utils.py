import requests
import os
import tqdm
import subprocess
import time

def download_and_preprocess_previews(preview_urls, track_ids, output_dir):
    wav_paths = []
    for url, track_id in tqdm.tqdm(zip(preview_urls, track_ids)):
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
                "-ar", "24000",  # Sample rate
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

    return wav_paths
