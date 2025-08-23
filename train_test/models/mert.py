from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torch
import torch.nn as nn
import torchaudio
import os
import soundfile as sf

class MERT(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, trust_remote_code=True)
        self.aggregator = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1).to(self.device)

    def load_wavs(self, wav_paths):
        """
        wav_paths: a list of paths to wav files
        """
        waveforms = []
        for wav_path in wav_paths:
            if not os.path.exists(wav_path):
                print(f"Skipping {wav_path} because it does not exist")
                waveform = np.zeros(24000*29)
            waveform, sr = sf.read(wav_path, dtype='float32')
            waveform = waveform.squeeze()
            waveforms.append(waveform)
        return waveforms

    def forward(self, wav_paths):
        """
        wav_paths: a list of paths to .wav files (mono, 24kHz)
        """
        waveforms = self.load_wavs(wav_paths)
        #print(waveforms)
        inputs = self.processor(waveforms, 
            sampling_rate=24000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=720000).to(self.device) #30s of 24kHz audio
    
        print(inputs['input_values'].shape) #batch size, samples (time*sample_rate)

        outputs = self.model(**inputs, output_hidden_states=True)
        
        all_layer_hidden_states = torch.stack(outputs.hidden_states, dim=1).squeeze()
        print("All hidden states shape:", all_layer_hidden_states.shape)  # [batch, 25, time, 1024]

        # Mean across time dimension to get one vector per layer
        time_reduced_hidden_states = all_layer_hidden_states.mean(dim=2)
        print("Time-reduced shape:", time_reduced_hidden_states.shape)  # [batch, 25, 1024]

        weighted_avg_hidden_states = self.aggregator(time_reduced_hidden_states).squeeze()
        print("Final embedding shape:", weighted_avg_hidden_states.shape)  # [batch, 1024]

        return weighted_avg_hidden_states
