import torch
import soundfile as sf
import librosa
import pandas as pd
import os
import torch
from encodec import EncodecModel


# Initial code generated using ChatGPT

# Load the 24 kHz model (32 codebooks)
model = EncodecModel.encodec_model_24khz()  
model.set_target_bandwidth(24.0)  # kbps → uses all 32 codebooks

# Load audio file (mono, 24kHz)
for file in os.listdir("a4_input/codec"):
    if file != ".DS_store":
        print(file)
        audio, sr = librosa.load(f"a4_input/codec/{file}", sr=24000, mono=True)
        audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0)  # shape: [1, num_samples]
        original_size = audio.numel() * 2
        # Encode once
        encoded = model.encode(audio)
        audio_codes = encoded[0][0]     # shape: [batch, codebooks=32, time]
        audio_scales = encoded[0][1]
        K = audio_codes.shape[1]  # total number of codebooks
        compression_stats=[]
        for n_codebooks in range(1, K+1):

            bitrate = 24.0 * (n_codebooks / 32)  # kbps

            # Encoded size estimate
            # Each codebook entry index stored as int16 (2 bytes)
            encoded_size_bytes = n_codebooks * audio_codes.shape[-1] * 2

            compression_ratio = original_size / encoded_size_bytes
            compression_stats.append([n_codebooks, bitrate, compression_ratio, encoded_size_bytes, original_size])

            # Keep only the first n_codebooks
            partial_codes = audio_codes[:, :n_codebooks, :]

            # Decode back to audio
            decoded_audio = model.decode([(partial_codes, audio_scales)]).squeeze().detach().numpy()

            # Save reconstructed audio
            os.makedirs(f"a4_output/{file.split('.')[0]}", exist_ok=True)
                
            output_name = f"a4_output/encodec/{file.split('.')[0]}/{n_codebooks}_codebooks.wav"
            sf.write(output_name, decoded_audio, sr)
            print(f"Saved: {output_name}")
        df = pd.DataFrame(compression_stats, columns=["n_codebooks","bitrate","compression_ratio", "encoded_size", "original_size"])
        df.to_csv(f"a4_output/encodec/{file.split('.')[0]}/stats.csv", index=False)
        