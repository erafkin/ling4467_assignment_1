import sounddevice as sd
import librosa
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
import torch
import wave
import webrtcvad
import numpy as np
import time
from chatbot import run_chatbot

def save_audio_to_file(output_filename, audio_data, sample_rate, channels:int=1):
    # Concatenate all audio chunks
    full_audio = np.concatenate(audio_data, axis=0)
    
    # Save to WAV file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 2 bytes per sample for int16
        wf.setframerate(sample_rate)
        wf.writeframes(full_audio.tobytes())
    
    print(f"Recording saved as {output_filename}")

def record_speech(output_filename, sample_rate=16000, chunk_duration=0.02):
    """
    Base code writen by Qwen3-Coder-30B-A3B-Instruct

    Record speech from microphone with VAD and save to file.
    
    Args:
        output_filename (str): Output filename
        rate (int): Audio sample rate (default 16000)
        chunk_duration (float): Duration of each audio chunk in seconds (default 0.02 for 20ms)
    """
    
    # Setup VAD (Voice Activity Detector)
    vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (most aggressive)
    
    # Audio parameters
    channels = 1
    dtype = np.int16
    
    # Calculate chunk size based on duration
    chunk_size = int(sample_rate * chunk_duration)
    
    print("Listening... Speak now (will stop automatically when silent)")
    
    audio_data = []
    silence_count = 0
    max_silence = 10  # Stop after 10 consecutive silent chunks
    
    def audio_callback(indata, frames, time, status):
        nonlocal silence_count
        
        # Convert to bytes for VAD processing
        audio_bytes = indata.ravel().astype(dtype).tobytes()
        
        # Check if speech is detected
        is_speech = vad.is_speech(audio_bytes, sample_rate)
        
        if is_speech:
            silence_count = 0  # Reset silence counter
        else:
            silence_count += 1
        
        # Store audio data
        audio_data.append(indata.copy())
        
        # Stop if silence threshold reached
        return (None, sd.CallbackStop) if silence_count >= max_silence else (None, None)
    
    try:
        # Start recording with callback
        with sd.InputStream(callback=audio_callback, 
                          channels=channels, 
                          samplerate=sample_rate, 
                          dtype=dtype,
                          blocksize=chunk_size):
            
            # Wait for recording to complete
            while silence_count < max_silence:
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
        print("Recording stopped due to silence")
        
    except Exception as e:
        print(f"Error during recording: {e}")
    
    save_audio_to_file(output_filename=output_filename, audio_data=audio_data, sample_rate=sample_rate, channels=channels)

def playback_audio(audio_path:str):
    """
        playback saved audio from file
    """
    data, sample_rate = librosa.load(audio_path)
    
    # Play the audio
    sd.play(data, sample_rate)
    sd.wait() # Wait until the audio is finished playing

def load_audio_file(audio_path:str):
    """
        load audio file
    """
    audio_data, sample_rate = librosa.load(audio_path)
    return audio_data, sample_rate

def wav_to_spectrogram(audio_file, output_file="output_spec"):
    """
        Take audio file, load it, create a spectrogram image, save image. 
    """
    # tried matplotlib and scipy -- they were worse.
    data, sr = load_audio_file(audio_file)

    D = librosa.stft(data, n_fft=4096)
    # in decibels 
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='gray_r')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_file)

def speech_to_text(audio_file:str, model:str="openai/whisper-tiny"):
    """
        pass audio file to Whisper. Whisper can take just the filepath. Run on GPU if available. 
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        device=device
    )
    transcribed_text = transcriber(audio_file, generate_kwargs={"language": "en"})
    return transcribed_text

def record_audio_to_llm_pipeline(audio_output_file:str="a2/recorded_audio.wav"):
    """
        String everything together! Record speech --> ASR --> Chatbot. 
        Print answer and latency measurements (after recording ends)
    """
    record_speech(output_filename=audio_output_file)
    start_time = time.time()
    transcribed_text = speech_to_text(audio_output_file)
    print("Transcribed text: ", transcribed_text)
    llm = {
            "type": "api",
            "model": "gpt-oss-120b",
            "token": False
        }
    prompt = {
        "type": "chat",
        "text": transcribed_text["text"]
    }
    llm_answer = run_chatbot(llm, prompt)
    end_time = time.time()
    print("LLM: ", llm_answer)
    print("Overall time: ", end_time - start_time)

    

if __name__ =="__main__":
    # data = record_speech("a2_output/my_recording2.wav")

    wav_to_spectrogram("a2_output/my_recording.wav", "a2_output/output_spec_4096")
    # playback_audio("a2_output/my_recording.wav")
    # speech_to_text("a2_output/my_recording.wav")
    # record_audio_to_llm_pipeline("a2_output/my_recording2.wav")