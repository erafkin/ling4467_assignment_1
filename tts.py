from gtts import gTTS
import time
from transformers import pipeline, FastSpeech2ConformerHifiGan
import soundfile as sf
import json
from tqdm import tqdm
from datasets import load_dataset
import torch


def run_gtts(text):
    # Create a gTTS object
    tts = gTTS(text)

    # Save the audio to a file
    tts.save(f"a4_output/tts/gtts/{text.split(' ')[0]}.mp3")

def bark(text):
    synthesiser = pipeline("text-to-speech", "suno/bark-small")
    speech = synthesiser(text, forward_params={"do_sample": True})
    sf.write(f"a4_output/tts/bark/{text.split(' ')[0]}.mp3", speech["audio"].squeeze(), samplerate=speech["sampling_rate"])


def bark_lg(text):
    synthesiser = pipeline("text-to-speech", "suno/bark")
    speech = synthesiser(text, forward_params={"do_sample": True})
    sf.write(f"a4_output/tts/bark_lg/{text.split(' ')[0]}.mp3", speech["audio"].squeeze(), samplerate=speech["sampling_rate"])


def vits(text):
    synthesiser = pipeline("text-to-speech",  "facebook/mms-tts-eng")
    speech = synthesiser(text)
    sf.write(f"a4_output/tts/vits/{text.split(' ')[0]}.mp3", speech["audio"].squeeze(), samplerate=speech["sampling_rate"])
   

def speech_t5(text):
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
    
    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
    sf.write(f"a4_output/tts/speech_t5/{text.split(' ')[0]}.mp3", speech["audio"].squeeze(), samplerate=speech["sampling_rate"])


def fastspeech2_conformer(text):
    synthesiser = pipeline(model="vibevoice/VibeVoice-1.5B")

    speech = synthesiser(text)

    sf.write(f"a4_output/tts/fastspeech2/{text.split(' ')[0]}.mp3", speech["audio"].squeeze(), samplerate=speech["sampling_rate"])

if __name__ == "__main__":
    texts = [
        "Hello world, my name is Emma Rafkin",
        "It is cold outside today",
        "He likes to read, but he read the book that she is reading already",
        "She blew out a breath, and the cold air made the blown breath appear blue",
        "That was a totally crazy thing for you to do!",
        "Why do you think that?",
        "Give papa a cup of proper coffee in a copper coffee cup.",
        "Peter Piper picked a peck of picked peppers",
        "Fuzzy wuzzy was a bear. Fuzzy wuzzy had no hair. Fuzzy wuzzy wasn't fuzzy, was he?",
        "Some people think that 74.5 degrees Farenheit is hot.",
        "In 1986, the USA got 1st and 2nd place at the Olympics",
        "The algorithm uses stochastic gradient descent for optimization.",
        "What are you eating?",
        "You are eating what?",
        "The algorithm uses stochastic gradient descent for optimization."
    ]
    models = [ vits, speech_t5, fastspeech2_conformer, bark_lg, run_gtts, bark]
    model_names = [ "vits", "speech_t5", "vibe",  "bark_lg", "gtts", "bark",]
    latency = {}
    for idx, model in tqdm(enumerate(models)):
        latency[model_names[idx]] = []
        for text in tqdm(texts):
            start = time.time()
            model(text)
            end = time.time()
            latency[model_names[idx]].append(end-start)
    with open("a4_output/tts/latency.json", "w") as f:
        json.dump(latency, f, indent=4)




