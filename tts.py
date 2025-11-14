from gtts import gTTS
import time
from transformers import pipeline, FastSpeech2ConformerHifiGan
import soundfile as sf
import json
from tqdm import tqdm

def run_gtts(text):
    # Create a gTTS object
    tts = gTTS(text)

    # Save the audio to a file
    tts.save(f"a4_output/tts/gtts/{text.split(' ')[0]}.mp3")

def bark(text):
    synthesiser = pipeline("text-to-speech", "suno/bark-small")
    speech = synthesiser(text, forward_params={"do_sample": True})
    sf.write(f"a4_output/tts/bark/{text.split(' ')[0]}.mp3", data=speech["audio"].squeeze(), rate=speech["sampling_rate"])

def mms(text):
    synthesiser = pipeline("text-to-speech", "facebook/mms-1b")
    speech = synthesiser(text, forward_params={"do_sample": True})
    sf.write(f"a4_output/tts/mms/{text.split(' ')[0]}.mp3", data=speech["audio".squeeze()], rate=speech["sampling_rate"])

def vits(text):
    synthesiser = pipeline("text-to-speech",  "facebook/mms-tts-eng")
    speech = synthesiser(text, forward_params={"do_sample": True})
    sf.write(f"a4_output/tts/vits/{text.split(' ')[0]}.mp3", data=speech["audio"].squeeze(), rate=speech["sampling_rate"])
   

def speech_t5(text):
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
    speech = synthesiser(text, forward_params={"do_sample": True})
    sf.write(f"a4_output/tts/speech_t5/{text.split(' ')[0]}.mp3", data=speech["audio"].squeeze(), rate=speech["sampling_rate"])


def fastspeech2_conformer(text):
    vocoder = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan")
    synthesiser = pipeline(model="espnet/fastspeech2_conformer", vocoder=vocoder)

    speech = synthesiser("Hello, my dog is cooler than you!")

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
    models = [run_gtts, bark, mms, vits, speech_t5, fastspeech2_conformer]
    model_names = ["gtts", "bark", "mms", "vits", "speech_t5", "fastspeech2"]
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




