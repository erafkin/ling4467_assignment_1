from transformers import pipeline
from faster_whisper import WhisperModel
import torch
from datasets import load_dataset, Audio
import pandas as pd
import time
from jiwer import wer
from tqdm import tqdm
from whisper_normalizer.basic import BasicTextNormalizer
normalizer = BasicTextNormalizer()

def load_whisper_pipeline(model_str:str, faster:bool=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if faster:
        return WhisperModel(model_str, device=device, compute_type="float16")
    else:
        return pipeline(
            "automatic-speech-recognition",
            model=model_str,
            device=device
        )



def run_model(transcriber, dataset, output_name):
    # Process audio files     
    # Calculate WER and RTF     
    # Return metrics and transcriptions      
    df_rows = []
    for row in tqdm(dataset):
        start_time = time.time()
        transcription = transcriber(row["audio"]["array"])
        end_time = time.time()
        rtf = (end_time - start_time) / (len(row["audio"]["array"])/row["audio"]["sampling_rate"])
        transcription = normalizer(transcription["text"])
        reference = normalizer(row["text"])
        row_wer = wer(reference=reference, hypothesis=transcription)
        df_rows.append([transcription, reference, row_wer, rtf])
    df = pd.DataFrame(df_rows, columns=["transcription", "reference", "wer", "rtf"])
    df.to_csv(output_name, index=False)


def compare_models(models, dataset_options):
    for dataset in dataset_options:
        if dataset == "librispeech":
            ds = load_dataset("openslr/librispeech_asr", split='test.clean', streaming=True).take(10)
            test_ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
            test_ds = list(ds)
        elif dataset == "fleurs":
            ds = load_dataset("google/fleurs", "sv_se", streaming=True)["test"].take(10) # swedish
            test_ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
            test_ds = test_ds.rename_column("transcription", "text")
        elif dataset == "local":
            ...
        else:
            raise Exception(f"dataset {dataset} does not exist")
        for model in models:
            whisper = load_whisper_pipeline(model_str=model, faster=False)
            run_model(whisper, test_ds, f"{dataset}_{model.split('/')[-1]}.csv")


if __name__ == "__main__":
    models = ["openai/whisper-tiny"]
    ds_options = ["fleurs"]
    compare_models(models, ds_options)