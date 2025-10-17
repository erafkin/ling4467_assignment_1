from transformers import pipeline
from faster_whisper import WhisperModel
import torch
from datasets import load_dataset, Audio, Dataset
import pandas as pd
import time
from jiwer import wer
from tqdm import tqdm
from whisper_normalizer.basic import BasicTextNormalizer
import psutil
import os 

proc = psutil.Process()

normalizer = BasicTextNormalizer()

def load_whisper_pipeline(model_str:str, faster:bool=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if faster:
        return WhisperModel(model_str, device=device, compute_type="int8")
    else:
        return pipeline(
            "automatic-speech-recognition",
            model=model_str,
            device=device,
            dtype=torch.int8
        )

def run_model(transcriber, dataset, output_name, phase):
    # Process audio files     
    # Calculate WER and RTF     
    # Return metrics and transcriptions      
    df_rows = []
    for row in tqdm(dataset):
        start_time = time.time()
        proc.cpu_percent(interval=None)
        if phase < 2:
            transcription = transcriber(row["audio"]["array"])["text"]
        else:
            segments, info = transcriber.transcribe(row["audio"]["array"])
            transcription = [segment.text for segment in segments]
            transcription = " ".join(transcription)
        cpu =  proc.cpu_percent(interval=None)
        end_time = time.time()
        rtf = (end_time - start_time) / (len(row["audio"]["array"])/row["audio"]["sampling_rate"])
        transcription = normalizer(transcription)
        reference = normalizer(row["text"])
        row_wer = wer(reference=reference, hypothesis=transcription)
        df_rows.append([transcription, reference, row_wer, rtf, cpu])
    df = pd.DataFrame(df_rows, columns=["transcription", "reference", "wer", "rtf", "cpu"])
    df.to_csv(output_name, index=False)


def compare_models(models, dataset_options, phase:int = 1):
    for dataset in dataset_options:
        if dataset == "librispeech":
            test_ds = load_dataset("openslr/librispeech_asr", split='test.clean', streaming=True).take(100)
            test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16_000))
            test_ds = list(test_ds)
        elif dataset == "fleurs":
            test_ds = load_dataset("google/fleurs", "sv_se", streaming=True, trust_remote_code=True)["test"].take(100) # swedish
            test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16_000))
            test_ds = test_ds.rename_column("transcription", "text")
        elif dataset == "local":
            df = pd.read_csv("dataset/dataset.csv")
            test_ds = Dataset.from_pandas(df)
            test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16_000))

        else:
            raise Exception(f"dataset {dataset} does not exist")
        for model in models:
            if not os.path.exists(f"a3_whisper_assessment/phase{phase}/{dataset}_{model.split('/')[-1]}.csv"):
                whisper = load_whisper_pipeline(model_str=model, faster=False if phase == 1 else True)
                run_model(whisper, test_ds, f"a3_whisper_assessment/phase{phase}/{dataset}_{model.split('/')[-1]}.csv", phase)

def create_results_dataframe(phase: int = 1):
    rows = []
    for results_file in os.listdir(f"a3_whisper_assessment/phase{phase}"):
        if results_file != ".DS_Store":
            csv = pd.read_csv(f"a3_whisper_assessment/phase{phase}/{results_file}")
            dataset = results_file.split("_")[0]
            model = results_file.split("_")[1][:-4]
            avg_wer = csv.wer.mean()
            avg_rtf = csv.rtf.mean()
            avg_cpu = csv.rtf.mean()
            rows.append([ model, dataset, avg_wer, avg_rtf, avg_cpu])
    df = pd.DataFrame(rows, columns = [ "model", "dataset","avg_wer", "avg_rtf", "avg_cpu"])
    df = df.sort_values(by=["model", 'dataset'])
    # transpose
    df = df.T
    # ChatGPT helped me make a multi-header df
    header_rows = 2                     # how many rows belong to the header?
    header = df.iloc[:header_rows]    # a tiny DataFrame with just those rows
    data   = df.iloc[header_rows:]    # the real numeric data
    cols_tuples = list(zip(*[header.iloc[i].tolist() for i in range(header_rows)]))

    # Create the MultiIndex and assign it to the data frame
    data.columns = pd.MultiIndex.from_tuples(cols_tuples)
    print(data)
    data.to_csv(f"a3_whisper_assessment/summary/phase_{phase}_summary.csv")
    data = data.T
    # Convert the DataFrame to an HTML string
    html_table = data.to_html()

    # Print the HTML string (optional)
    print(html_table)

    # Save the HTML string to a file
    with open(f"a3_whisper_assessment/summary/phase_{phase}_summary.html", "w") as f:
        f.write(html_table)

def phase_1():
    models = [
            "openai/whisper-tiny", 
            "openai/whisper-small", 
            "openai/whisper-medium", 
            "openai/whisper-base", 
            "distil-whisper/distil-large-v3.5", 
            "distil-whisper/distil-large-v2",
            "openai/whisper-tiny.en", 
            "openai/whisper-large-v3-turbo"]
    ds_options = ["fleurs", "local", "librispeech"]
    compare_models(models, ds_options)
    create_results_dataframe()

def phase_1_again():
    models = [
            "openai/whisper-tiny", 
            "openai/whisper-small", 
            "distil-whisper/distil-large-v3.5"]
    ds_options = ["fleurs", "local", "librispeech"]
    compare_models(models, ds_options, 1)
    create_results_dataframe(1)

def phase_2():
    models = ["small", "medium", "distil-whisper/distil-large-v3-ct2"]
    ds_options = ["fleurs", "local", "librispeech"]
    compare_models(models, ds_options, 2)
    create_results_dataframe(2)
    
if __name__ == "__main__":
    phase_2()