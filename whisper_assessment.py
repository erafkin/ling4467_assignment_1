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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if faster:
        return WhisperModel(model_str, device=device, compute_type="float16")
    else:
        return pipeline(
            "automatic-speech-recognition",
            model=model_str,
            device=device,
            dtype=torch.float32
        )

def run_model(transcriber, dataset, output_name):
    # Process audio files     
    # Calculate WER and RTF     
    # Return metrics and transcriptions      
    df_rows = []
    for row in tqdm(dataset):
        start_time = time.time()
        proc.cpu_percent(interval=None)
        transcription = transcriber(row["audio"]["array"])
        cpu =  proc.cpu_percent(interval=None)
        end_time = time.time()
        rtf = (end_time - start_time) / (len(row["audio"]["array"])/row["audio"]["sampling_rate"])
        transcription = normalizer(transcription["text"])
        reference = normalizer(row["text"])
        row_wer = wer(reference=reference, hypothesis=transcription)
        df_rows.append([transcription, reference, row_wer, rtf, cpu])
    df = pd.DataFrame(df_rows, columns=["transcription", "reference", "wer", "rtf", "cpu"])
    df.to_csv(output_name, index=False)


def compare_models(models, dataset_options):
    for dataset in dataset_options:
        if dataset == "librispeech":
            ds = load_dataset("openslr/librispeech_asr", split='test.clean', streaming=True).take(100)
            test_ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
            test_ds = list(ds)
        elif dataset == "fleurs":
            ds = load_dataset("google/fleurs", "sv_se", streaming=True, trust_remote_code=True)["test"].take(100) # swedish
            test_ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
            test_ds = test_ds.rename_column("transcription", "text")
        elif dataset == "local":
            df = pd.read_csv("dataset/dataset.csv")
            ds = Dataset.from_pandas(df)
            test_ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

        else:
            raise Exception(f"dataset {dataset} does not exist")
        for model in models:
            if not os.path.exists(f"a3_whisper_assessment/{dataset}_{model.split('/')[-1]}.csv"):
                whisper = load_whisper_pipeline(model_str=model, faster=False)
                run_model(whisper, test_ds, f"a3_whisper_assessment/{dataset}_{model.split('/')[-1]}.csv")

def create_results_dataframe():
    rows = []
    for results_file in os.listdir("a3_whisper_assessment"):
        if results_file != ".DS_Store":
            csv = pd.read_csv(f"a3_whisper_assessment/{results_file}")
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
    data.to_csv("a3_whisper_assessment/summary/phase_1_summary.csv")
    data = data.T
    # Convert the DataFrame to an HTML string
    html_table = data.to_html()

    # Print the HTML string (optional)
    print(html_table)

    # Save the HTML string to a file
    with open("a3_whisper_assessment/summary/phase_1_summary.html", "w") as f:
        f.write(html_table)
    
if __name__ == "__main__":
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
    # compare_models(models, ds_options)
    create_results_dataframe()