# Author: Emma Rafkin
# Ling 4467 Assignment 1: Chatbot

import pandas as pd
import time
from transformers import  pipeline
import torch
from dotenv import load_dotenv
import os
import numpy as np
from cerebras.cloud.sdk import Cerebras
from pynvml import *
import psutil

#load environment vars, set up devices, set up GPU usage tracking.
load_dotenv() 
device = 0 if torch.cuda.is_available() else -1
nvmlInit()
gpu = nvmlDeviceGetHandleByIndex(0)

"""
    Core Tasks
    Your chatbot must support:

    Chat: Natural conversation (e.g., "Tell me a joke", "What's your favorite color?")

    Question Answering: Factual queries (e.g., "What is the capital of France?", "Explain photosynthesis")

    Translation: Between language pairs - Chinese/English and one other language/English

    Model Requirements
    Minimum 6 models from minimum 3 different institutions

    Mix of local models (running on your hardware) and API-based models

    Use free models and APIs only - you're responsible for any paid API costs

    Implementation
    Environment: Python + PyTorch or similar

    Starting point: Hugging Face Transformers for local models, API calls for remote models

    Code organization: Single script preferred, with local vs API handled by if/else statements

    Hardware reality: Choose models that actually run on your hardware
"""

def run_local_model(generator, prompt):     
    # Load and run local model. huggingface pipeline
    # set up prompt for the tasks
    text=prompt["text"]
    if prompt["type"] == "QA":
        text = "Answer Question: " + text
    elif prompt["type"] == "translate":
        text = f"Translate from {prompt['source_lang']} to {prompt['target_lang']}: " + text
    # measure inference time
    start_time = time.time()
    answer = generator([{"role": "user", "content": text}],do_sample=False)
    answer = answer[0]["generated_text"][1]["content"]
    end_time = time.time()
    time_to_answer = end_time-start_time
    return answer, time_to_answer


def run_api_model(client, model, prompt):     
    # Call API model    
    # set up prompt based on task 
    text=prompt["text"]
    if prompt["type"] == "QA":
        text = "Answer Question: " + text
    elif prompt["type"] == "translate":
        text = f"Translate from {prompt['source_lang']} to {prompt['target_lang']}: " + text
    start_time = time.time() # measure time but i think this would be pretty fast.
    # ping API :) Structure from Cerebras docs.
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content":text,
            }
        ],
        model=model,
    )
    end_time = time.time()
    time_to_answer = end_time-start_time
    answer = chat_completion.choices[0].message.content
    return answer, time_to_answer

def evaluate_model(model_info, test_prompts):      
    """
        depeding on the model passed in, run prompt through model or ping api. measure inference time and compute resources. Return these
        with answer. Evaluate answer later. 
    
    """
    # Run evaluation and collect metrics 
    rows = []
    load_model_time = None
    if model_info['type'] == 'local':
        # if local load model, measure how long that takes.
        load_model_start = time.time()

        if model_info["token"]:
            generator = pipeline(model=model_info["model"],  device_map="auto", token = os.environ["hf_token"])
        else:
            generator = pipeline(model=model_info["model"], device_map="auto")
        load_model_end = time.time()
        load_model_time = load_model_end - load_model_start
        # for each prompt, get answer and measure inference time and compute resources.
        for prompt in test_prompts:
            cpu, gpu, memory, = show_stats()
            answer, time_to_answer = run_local_model(generator, prompt)   
            rows.append([answer, cpu, gpu, memory, time_to_answer, model_info["model"]])  
    else:
        # if remote, set up cerebras client 
        client = Cerebras(
            api_key=os.environ["cerebras_token"]
        )
        # same as above but not running locally
        for prompt in test_prompts:
            cpu, gpu, memory, = show_stats()
            answer, time_to_answer = run_api_model(client, model_info["model"], prompt)
            rows.append([answer, cpu, gpu, memory, time_to_answer, model_info["model"]])  

    # organize into rows, going into a dataframe
    answer_times = [a[2] for a in rows]
    return load_model_time, rows, np.mean(answer_times)

def run_chatbot(model, prompt):
    """
        The assignment said "build a chatbot" so this is that? 
    """
    if model['type'] == 'local':
        if model["token"]:
            generator = pipeline(model=model["model"],  device_map="auto", token = os.environ["hf_token"])
        else:
            generator = pipeline(model=model["model"], device_map="auto")
        
        answer, _ = run_local_model(generator, prompt)   
    else:
        client = Cerebras(
            api_key=os.environ["cerebras_token"]
        )
       
        answer, _ = run_api_model(client, model["model"], prompt)
    return answer

def show_stats():
    """
        returns CPU, GPU, and memory IN PERCENT
        NOTE: used OpenAI GPT OSS 120B for this code. Edited it to return what I wanted. 
    """
    cpu = psutil.cpu_percent(interval=1)
    util = nvmlDeviceGetUtilizationRates(gpu)
    print(f"[{time.strftime('%H:%M:%S')}] CPU {cpu:5.1f}% | GPU {util.gpu:3d}% | Mem {util.memory:3d}%")
    return cpu, util.gpu, util.memory

def run_eval():
    """
        Run the evaluation code. 
    """

    # A prompt is in the form: {
    #   "type": "chat"|"QA"|"translate"
    #   "text": str
    #   "target_lang": lang,
    #   "source_lang": lang,
    # }
    # target and source langs only for translate
    test_prompts = [
        {
            "type": "chat",
            "text": "Are you alive?",
        },
        {
            "type": "chat",
            "text": "Plan me a roadtrip from OK to ND?"
        },
        {
            "type": "translate",
            "target_lang": "English",
            "source_lang": "Spanish",
            "text": "Me gusta tocar la guitarra"
        },
        {
            "type": "translate",
            "target_lang": "Chinese",
            "source_lang": "English",
            "text": "I like to play the guitar"
        },
        {
            "type": "QA",
            "text": "What is the capital of France?"
        },
        {
            "type": "QA",
            "text": "How does photosynthesis work?"
        }
    ]
    # model is in the form {
    #         "type": "local" | "api",
    #         "model": huggingface_str or Cerebras str,
    #         "token": bool (if auth token needed)
    # }
    models = [
        {
            "type": "api",
            "model": "gpt-oss-120b",
            "token": False
        },
        {
            "type": "api",
            "model": "llama-4-maverick-17b-128e-instruct",
            "token": False
        },
        {
            "type": "api",
            "model": "qwen-3-235b-a22b-instruct-2507",
            "token": False
        },
        {
            "type": "local",
            "model": "google/gemma-3-270m-it",
            "token": True
        },
        {
            "type": "local",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "token": False
        },
        {
            "type": "local",
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "token": True
        }
        
    ]
    rows = []
    model_times = []
    for m in models:
        load_model_time, model_rows, mean_answer_time = evaluate_model(model_info=m, test_prompts=test_prompts) 
        print("time to load model: ", load_model_time)
        model_times.append([m["model"], load_model_time, mean_answer_time])
        rows += model_rows
    # spit out information in a dataframe for evaluation
    df = pd.DataFrame(rows, columns=["answer",  "cpu", "gpu", "memory", "time", "model_name"]) 
    df.to_csv("./output/answers.csv", index=False)
    df_times = pd.DataFrame(model_times, columns=["model", "model_load_time", "mean_answer_time"])
    df_times.to_csv("./output/model_specs.csv", index=False)


if __name__ == "__main__":
    mode = "eval" # "eval" or "chat"
    if mode == "eval":
        run_eval()
    else:
        # e.g. model and answer. later these could be args. 
        model = {
            "type": "api",
            "model": "gpt-oss-120b",
            "token": False
        }
        prompt = {
            "type": "translate",
            "target_lang": "English",
            "source_lang": "Spanish",
            "text": "Me gusta tocar la guitarra"
        }
        answer = run_chatbot(model, prompt)
        print(answer)