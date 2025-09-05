# Author: Emma Rafkin
# Ling 4467 Assignment 1: Chatbot

import pandas as pd
import resource
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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

def run_local_model(model, tokenizer, prompt):     
    # Load and run local model    
    start_time = time.time()
    text=prompt["text"]
    if prompt["type"] == "QA":
        text = "Answer Question: " + text
    elif prompt["type"] == "translate":
        text = f"Translate from {prompt['source_lang']} to {prompt['target_lang']}: " + text
    print(prompt)
    model_inputs = tokenizer([text], return_tensors="pt")
    generated_ids = model.generate(**model_inputs)
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(answer)
    end_time = time.time()
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss 
    time_to_answer = end_time-start_time
    return answer, max_rss, time_to_answer


def run_api_model(api, prompt):     
    # Call API model     
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss 
    return max_rss

def evaluate_model(model_info, test_prompts):      
    """
        Latency: Inference time per request

        Accuracy: Does it give reasonable/correct responses?

        Model size: Disk space and memory requirements + model loading time

        Resource usage: CPU/GPU utilization during inference

        TODO: Create a dataframe with all of the questions, the time it takes to answer them, and their accuracy
                also create a dataframe with the model size & average resource usage per call
    
    """
    # Run evaluation and collect metrics 
    rows = []
    if model_info['type'] == 'local':
        load_model_start = time.time()
        #TODO LOAD MODEL
        # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        if "token" in model_info:
            model =  AutoModelForCausalLM.from_pretrained(model_info["model"], device_map="auto", token = model_info["token"])
        else:
            model =  AutoModelForCausalLM.from_pretrained(model_info["model"], device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_info["model"], padding_side="left")
        load_model_end = time.time()
        load_model_time = load_model_end - load_model_start
        for prompt in test_prompts:
            answer, max_rss, time_to_answer = run_local_model(model, tokenizer, prompt)   
            rows.append([answer, max_rss, time_to_answer])  
    else:
        max_rss = run_api_model(...)
    df = pd.DataFrame(rows, columns=["answer", "max_rss", "time"]) # TODO COLUMNS
    load_model_time # do something with this?
    return load_model_time, df
    

if __name__ == "__main__":
    test_prompts = [{
        "type": "chat",
        "text": "Tell me a joke",
    },
    {
        "type": "chat",
        "text": "What's your favorite color?"
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
    }
    ]
    models = [
        {
            "type": "local",
            "model": "google/gemma-3-270m-it",
        },
    ]
    for m in models:
        load_model_time, df = evaluate_model(model_info=m, test_prompts=test_prompts) 
        print("time to load model: ", load_model_time)
        print(df)  