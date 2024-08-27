import csv
import tqdm
import torch
import json
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from tqdm import tqdm
import os
import numpy as np
from datasets import load_dataset
import random
import pdb
import os
import gc
from generate_code_with_REPE import sample_model


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, token=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", legacy=False, token=True)
    model.to(device)
    return model, tokenizer

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

torch.manual_seed(42)
random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

human_eval_data = load_dataset("openai/openai_humaneval")
human_eval_dict = {q['task_id']: q for q in human_eval_data['test']}
wanted_q = ['HumanEval/3', 'HumanEval/4', 'HumanEval/7', 'HumanEval/8', 'HumanEval/12', 'HumanEval/22', 'HumanEval/27', 'HumanEval/28', 'HumanEval/29', 'HumanEval/30', 'HumanEval/42', 'HumanEval/43', 'HumanEval/44', 'HumanEval/149', 'HumanEval/150', 'HumanEval/157', 'HumanEval/161', 'HumanEval/162']
human_eval_dict = {q['task_id']: q for q in human_eval_data['test'] if q['task_id'] in wanted_q}


model_dir = '/cs/labs/shashua/binyamin/REPE_alignment_helpfulness_tradeoff/lora_finetuned_model_22_08_ver3/'
model_subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

generation_path = 'fine-tuned_model_generations.json'
generation_dict = read_json_if_exists(generation_path)
for model_subdir in model_subdirs:
    model_path = os.path.join(model_dir, model_subdir)
    if model_subdir not in generation_dict:
        generation_dict[model_subdir] = dict()
    print(f"Testing model in: {model_path}")

    model, tokenizer = load_model(model_path)

    for q_id, q in human_eval_dict.items():
        if q_id in generation_dict[model_subdir]:
            continue
        print(f"Question: {q_id}")
        print(f"Model: {model_subdir}")
        print("Generating samples...")

        samples = sample_model(model, tokenizer, q, num_samples=16, batch_size=16)
        generation_dict[model_subdir][q_id] = samples

    with open(generation_path, 'w') as file:
        json.dump(generation_dict, file)

    # Clear memory
    del model
    del tokenizer
    clear_memory()
    print("Memory cleared")
    print()

