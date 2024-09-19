import torch
import json
from datasets import load_dataset
import pdb
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import sample_model, read_json_if_exists, clear_memory, load_model, set_seed, get_checkpoint_models
import pandas as pd
import numpy as np


def load_mmlu_dataset(mmlu_dataset_name):
    dataset = load_dataset('lukaemon/mmlu', mmlu_dataset_name, trust_remote_code=True)
    pd_dataset = pd.DataFrame(dataset['test'])
    return pd_dataset


def test_model(model, tokenizer, dataset):
    score = 0
    letter_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    prob_sum = 0

    for i in range(len(dataset)):
        example = dataset.iloc[i]
        instruction = f"<|begin_of_text|> {example['input']}\nA. {example['A']}\nB. {example['B']}\nC. {example['C']}\nD. {example['D']}\n The letter of the correct answer is "
        prompt = torch.tensor(tokenizer.encode_plus(instruction, return_tensors="pt", padding=True)['input_ids'])
        label = example['target']
        label_number = letter_to_number[label]
        model.eval()
        with torch.no_grad():
            logits = model(input_ids=prompt.cuda()).logits[0, -1]
            res_answer = logits[[32, 33, 34, 35]]
            probs = torch.nn.functional.softmax(res_answer, dim=0)
            prediction = torch.argmax(res_answer).cpu().item()
        if prediction == label_number:
            score += 1
        prob_sum += probs[label_number]

    accuracy = score / len(dataset)
    prob_mean = prob_sum / len(dataset)

    return accuracy, float(prob_mean)

question_template_llama_3_1 = \
'''<|begin_of_text|>{question}\nA) {answerA}.\nB) {answerB}.\nC) {answerC}.\nD) {answerD}.\n The answer is'''

set_seed(42)
clear_memory()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_dir = '/cs/labs/shashua/binyamin/REPE_alignment_helpfulness_tradeoff/lora_finetuned_model_17_09_regular_100'
model_subdirs = get_checkpoint_models(model_dir)

accuracy_dict = dict()
for model_path in model_subdirs:
    print(f"Testing model in: {model_path}")

    model, tokenizer = load_model(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    accuracy_per_dataset =[]
    prob_mean_per_dataset = []
    for mmlu_dataset_name in ['international_law', 'high_school_computer_science', 'medical_genetics',
                              'clinical_knowledge', 'college_mathematics']:
        dataset = load_mmlu_dataset(mmlu_dataset_name)
        accuracy, prob_mean = test_model(model, tokenizer, dataset)
        accuracy_per_dataset.append(accuracy)
        prob_mean_per_dataset.append(prob_mean)
        # pdb.set_trace()

    accuracy_dict[model_path] = [np.mean(accuracy_per_dataset), np.mean(prob_mean_per_dataset)]

    clear_memory(model, tokenizer)
    print("Memory cleared")
    print()

print(accuracy_dict)
