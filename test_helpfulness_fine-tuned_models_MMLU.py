import torch
import json
from datasets import load_dataset
import pdb
import os
from utils import sample_model, read_json_if_exists, clear_memory, load_model, set_seed, get_checkpoint_models
import fairness_experiments.fairness_utils as fairness_utils
import pandas as pd
import numpy as np


def load_mmlu_dataset(mmlu_dataset_name):
    dataset = load_dataset('lukaemon/mmlu', mmlu_dataset_name)
    pd_dataset = pd.DataFrame(dataset['test'])
    return pd_dataset


def test_model(model, tokenizer, dataset):
    score = 0
    letter_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    prob_sum = 0
    log_prob_sum = 0

    for i in range(len(dataset)):
        example = dataset.iloc[i]
        instruction = f"<|begin_of_text|> {example[0]}\nA. {example[1]}\nB. {example[2]}\nC. {example[3]}\nD. {example[4]}\n The letter of the correct answer is "
        prompt = torch.unsqueeze(torch.tensor(tokenizer.encode(instruction)), dim=0)
        label = example[-1]
        label_number = letter_to_number[label]
        with torch.no_grad():
            logits = model(input_ids=prompt.cuda()).logits[0, -1]
            prob = torch.nn.functional.softmax(logits, dim=0)
            log_prob = torch.log(prob)
            res_answer = prob[[32, 33, 34, 35]]
            res_answer_log = log_prob[[32, 33, 34, 35]]
            prediction = torch.argmax(res_answer).cpu().item()
        if prediction == label_number:
            score += 1
        prob_sum += res_answer[label_number]
        log_prob_sum += res_answer_log[label_number]

    accuracy = score / len(dataset)
    prob_mean = prob_sum / len(dataset)
    log_prob_mean = log_prob_sum / len(dataset)

    return accuracy, prob_mean, log_prob_mean

question_template_llama_3_1 = \
'''<|begin_of_text|>{question}\nA) {answerA}.\nB) {answerB}.\nC) {answerC}.\nD) {answerD}.\n The answer is'''

set_seed(42)
clear_memory()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_dir = '/cs/labs/shashua/binyamin/REPE_alignment_helpfulness_tradeoff/lora_finetuned_model_22_08_ver3/'
model_subdirs = get_checkpoint_models(model_dir)

accuracy_dict = dict()
for model_subdir in model_subdirs:
    model_path = os.path.join(model_dir, model_subdir)
    print(f"Testing model in: {model_path}")

    model, tokenizer = load_model(model_path)
    # tokenizer.pad_token = tokenizer.eos_toke
    accuracy_per_dataset =[]
    prob_mean_per_dataset = []
    for mmlu_dataset_name in ['international_law',
                              'high_school_computer_science',
                              'medical_genetics']:  # 'international_law', 'clinical_knowledge'
        dataset = load_mmlu_dataset(mmlu_dataset_name)
        accuracy, prob_mean, log_prob_mean = test_model(model, tokenizer, dataset)
        accuracy_per_dataset.append(accuracy)
        # pdb.set_trace()

    accuracy_dict[model_subdir] = [np.mean(accuracy_per_dataset), np.mean(prob_mean_per_dataset)]

    clear_memory(model, tokenizer)
    print("Memory cleared")
    print()

print(accuracy_dict)
