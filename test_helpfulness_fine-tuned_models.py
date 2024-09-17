import torch
import json
from datasets import load_dataset
import pdb
import os
from utils import sample_model, read_json_if_exists, clear_memory, load_model, set_seed, get_checkpoint_models

question_template_llama_3_1 = \
"""<|begin_of_text|>{user_prompt}"""

set_seed(42)
clear_memory()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

human_eval_data = load_dataset("openai/openai_humaneval")
human_eval_dict = {q['task_id']: q for q in human_eval_data['test']}
wanted_q = ['HumanEval/1', 'HumanEval/2', 'HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/6', 'HumanEval/7', 'HumanEval/8', 'HumanEval/9', 'HumanEval/12', 'HumanEval/22', 'HumanEval/27', 'HumanEval/28', 'HumanEval/29', 'HumanEval/30', 'HumanEval/42', 'HumanEval/43', 'HumanEval/44', 'HumanEval/149', 'HumanEval/150', 'HumanEval/157', 'HumanEval/161', 'HumanEval/162']
human_eval_dict = {q['task_id']: q for q in human_eval_data['test'] if q['task_id'] in wanted_q}


model_dir = '/cs/labs/shashua/binyamin/REPE_alignment_helpfulness_tradeoff/lora_finetuned_model_17_09_regular_100/'
model_subdirs = get_checkpoint_models(model_dir)

generation_path = 'code_generations/fine-tuned_model_generations_17_09_2024_regular_100_epochs.json'
generation_dict = read_json_if_exists(generation_path)
for model_subdir in model_subdirs:
    model_path = os.path.join(model_dir, model_subdir)
    if model_subdir not in generation_dict:
        generation_dict[model_subdir] = dict()
    print(f"Testing model in: {model_path}")

    model, tokenizer = load_model(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    for q_id, q in human_eval_dict.items():
        if q_id in generation_dict[model_subdir]:
            continue
        print(f"Question: {q_id}")
        print(f"Model: {model_subdir}")
        print("Generating samples...")

        samples = sample_model(model, tokenizer, q, num_samples=16, batch_size=8,
                               question_template_for_sample=question_template_llama_3_1)
        generation_dict[model_subdir][q_id] = samples

    with open(generation_path, 'w') as file:
        json.dump(generation_dict, file)

    clear_memory(model, tokenizer)
    print("Memory cleared")
    print()
