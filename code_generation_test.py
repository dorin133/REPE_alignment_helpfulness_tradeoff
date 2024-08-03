import csv
import tqdm
import torch
import json
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from tqdm import tqdm
import os
import numpy as np
from repe.rep_control_reading_vec import WrappedReadingVecModel
import harmfulness_utils_ver2
from datasets import load_dataset
from repe import repe_pipeline_registry
import random
import pdb
repe_pipeline_registry()

random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

question_template = \
"""[INST] <<SYS>>
You are a helpful, respectful and honest assistant. You will be given a python function to complete.
Complete the function correctly. Separate the code of the function from the rest of your message.
Avoid unnecessary indentation in your answer. Only give one answer. 
<</SYS>>
{user_prompt} [/INST]"""


def sample_model(model, tokenizer, question, num_samples=32, batch_size=2):
    prompt = question_template.format(user_prompt=question['prompt'])
    q_encoding = tokenizer.encode_plus(prompt, return_tensors="pt", padding=True)
    input_ids = q_encoding['input_ids'].to(device)
    attn_mask = q_encoding['attention_mask'].to(device)
    num_batches = num_samples // batch_size

    all_answers = []
    for j in range(num_batches):
        with torch.inference_mode():
            outputs = model.generate(input_ids, max_new_tokens=1800, temperature=1.0, do_sample=True,
                                     top_p=0.95, attention_mask=attn_mask,
                                     return_dict_in_generate=True,
                                     pad_token_id=tokenizer.pad_token_id, num_return_sequences=batch_size)
        # decode the input only
        partial_given_answers = [tokenizer.decode(output_sequence[:input_ids.shape[1]], skip_special_tokens=True) for
                                 output_sequence in outputs.sequences]
        # decode the entire output, and remove the input from it
        curr_answers = [
            tokenizer.decode(outputs.sequences[i], skip_special_tokens=True).replace(partial_given_answers[i],
                                                                                     "").replace(
                '<s>', "").replace('</s>', "") for i in range(len(partial_given_answers))]
        all_answers += curr_answers

    return all_answers


# model_name = "Llama-2-13b"
################################# load model
# model_name_or_path_chat = f"../../llama2/{model_name}/"
# model_name_or_path_chat = 'meta-llama/Llama-2-13b-hf'
model_name_or_path = "/cs/labs/shashua/noamw02/llama_weights_hf/llama-2-13b-chat/"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", token=True).eval()
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1
print("load model finished!")

human_eval_data = load_dataset("openai/openai_humaneval")
human_eval_dict = {q['task_id']: q for q in human_eval_data['test']}
wanted_q = ['HumanEval/3', 'HumanEval/4', 'HumanEval/7', 'HumanEval/8', 'HumanEval/12', 'HumanEval/22', 'HumanEval/27', 'HumanEval/28', 'HumanEval/29', 'HumanEval/30']
filtered_human_eval_dict = {q['task_id']: q for q in human_eval_data['test'] if q['task_id'] in wanted_q}

################################# load the llama2 model vocabulary
vocabulary = tokenizer.get_vocab()

####################### load the harmful dataset behavior - by github
train_data, train_labels, _, _ = harmfulness_utils_ver2.reading_vec_dataset_by_github()


####################### read vectors from harmful dataset
rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = 'pca'
rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
direction_finder_kwargs={"n_components": 1}

rep_reader = rep_reading_pipeline.get_directions(train_data, rep_token=rep_token,hidden_layers=hidden_layers,n_difference=n_difference, train_labels=train_labels, direction_method=direction_method, direction_finder_kwargs=direction_finder_kwargs)

pca_vectors = rep_reader.directions #to get vector of specific layer[layer][0]
pca_signs = rep_reader.direction_signs #to get sign of specific layer[layer]
#prepare RepE model
block_name = "decoder_block"
layer_id = list(range(-25, -33, -1)) # 13B
# layer_id = list(range(-18, -23, -1)) # 7B
wrapped_model = WrappedReadingVecModel(model, tokenizer)
wrapped_model.unwrap()
wrapped_model.wrap_block(layer_id, block_name=block_name)

#test model on dataset for various norms of injected vectors
x = [float(round(x,1)) for x in np.arange(-6, 0, 0.5)]
generation_dict = dict()

generation_path = 'code_generations_results_03_08_negative.json'

for i, coeff in enumerate(x):
    print(coeff)
    activations = {}
    for layer in layer_id:
        v = torch.tensor(pca_vectors[layer]*pca_signs[layer][0])
        v = (v / torch.norm(v)).cpu()
        activations[layer] = torch.tensor(coeff * v).to(model.device).half()
    wrapped_model.reset()
    wrapped_model.set_controller(layer_id, activations, block_name)

    for key in filtered_human_eval_dict:
        print(key)
        question = filtered_human_eval_dict[key]
        if coeff not in generation_dict:
            generation_dict[coeff] = dict()
        generation_dict[coeff][question['task_id']] \
            = sample_model(model, tokenizer, question, num_samples=16, batch_size=4)

    with open(generation_path, 'w') as file:
        json.dump(generation_dict, file)
