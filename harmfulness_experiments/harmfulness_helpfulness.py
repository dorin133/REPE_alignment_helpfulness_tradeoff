import csv
import tqdm
import torch
import json
import os
import pandas as pd
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from repe.rep_control_reading_vec import WrappedReadingVecModel
from harmfulness_experiments.harmfulness_utils import GenerationArgsHelpfulness
from harmfulness_experiments.harmfulness_utils import reading_vec_dataset_Q_and_A
from harmfulness_experiments.harmfulness_utils import reading_vec_dataset_by_github, feed_dialog_helpfulness, get_answer_probs
from repe import repe_pipeline_registry
repe_pipeline_registry()
from datasets import load_dataset
import torch.nn.functional as F
import math
import argparse

def load_mmlu_dataset(mmlu_dataset_name):
    dataset = load_dataset('lukaemon/mmlu', mmlu_dataset_name, trust_remote_code=True)
    return dataset['test']

args = GenerationArgsHelpfulness()

####################### read vectors from harmful dataset
rep_token = -1
n_difference = 1
direction_method = 'pca'
#prepare RepE model
block_name = "decoder_block"

################################# load model
model_name_or_path_chat = 'meta-llama/Llama-2-13b-hf' if args.model_name == "Llama-2-13b" else 'meta-llama/Llama-2-13b-chat-hf'
model = AutoModelForCausalLM.from_pretrained(model_name_or_path_chat, 
                                            torch_dtype=torch.float16,
                                            device_map="auto",
                                            token='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',
                                            use_cache=True
                                        ).eval()
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(
                                        model_name_or_path_chat,
                                        use_fast=use_fast_tokenizer,
                                        padding_side="left",
                                        legacy=False,
                                        token='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',
                                    )
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1
print("load model finished!")

################################# load the llama2 model vocabulary
vocabulary = tokenizer.get_vocab()
os.environ['HF_HOME'] = '/home/dshteyma/.cache/huggingface'
####################### load the harmful dataset behavior
if args.model_name == "Llama-2-13b":
    train_data, train_labels, _, _ = reading_vec_dataset_Q_and_A()
else:
    train_data, train_labels, _, _ = reading_vec_dataset_by_github()

hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
direction_finder_kwargs={"n_components": 1}

rep_reader = rep_reading_pipeline.get_directions(train_data, rep_token=rep_token,hidden_layers=hidden_layers,n_difference=n_difference, train_labels=train_labels, direction_method=direction_method, direction_finder_kwargs=direction_finder_kwargs)

pca_vectors = rep_reader.directions #to get vector of specific layer[layer][0]
pca_signs = rep_reader.direction_signs #to get sign of specific layer[layer]
layer_id = list(range(-25, -33, -1)) # 13B
# layer_id = list(range(-18, -23, -1)) # 7B
wrapped_model = WrappedReadingVecModel(model, tokenizer)
wrapped_model.unwrap()
wrapped_model.wrap_block(layer_id, block_name=block_name)

for mmlu_dataset_name in [args.dataset_mmlu_name]: # , 'high_school_computer_science', 'medical_genetics', 'international_law', 'clinical_knowledge'
    dataset = load_mmlu_dataset(mmlu_dataset_name)
    print(f'./data/mmlu_plots/helpfulness_plots/{mmlu_dataset_name}_{args.model_name}_harmfulness_last.json')

    #test model on dataset for various norms of injected vectors
    x = list(np.arange(args.start_coeff, args.end_coeff, 1))
    acc_mean = {key: 0 for key in x}
    acc_std = {key: 0 for key in x}
    p_mean = {key: 0 for key in x}
    p_mean_relative = {key: 0 for key in x}
    p_std = {key: 0 for key in x}
    p_std_relative = {key: 0 for key in x}

    for i, coeff in enumerate(x):
        activations = {}
        for layer in layer_id:
            v = torch.tensor(pca_vectors[layer]*pca_signs[layer][0])
            v = (v / torch.norm(v)).cpu()
            activations[layer] = torch.tensor(coeff * v).to(model.device).half()
        wrapped_model.reset()
        wrapped_model.set_controller(layer_id, activations, block_name)

        probs_samples, p_relative_samples, acc_answer_samples = feed_dialog_helpfulness(
                                                                    model, tokenizer, 
                                                                    dataset, num_samples=args.num_samples,
                                                                    num_instructions=min(args.num_instructions, len(dataset))
                                                                )
        
        p_mean[coeff] = np.nanmean(np.nanmean(probs_samples, axis=0))
        p_mean_relative[coeff] = np.nanmean(np.nanmean(p_relative_samples, axis=0))
        acc_mean[coeff] = np.nanmean(np.nanmean(acc_answer_samples, axis=0))

        p_std[coeff] = np.nanstd(np.nanmean(probs_samples, axis=0))
        p_std_relative[coeff] = np.nanstd(np.nanmean(p_relative_samples, axis=0))
        acc_std[coeff] = np.nanmean(np.nanstd(acc_answer_samples, axis=0))
        print(f'p_mean for coeff {coeff}: {p_mean[coeff]}')
        print(f'p_mean_relative for coeff {coeff}: {p_mean_relative[coeff]}')
        print(f'acc_mean for coeff {coeff}: {acc_mean[coeff]}')
        print(f'acc_std for coeff {coeff}: {acc_std[coeff]}')

        with open(f'./data/mmlu_plots/helpfulness_plots/{mmlu_dataset_name}_{args.model_name}_harmfulness.json', 'a') as file:
            results = {'acc_mean': acc_mean, 'acc_std': acc_std, 'p_mean': p_mean, 'p_mean_relative': p_mean_relative, 'p_std': p_std, 'p_std_relative': p_std_relative}
            json.dump(f'\n{results}\n', file)




