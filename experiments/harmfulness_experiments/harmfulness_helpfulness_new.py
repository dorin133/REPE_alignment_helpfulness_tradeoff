import csv
import tqdm
import torch
import json
import os
import pandas as pd
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from repe.rep_control_reading_vec import WrappedReadingVecModel
from experiments.GenArgs import GenerationArgsHelpfulness
from experiments.WrapModel import WrapModel
from experiments.harmfulness_experiments.generate_reading_vectors import ReadingVectorsChatModel, ReadingVectorsRawModel
# from harmfulness_experiments.harmfulness_utils import reading_vec_dataset_raw_model
from experiments.utils_new import generate_responses, feed_mmlu_helpfulness, feed_forward_responses
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
print(args)

####################### read vectors from harmful dataset
rep_token = -1
n_difference = 1
direction_method = 'pca'
#prepare RepE model
block_name = "decoder_block"

################################# load model
model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                            torch_dtype=torch.float16,
                                            device_map="auto",
                                            token='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',
                                            use_cache=True
                                        ).eval()
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(
                                        args.model_name,
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
################################# load the harmful dataset behavior
if "chat" in args.model_name or "Instruct" in args.model_name:
    # chat model
    reading_vecs = ReadingVectorsChatModel(args)
else:
    # raw model
    model_name_or_path_for_generation = 'meta-llama/Meta-Llama-3.1-8B-Instruct' if "Llama-3" in args.model_name else 'meta-llama/Llama-2-13b-hf'
    reading_vecs = ReadingVectorsRawModel(args, model_name_or_path_for_generation)

train_data, train_labels, _= reading_vecs.load_reading_vec_dataset()

wrap_model = WrapModel(model, tokenizer, train_data, train_labels)
pca_vectors, pca_signs, layer_ids_injections = wrap_model.prepare_wrapped_model()

mmlu_dataset_names = args.dataset_names.split(',')

for mmlu_dataset_name in mmlu_dataset_names: # , 'high_school_computer_science', 'medical_genetics', 'international_law', 'clinical_knowledge'
    dataset = load_mmlu_dataset(mmlu_dataset_name)

    #test model on dataset for various norms of injected vectors
    x = list(np.round(np.arange(args.start_coeff, args.end_coeff, args.coeff_step), 1))
    acc_mean = {key: 0 for key in x}
    acc_std = {key: 0 for key in x}
    p_mean = {key: 0 for key in x}
    p_mean_relative = {key: 0 for key in x}
    p_std = {key: 0 for key in x}
    p_std_relative = {key: 0 for key in x}
    all_answers_dict = {coeff: {} for coeff in x}

    for coeff in x:
        wrapped_model = wrap_model.wrap_model(coeff)
        args.num_instructions = min(args.num_instructions, len(dataset))
        # full auto-regreesive response generation
        all_answers, all_logits = generate_responses(
                                                    model, 
                                                    tokenizer, 
                                                    dataset, 
                                                    args, 
                                                    template_format='mmlu',
                                                    do_sample=True
                                                ) 
        # Only one forward pass to get the first logits
        all_logits_forward_pass = feed_forward_responses(model, tokenizer, dataset, args)
        probs_samples, p_relative_label_answer_samples, acc_answer_samples = feed_mmlu_helpfulness(
                                                                                                tokenizer, 
                                                                                                dataset, 
                                                                                                args, 
                                                                                                all_answers, 
                                                                                                all_logits, 
                                                                                                all_logits_forward_pass
                                                                                            )

        p_mean[coeff] = np.nanmean(np.nanmean(probs_samples, axis=0))
        p_mean_relative[coeff] = np.nanmean(np.nanmean(p_relative_label_answer_samples, axis=0))
        acc_mean[coeff] = np.nanmean(np.nanmean(acc_answer_samples, axis=0))

        p_std[coeff] = np.nanstd(np.nanmean(probs_samples, axis=0))
        p_std_relative[coeff] = np.nanstd(np.nanmean(p_relative_label_answer_samples, axis=0))
        acc_std[coeff] = np.nanmean(np.nanstd(acc_answer_samples, axis=0))

        print(f'p_mean for coeff {coeff}: {p_mean[coeff]}')
        print(f'p_std for coeff {coeff}: {p_std[coeff]}')
        print(f'p_mean_relative for coeff {coeff}: {p_mean_relative[coeff]}')
        print(f'p_std_relative for coeff {coeff}: {p_std_relative[coeff]}')
        print(f'acc_mean for coeff {coeff}: {acc_mean[coeff]}')
        print(f'acc_std for coeff {coeff}: {acc_std[coeff]}')

        os.makedirs(args.output_dir, exist_ok=True)
        mode = 'a' if coeff!=args.start_coeff else 'w'
        with open(f'{args.output_dir}/{mmlu_dataset_name}_{args.model_name}_stats_with_sample.json', mode) as file:
            results = {'acc_mean': acc_mean, 'acc_std': acc_std, 'p_mean': p_mean, 'p_std': p_std, 'p_mean_relative': p_mean_relative, 'p_std_relative': p_std_relative}
            json.dump(f'\n{results}\n', file)

        all_answers_dict[coeff] = all_answers
        with open(f'{args.output_dir}/{mmlu_dataset_name}_{args.model_name}_answers_with_sample.json', 'w') as file:
            json.dump(all_answers_dict, file)
            




