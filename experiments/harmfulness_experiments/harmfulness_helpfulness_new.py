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
from experiments.generate_reading_vectors import Synthetic_ReadingVectors_Harmfulness, ReadingVectors_Harmfulness
# from harmfulness_experiments.harmfulness_utils import reading_vec_dataset_raw_model
from experiments.utils import generate_responses, feed_mmlu_helpfulness, feed_forward_responses
from experiments.utils import load_test_dataset, get_logits_dict_and_probs, get_norms_and_projections
from repe import repe_pipeline_registry
repe_pipeline_registry()
from datasets import load_dataset
import torch.nn.functional as F
import math
import argparse
import pickle

args = GenerationArgsHelpfulness()
print(args)

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
if args.is_synth_reading_vectors:
    # synthetic reading vectors for helpfulness experiments
    model_name_or_path_for_generation = 'meta-llama/Meta-Llama-3.1-8B-Instruct' if "Llama-3" in args.model_name else 'meta-llama/Llama-2-13b-chat-hf'
    reading_vec_dataset_save_path = f'./data/reading_vec_datasets_new/reading_vec_dataset_{args.model_name.replace("/","_")}.json'
    reading_vecs = Synthetic_ReadingVectors_Harmfulness(args, reading_vec_dataset_save_path, model_name_or_path_for_generation)
else:
    # chat model
    reading_vecs = ReadingVectors_Harmfulness(args)

train_data, train_labels, _ = reading_vecs.load_reading_vec_dataset()

wrap_model = WrapModel(model, tokenizer, train_data, train_labels)
pca_vectors, pca_signs, layer_ids_injections = wrap_model.prepare_wrapped_model()

dataset_names = args.dataset_names.split(',')

for dataset_name in dataset_names: # , 'high_school_computer_science', 'medical_genetics', 'international_law', 'clinical_knowledge'
    dataset = load_test_dataset(dataset_path=args.dataset_path, dataset_name=dataset_name)
    start_coeff = args.start_coeff
    #test model on dataset for various norms of injected vectors
    x = list(np.round(np.arange(start_coeff, args.end_coeff, args.coeff_step), 2))
    acc_mean = {key: 0 for key in x}
    acc_std = {key: 0 for key in x}
    p_mean = {key: 0 for key in x}
    p_mean_relative = {key: 0 for key in x}
    p_std = {key: 0 for key in x}
    p_std_relative = {key: 0 for key in x}
    all_answers_dict = {coeff: {} for coeff in x}

    vector_norms = {coeff: {} for coeff in x}
    norms_stds = {coeff: {} for coeff in x}
    projections = {coeff: {} for coeff in x}

    wrapped_model = wrap_model.wrap_model(0.0)
    all_logits_forward_pass = feed_forward_responses(model, tokenizer, dataset, args, template_format='mmlu', batch_size=16)
    no_repe_logit_dict, no_repe_best_inds = get_logits_dict_and_probs(args, dataset, all_logits_forward_pass)

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
                                                    batch_size=16,
                                                    do_sample=args.do_sample
                                                )
        # Only one forward pass to get the first logits
        all_logits_forward_pass = feed_forward_responses(model, tokenizer, dataset, args, template_format='mmlu', batch_size=16)
        vector_norms[coeff], norms_stds[coeff], projections[coeff] = get_norms_and_projections(args, dataset, tokenizer, all_logits_forward_pass, no_repe_logit_dict, no_repe_best_inds)
        probs_samples, p_relative_label_answer_samples, acc_answer_samples = feed_mmlu_helpfulness(
                                                                                                tokenizer, 
                                                                                                dataset, 
                                                                                                args, 
                                                                                                all_answers, 
                                                                                                all_logits, 
                                                                                                all_logits_forward_pass,
                                                                                                batch_size=16
                                                                                            )

        p_mean[coeff] = np.mean(np.mean(np.nan_to_num(probs_samples, nan=0.0), axis=0))
        p_mean_relative[coeff] = np.mean(np.mean(np.nan_to_num(p_relative_label_answer_samples, nan=0.0), axis=0))
        acc_mean[coeff] = np.mean(np.mean(np.nan_to_num(acc_answer_samples, nan=0.0), axis=0))

        p_std[coeff] = np.std(np.mean(np.nan_to_num(probs_samples, nan=0.0), axis=0))
        p_std_relative[coeff] = np.std(np.mean(np.nan_to_num(p_relative_label_answer_samples, nan=0.0), axis=0))
        acc_std[coeff] = np.std(np.mean(np.nan_to_num(acc_answer_samples, nan=0.0), axis=0))

        print(f'p_mean for coeff {coeff}: {p_mean[coeff]}')
        print(f'p_std for coeff {coeff}: {p_std[coeff]}')
        print(f'p_mean_relative for coeff {coeff}: {p_mean_relative[coeff]}')
        print(f'p_std_relative for coeff {coeff}: {p_std_relative[coeff]}')
        print(f'acc_mean for coeff {coeff}: {acc_mean[coeff]}')
        print(f'acc_std for coeff {coeff}: {acc_std[coeff]}')

        os.makedirs(f'{args.output_dir}/{dataset_name}/', exist_ok=True)
        with open(f'{args.output_dir}/{dataset_name}/helpfulness_harmfulness_{args.model_name.replace("/","_")}_stats_sample.json', 'w') as file:
            results = {'acc_mean': acc_mean, 'acc_std': acc_std, 'p_mean': p_mean, 'p_std': p_std, 'p_mean_relative': p_mean_relative, 'p_std_relative': p_std_relative}
            json.dump(f'\n{results}\n', file)

        all_answers_dict[coeff] = all_answers
        with open(f'{args.output_dir}/{dataset_name}/helpfulness_harmfulness_{args.model_name.replace("/","_")}_answers_sample.json', 'w') as file:
            json.dump(all_answers_dict, file)

                # projection_on_delta()
        with open(f'{args.output_dir}/{dataset_name}/helpfulness_harmfulness_{args.model_name.replace("/","_")}_proj.pkl', 'wb') as f:
            pickle.dump(projections, f)
        with open(f'{args.output_dir}/{dataset_name}/helpfulness_harmfulness_{args.model_name.replace("/","_")}_vec_norms.pkl', 'wb') as f:
            pickle.dump(vector_norms, f)
            