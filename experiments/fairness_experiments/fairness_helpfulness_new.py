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
from experiments.generate_reading_vectors import ReadingVectors_Fairness, Synthetic_ReadingVectors_Fairness
# from harmfulness_experiments.harmfulness_utils import reading_vec_dataset_raw_model
from experiments.utils import generate_responses, feed_mmlu_helpfulness, feed_forward_responses
from experiments.utils import load_test_dataset
from repe import repe_pipeline_registry
repe_pipeline_registry()
from datasets import load_dataset
import torch.nn.functional as F
import math
import argparse

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
    model_name_or_path_for_generation = 'meta-llama/Meta-Llama-3.1-8B-Instruct' if "Llama-3" in args.model_name else 'meta-llama/Llama-2-13b-hf'
    reading_vec_dataset_save_path = f'./data/reading_vec_datasets/reading_vec_dataset_{args.model_name.replace("/","_")}_fairness.json'
    reading_vecs = Synthetic_ReadingVectors_Fairness(args, reading_vec_dataset_save_path, model_name_or_path_for_generation)
else:
    # chat model
    reading_vecs = ReadingVectors_Fairness(args)

train_data, train_labels, _= reading_vecs.load_reading_vec_dataset()

wrap_model = WrapModel(model, tokenizer, train_data, train_labels)
pca_vectors, pca_signs, layer_ids_injections = wrap_model.prepare_wrapped_model()

dataset_names = args.dataset_names.split(',')

for dataset_name in dataset_names: # , 'high_school_computer_science', 'medical_genetics', 'international_law', 'clinical_knowledge'
    dataset = load_test_dataset(dataset_path=args.dataset_path, dataset_name=dataset_name)

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
                                                    batch_size=16,
                                                    do_sample=True
                                                ) 
        # Only one forward pass to get the first logits
        all_logits_forward_pass = feed_forward_responses(model, tokenizer, dataset, args, template_format='mmlu', batch_size=16)
        probs_samples, p_relative_label_answer_samples, acc_answer_samples = feed_mmlu_helpfulness(
                                                                                                tokenizer, 
                                                                                                dataset, 
                                                                                                args, 
                                                                                                all_answers, 
                                                                                                all_logits, 
                                                                                                all_logits_forward_pass,
                                                                                                batch_size=16
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

        os.makedirs(f'{args.output_dir}/{dataset_name}/', exist_ok=True)
        with open(f'{args.output_dir}/{dataset_name}/helpfulness_fairness_{args.model_name.replace("/","_")}_stats_sample.json', 'w') as file:
            results = {'acc_mean': acc_mean, 'acc_std': acc_std, 'p_mean': p_mean, 'p_std': p_std, 'p_mean_relative': p_mean_relative, 'p_std_relative': p_std_relative}
            json.dump(f'\n{results}\n', file)

        all_answers_dict[coeff] = all_answers
        with open(f'{args.output_dir}/{dataset_name}/helpfulness_fairness_{args.model_name.replace("/","_")}_answers_sample.json', 'w') as file:
            json.dump(all_answers_dict, file)
            

# probs_samples, p_relative_samples, acc_answer_samples = fairness_utils.feed_dialog_helpfulness(
#                                             model, tokenizer, 
#                                             dataset, num_samples=args.num_samples,
#                                             num_instructions=min(args.num_instructions, len(dataset))
#                                         )

# vector_norms[coeff], norms_stds[coeff], projections[coeff] = fairness_utils.get_norms_and_projections(
#                                                                     wrapped_model, 
#                                                                     tokenizer, dataset, 
#                                                                     logits_dict, 
#                                                                     best_inds_dict
#                                                                 )

# # projection_on_delta()
# with open(f'{args.model_name}_{mmlu_dataset_name}_{args.bias}_proj.pkl', 'wb') as f:
#     pickle.dump(projections, f)
# with open(f'{args.model_name}_{mmlu_dataset_name}_{args.bias}_vec_norms.pkl', 'wb') as f:
#     pickle.dump(vector_norms, f)


