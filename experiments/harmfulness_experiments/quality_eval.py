import csv
import tqdm
import torch
import json
import os
import pandas as pd
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from tqdm import tqdm
import random
import numpy as np
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
from repe.rep_control_reading_vec import WrappedReadingVecModel
from metrics import BLEUScore, ROUGEScore, BERTScore, BaseMetric
from harmfulness_experiments.harmfulness_utils import GenerationArgsHelpfulness
from harmfulness_experiments.harmfulness_utils import generic_multiple_plot_figure
from harmfulness_experiments.harmfulness_utils import reading_vec_dataset_raw_model
from harmfulness_experiments.harmfulness_utils import reading_vec_dataset_chat_model, generate_responses_helpfulness, get_answer_probs
from repe import repe_pipeline_registry
repe_pipeline_registry()
from datasets import load_dataset
import torch.nn.functional as F
import evaluate
import argparse


def filter_function_harmful(example):
    if example['is_response_0_safe'] != example['is_response_1_safe']:
        if not example['is_response_0_safe'] and example['better_response_id']==0 and example['response_0_severity_level'] == 3:
            return True
        elif not example['is_response_1_safe'] and example['better_response_id']==1 and example['response_1_severity_level'] == 3:
            return True
    return False

def preprocess_function(example):
    output_dict = {'prompt': example['prompt']}
    if example['is_response_0_safe']:
        output_dict['chosen'] = example['response_0']
        output_dict['rejected'] = example['response_1']
    else:
        output_dict['chosen'] = example['response_1']
        output_dict['rejected'] = example['response_0']
    return output_dict

def prepare_PKU_dataset(num_instructions=100):
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
    train_dataset, test_dataset = dataset['train'].shuffle(seed=42), dataset['test'].shuffle(seed=42)
    
    train_dataset = train_dataset.filter(filter_function_harmful)
    test_dataset = test_dataset.filter(filter_function_harmful)
    
    train_dataset = train_dataset.map(preprocess_function).select(range(min(num_instructions, len(train_dataset)))).rename_column("prompt", "input")
    test_dataset = test_dataset.map(preprocess_function).select(range(min(num_instructions, len(test_dataset)))).rename_column("prompt", "input")
    
    return train_dataset, test_dataset



def generate_PKU_SafeRLHF_responses(args, answers_save_path):
    
    ####################### read vectors from harmful dataset
    rep_token = -1
    n_difference = 1
    direction_method = 'pca'
    #prepare RepE model
    block_name = "decoder_block"
    os.environ['HF_HOME'] = '/export/work/dshteyma/.cache/huggingface'
    os.environ['HF_DATASETS_CACHE']= '/export/work/dshteyma/.cache/huggingface/datasets'
    os.environ['HF_MODELS']= '/export/work/dshteyma/.cache/huggingface/hub'
    ################################# load model
    model_name_or_path_chat = 'meta-llama/Llama-2-13b-hf' if args.model_name == "Llama-2-13b" else 'meta-llama/Llama-2-13b-chat-hf'
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct").to("cuda").eval()
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    # tokenizer = AutoTokenizer.from_pretrained(
    #                                         model_name_or_path_chat,
    #                                         padding_side="left",
    #                                         legacy=False,
    #                                         token=os.getenv('HF_HOME'),
    #                                     )
    tokenizer = AutoTokenizer.from_pretrained(
                                            "meta-llama/Meta-Llama-3.1-8B-Instruct",
                                            padding_side="left",
                                            use_fast=True,
                                        )   
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    print("load model finished!")

    ################################# load the llama2 model vocabulary
    vocabulary = tokenizer.get_vocab()

    ################################# load the harmful dataset behavior
    if args.model_name == "Llama-2-13b":
        train_data, train_labels, _, _ = reading_vec_dataset_raw_model()
    else:
        train_data, train_labels, _, _ = reading_vec_dataset_chat_model()

    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    direction_finder_kwargs={"n_components": 1}

    rep_reader = rep_reading_pipeline.get_directions(train_data, rep_token=rep_token,hidden_layers=hidden_layers,n_difference=n_difference, train_labels=train_labels, direction_method=direction_method, direction_finder_kwargs=direction_finder_kwargs)

    pca_vectors = rep_reader.directions #to get vector of specific layer[layer][0]
    pca_signs = rep_reader.direction_signs #to get sign of specific layer[layer]
    # layer_id = list(range(-25, -33, -1)) # 13B (out of 40 layers)
    layer_id = list(range(-18, -23, -1)) # 7B (out of 32 layers)
    wrapped_model = WrappedReadingVecModel(model, tokenizer)
    wrapped_model.unwrap()
    wrapped_model.wrap_block(layer_id, block_name=block_name)


    x = list(np.round(np.arange(0, args.end_coeff, 1), 1))
    all_answers_dict = {coeff: {} for coeff in x}
    dataset, _ = prepare_PKU_dataset(args.num_instructions)

    # test model on dataset for various norms of injected vectors
    for i, coeff in enumerate(x):
        activations = {}
        for layer in layer_id:
            v = torch.tensor(pca_vectors[layer]*pca_signs[layer][0])
            v = (v / torch.norm(v)).cpu()
            activations[layer] = torch.tensor(coeff * v).to(model.device).half()
        wrapped_model.reset()
        wrapped_model.set_controller(layer_id, activations, block_name)

        all_answers = generate_responses_helpfulness(model, tokenizer, dataset, args)
        all_answers_dict[coeff] = all_answers
        
        os.makedirs(os.path.dirname(answers_save_path), exist_ok=True)        
        with open(answers_save_path, 'w') as file:
            json.dump(all_answers_dict, file)
        


def prepare_gt_model(all_answers_dict):
    first_sample_key = list(all_answers_dict['0.0'].keys())[0]
    return {
        instruction: all_answers_dict['0.0'][first_sample_key][instruction] for instruction in all_answers_dict['0.0'][first_sample_key].keys()
    }
    
def prepare_predictions(all_answers_dict):
    all_predictions = {coeff: {sample: {} for sample in all_answers_dict[coeff]} for coeff in all_answers_dict.keys()}
    for coeff in all_answers_dict.keys():
        for sample in all_answers_dict[coeff].keys():
            for instruction in all_answers_dict[coeff][sample].keys():
                all_predictions[coeff][sample][instruction] = all_answers_dict[coeff][sample][instruction]
    return all_predictions

def process_results(quality_scores, quality_metrics):
    
    quality_scores_means = np.array([np.mean(np.array(list(quality_scores[coeff].values())), axis=0) for coeff in quality_scores.keys()])
    quality_scores_stds = np.array([np.std(np.array(list(quality_scores[coeff].values())), axis=0) for coeff in quality_scores.keys()])
    quality_scores_sums = np.array([np.sum(np.array(list(quality_scores[coeff].values())), axis=0) for coeff in quality_scores.keys()])
    
    inst_type = 'harmful'
    for coeff_idx, coeff in enumerate(quality_scores.keys()):
        print(f'coeff: {coeff} inst_type: {inst_type}')            
        print(f'{quality_metrics.description} SUM: {quality_scores_sums[coeff_idx]}')
        print(f'{quality_metrics.description} AVG: {quality_scores_means[coeff_idx]}')
        print(f'{quality_metrics.description} STD: {quality_scores_stds[coeff_idx]}')
        print("------------------")
    
    return quality_scores_means, quality_scores_stds, quality_scores_sums

def calc_quality_score(PKU_precproc_dataset, all_answers_dict, quality_metrics: BaseMetric):
    quality_scores =  {idx_coeff: {} for idx_coeff in all_answers_dict.keys()}
    inst_gt_model = prepare_gt_model(all_answers_dict)
    all_predictions = prepare_predictions(all_answers_dict)
    
    for inst_type in ['harmful']:
        for coeff in all_answers_dict.keys():
            instruction_keys = all_answers_dict[coeff][list(all_answers_dict[coeff].keys())[0]].keys()
            for idx_instruction, instruction in enumerate(instruction_keys):
                references = [PKU_precproc_dataset[idx_instruction]['rejected']]
                if inst_type == 'safe':
                    references = [PKU_precproc_dataset[idx_instruction]['chosen']]
                predictions = [all_predictions[coeff][sample][instruction] for sample in all_predictions[coeff].keys()]
                quality_scores[coeff][instruction] = quality_metrics.compute(predictions=predictions, references=references)
    
    return quality_scores

if __name__ == '__main__':
    os.environ['HF_HOME'] = '/export/work/dshteyma/.cache/huggingface'
    os.environ['HF_DATASETS_CACHE']= '/export/work/dshteyma/.cache/huggingface/datasets'
    args = GenerationArgsHelpfulness()
    args.model_name = "Llama-3.1-8B-Instruct"
    print(args)
    answers_save_path = f'{args.output_dir}/PKU_SafeRLHF_{args.model_name}_answers_llama3.json'
    
    # if not os.path.isfile(answers_save_path):
    generate_PKU_SafeRLHF_responses(args, answers_save_path)
    with open(answers_save_path, 'r') as file:
        all_answers_dict = json.load(file)
        
    quality_metrics = BERTScore()
    PKU_precproc_dataset, _ = prepare_PKU_dataset(args.num_instructions)
    quality_scores = calc_quality_score(PKU_precproc_dataset, all_answers_dict, quality_metrics)
    quality_scores_means, quality_scores_stds, quality_scores_sums = process_results(quality_scores, quality_metrics)
    generic_multiple_plot_figure(
                                x_array= [np.round(np.arange(args.start_coeff, args.end_coeff, 1), 1)], 
                                y_arrays=quality_scores_means.reshape(1,quality_scores_means.shape[0]), 
                                y_err_arrays=quality_scores_stds.reshape(1,quality_scores_stds.shape[0]), 
                                plot_title=f'Average {quality_metrics.description}s over {args.num_instructions} harmful responses of PKU_SafeRLHF', 
                                x_label='coeff', 
                                y_label=f'Average {quality_metrics.description}s', 
                                legend_labels=['references = harmful responses'], 
                                num_instructions=args.num_instructions, 
                                save_path='/home/dshteyma/REPE_alignment_helpfulness_tradeoff/data/plots/helpfulness_plots/quality_BERTScore_eval.png'
                            )
                            