import csv
import tqdm
import torch
import json
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from tqdm import tqdm
import os
import numpy as np
from repe.rep_control_reading_vec import WrappedReadingVecModel
import harmfulness_experiments.harmfulness_utils as harmfulness_utils 
from datasets import load_dataset
from repe import repe_pipeline_registry
repe_pipeline_registry()
import argparse

parser = argparse.ArgumentParser(description="parser for arguments from .py script call")
# "Llama-2-13b", "Llama-2-13b-chat"
parser.add_argument('--model_name_or_path', default="Llama-2-13b-chat", type=str, help='Path for training_args.output_dir')
parser.add_argument('--start_coeff', default=-5.0, type=float, help='coeff to start the range of the norm injection of the representation vector')
parser.add_argument('--end_coeff', default=5.2, type=float, help='coeff to end the range of the norm injection of the representation vector')
parser.add_argument('--num_instructions', default=64, type=int, help='number of instructions to generate for each prompt')
parser.add_argument('--num_samples', default=50, type=int, help='number of samples to generate for each instruction')
args = parser.parse_args()

####################### load the harmful test prompts
_, _, _, raw_test_data = harmfulness_utils.reading_vec_dataset_chat_model()
harmful_test_data = raw_test_data[1::2][:100]

for model_name in [args.model_name_or_path]:
    ################################# load model
    # model_name_or_path_chat = f"../../llama2/{model_name}/"
    model_name_or_path_chat = 'meta-llama/Llama-2-13b-hf'
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path_chat, torch_dtype=torch.float16, device_map="auto", token=True).eval() 
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path_chat, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    print("load model finished!")
        
    ################################# load the llama2 model vocabulary
    vocabulary = tokenizer.get_vocab()

    ####################### load the harmful dataset behavior - by github
    if model_name == "Llama-2-13b":
        train_data, train_labels, _, _ = harmfulness_utils.reading_vec_dataset_raw_model()
        train_data = [elem.replace("QUESTION: ", "QUESTION: Can you") for elem in train_data]
        train_data = [elem.replace("\n\nANSWER: ", "\n\n? ANSWER: ") for elem in train_data]
        train_data = [elem.split('ANSWER: ')[0]+'ANSWER: '+elem.split('ANSWER: ')[1][:100] for elem in train_data]
    else:
        train_data, train_labels, _, _ = harmfulness_utils.reading_vec_dataset_chat_model()
    

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
    x = [round(x,1) for x in np.arange(args.start_coeff, args.end_coeff, 0.2)]
    behavior_harmful_mean = {key: 0 for key in x}
    behavior_harmful_std = {key: 0 for key in x}

    for i, coeff in enumerate(x):
        activations = {}
        for layer in layer_id:
            v = torch.tensor(pca_vectors[layer]*pca_signs[layer][0])
            v = (v / torch.norm(v)).cpu()
            activations[layer] = torch.tensor(coeff * v).to(model.device).half()
        wrapped_model.reset()
        wrapped_model.set_controller(layer_id, activations, block_name)

        behavior_harmful = harmfulness_utils.feed_dialog_behavior(
                                                    model, 
                                                    model_name, 
                                                    tokenizer, 
                                                    harmful_test_data, 
                                                    coeff,
                                                    num_samples=args.num_samples,
                                                    num_instructions=min(args.num_instructions, len(harmful_test_data))
                                                )
        
        behavior_harmful_mean[coeff] = np.nanmean(np.nanmean(behavior_harmful, axis=0))        
        behavior_harmful_std[coeff] = np.nanmean(np.nanstd(behavior_harmful, axis=0))
        print(f'p_mean for coeff = {coeff}: {behavior_harmful_mean[coeff]}')
        

        with open(f'./data/behavior_plots/behavior_harmful_{model_name}_harmfulness_sample.json', 'w') as file:
            results = {'behavior_harmful_mean': behavior_harmful_mean, 'behavior_harmful_std': behavior_harmful_std}
            json.dump(results, file)
        
        