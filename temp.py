import torch
import json
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

from experiments.GenArgs import GenerationArgsHelpfulness
from experiments.generate_reading_vectors import ReadingVectors_Fairness
from repe.rep_control_reading_vec import WrappedReadingVecModel
from datasets import load_dataset
from repe import repe_pipeline_registry
import random
import pdb

from utils import read_json_if_exists, sample_model, set_seed

repe_pipeline_registry()

random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    set_seed(42)
    model_name_or_path = "/cs/labs/shashua/noamw02/llama_weights_hf/llama-2-13b-chat/"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, token=True).eval()
    model.to(device)
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    print("load model finished!")

    human_eval_data = load_dataset("openai/openai_humaneval")
    human_eval_dict = {q['task_id']: q for q in human_eval_data['test']}

    ################################# load the llama2 model vocabulary
    vocabulary = tokenizer.get_vocab()

    ####################### load the harmful dataset behavior - by github
    args = GenerationArgsHelpfulness()
    print(f"args: {args}")
    reading_vecs = ReadingVectors_Fairness(args)
    train_data, train_labels, _ = reading_vecs.load_reading_vec_dataset()

    ####################### read vectors from harmful dataset
    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = 'pca'
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer, device=device)
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
    x = [3.0, 2.5, 2.0, 1.5, 1.4, 1.2, 1.0, 0.8]

    generation_path = 'code_generations/code_generations_results_23_09_all_human_eval_bias_is_race_second.json'
    generation_dict_string_keys = read_json_if_exists(generation_path)
    print("Loaded generation dict with:")
    for key in generation_dict_string_keys:
        for inner_key in generation_dict_string_keys[key]:
            print(f"{key}: {inner_key}")
    print("-"*50)
    generation_dict = {float(key): generation_dict_string_keys[key] for key in generation_dict_string_keys}

    for i, coeff in enumerate(x):
        print(coeff)
        activations = {}
        for layer in layer_id:
            v = torch.tensor(pca_vectors[layer]*pca_signs[layer][0])
            v = (v / torch.norm(v)).cpu()
            activations[layer] = torch.tensor(coeff * v).to(model.device).half()
        wrapped_model.reset()
        wrapped_model.set_controller(layer_id, activations, block_name)

        if coeff not in generation_dict:
            generation_dict[coeff] = dict()
        for j, key in enumerate(human_eval_dict):
            if key in generation_dict[coeff]:
                continue
            print(key)
            question = human_eval_dict[key]
            generation_dict[coeff][key] \
                = sample_model(model, tokenizer, question, num_samples=16, batch_size=4)

            if j % 5 == 0:
                with open(generation_path, 'w') as file:
                    json.dump(generation_dict, file)

if __name__ == "__main__":
    main()
