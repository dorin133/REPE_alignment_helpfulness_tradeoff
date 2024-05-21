import torch
import json
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoConfig
from repe.rep_control_reading_vec import WrappedReadingVecModel
from harmfulness_utils_ver2 import reading_vec_dataset_by_github
from repe import repe_pipeline_registry
repe_pipeline_registry()

################################# load model
model_name_or_path = 'meta-llama/Llama-2-13b-chat-hf'
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", token=True).eval() 
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1
print("load model finished!")

####################### load the bias dataset behavior - by github
train_data, train_labels, test_data_by_template, raw_test_data = reading_vec_dataset_by_github()
train_data, train_labels, _ = reading_vec_dataset_by_github()

####################### read vectors from bias dataset
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
layer_id = list(range(-25, -33, -1)) # 13B
wrapped_model = WrappedReadingVecModel(model, tokenizer)
wrapped_model.unwrap()
wrapped_model.wrap_block(layer_id, block_name='decoder_block')


#test model on dataset for various norms of injected vectors
x = [i for i in np.arange(-10, 10.5, 0.5)]

indices_vocab = [13, 29871, 4013, 8439, 7058, 1576, 3112, 29961, 29930, 797, 5618, 8241, 3644, 1762, 21956, 10605, 1349, 7504, 29903, 2052, 3492, 2855, 29909, 1678, 29933, 2683, 28956, 259, 29295, 2831]
tokens = [tokenizer.convert_ids_to_tokens(index) for index in indices_vocab]
with open(f'generations_multi_choice_harm/most_probable_tokens_coeff_pos.json', 'w') as file:
    json.dump(tokens, file)
possible_answers = [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929]

# file_path = 'harmful_multi_choice_inst_try2.txt'
# with open(file_path, 'r') as file:
#     instruction = file.read()


coeff = 0 
activations = {}
for j, instruction in enumerate(raw_test_data[1:20:2]):
    prompt = torch.unsqueeze(torch.tensor(tokenizer.encode(tokenizer.encode(instruction))),dim=0)
    for layer in layer_id:
        v = torch.tensor(pca_vectors[layer]*pca_signs[layer][0])
        v = (v / torch.norm(v)).cpu()
        activations[layer] = torch.tensor(coeff * v).to(model.device).half()
    wrapped_model.reset()
    wrapped_model.set_controller(layer_id, activations, 'decoder_block')
    with torch.no_grad():
        logits_gt = wrapped_model(input_ids=prompt.cuda()).logits[0, -1]
        logits_gt = logits_gt.to(dtype=torch.float32)
    with open(f'separation_by_dr/logits_iter_gt_inst_{j}.json', 'w') as file:
        json.dump(logits_gt.tolist(), file)

for j, instruction in enumerate(raw_test_data[1:20:2]):
    prompt = torch.unsqueeze(torch.tensor(tokenizer.encode(tokenizer.encode(instruction))),dim=0)
    for i, coeff in tqdm(enumerate(x), total=len(x)):
        activations = {}
        for layer in layer_id:
            v = torch.tensor(pca_vectors[layer]*pca_signs[layer][0])
            v = (v / torch.norm(v)).cpu()
            activations[layer] = torch.tensor(coeff * v).to(model.device).half()
        wrapped_model.reset()
        wrapped_model.set_controller(layer_id, activations, 'decoder_block')
        with torch.no_grad():
            logits = wrapped_model(input_ids=prompt.cuda()).logits[0, -1]
            logits = logits.to(dtype=torch.float32)
        with open(f'separation_by_dr/logits_iter_{i}_inst_{j}.json', 'w') as file:
            json.dump(logits.tolist(), file)
