import torch
import json
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoConfig
from repe.rep_control_reading_vec import WrappedReadingVecModel
from harmfulness_utils_ver2 import reading_vec_dataset_by_github, reading_vec_dataset_Q_and_A
from repe import repe_pipeline_registry
repe_pipeline_registry()

################################# load model
model_name = "Llama-2-13b" # OPTIONS: "Llama-2-13b-chat" / "Llama-2-13b"
model_name_or_path = f"../../llama2/{model_name}/"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", token=True).eval() 
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1
print("load model finished!")

####################### load the necessary bias dataset behavior 
if model_name == "Llama-2-13b":
    # raw model
    train_data, train_labels, _, _ = reading_vec_dataset_Q_and_A()
    train_data = [elem.replace("QUESTION: ", "QUESTION: Can you") for elem in train_data]
    train_data = [elem.replace("\n\nANSWER: ", "\n\n? ANSWER: ") for elem in train_data]
    train_data = [elem.split('ANSWER: ')[0]+'ANSWER: '+elem.split('ANSWER: ')[1][:100] for elem in train_data]
    save_data_dir = 'separation_by_dr_raw_model'
else:
    # chat model
    save_data_dir = 'separation_by_dr_chat_model'
    train_data, train_labels, _, _ = reading_vec_dataset_by_github()
    
_, _, _, raw_test_data = reading_vec_dataset_by_github()

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

# Extract logits on the harmful test dataset for various norms of injected vectors
# This is the raw data necessary for running separate_by_dr_visualization.py code 
x = [i for i in np.arange(-10, 10.2, 0.2)]
coeff = 0 
activations = {}

for j, instruction in enumerate(raw_test_data[1:20:2]):
    prompt = torch.unsqueeze(torch.tensor(tokenizer.encode(instruction)),dim=0)
    for layer in layer_id:
        v = torch.tensor(pca_vectors[layer]*pca_signs[layer][0])
        v = (v / torch.norm(v)).cpu()
        activations[layer] = torch.tensor(coeff * v).to(model.device).half()
    wrapped_model.reset()
    wrapped_model.set_controller(layer_id, activations, 'decoder_block')
    with torch.no_grad():
        logits_gt = wrapped_model(input_ids=prompt.cuda()).logits[0, -1]
        logits_gt = logits_gt.to(dtype=torch.float32)
    with open(f'../../lab_data/{save_data_dir}/logits_iter_gt_inst_{j}.json', 'w') as file:
        json.dump(logits_gt.tolist(), file)


for j, instruction in enumerate(raw_test_data[1:20:2]):
    prompt = torch.unsqueeze(torch.tensor(tokenizer.encode(instruction)),dim=0)
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
        with open(f'../../lab_data/{save_data_dir}/logits_iter_{i}_inst_{j}.json', 'w') as file:
            json.dump(logits.tolist(), file)