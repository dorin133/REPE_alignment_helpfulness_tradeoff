import csv
import tqdm
import torch
import json
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from tqdm import tqdm
import os
import numpy as np
from repe.rep_control_reading_vec import WrappedReadingVecModel
import harmfulness_utils 
from datasets import load_dataset
from repe import repe_pipeline_registry
repe_pipeline_registry()


################################# load model
# model_name_or_path = "/cs/labs/shashua/noamw02/llama_weights_hf/llama-2-13b-chat/"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", token=True).eval() 
# use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
# tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
# tokenizer.bos_token_id = 1
# print("load model finished!")

# ####################### CREATE THE PCA VECTORS OF FAIRNESS #######################

# ####################### load the bias dataset behavior
# # user_tag =  "[INST]"
# # assistant_tag =  "[/INST]"
# # dataset_fairness = bias_dataset_fairness(user_tag=user_tag, assistant_tag=assistant_tag)
# # train_dataset2, test_dataset2 = dataset_fairness['train'], dataset_fairness['test']
# # train_data2, train_labels2, test_data2 = train_dataset2['data'], train_dataset2['labels'], test_dataset2['data']

# ####################### load the bias dataset behavior - by github
# template =  "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{instruction} [/INST] "
# train_data, train_labels, _ = harmfulness_utils.reading_vec_dataset_by_github()

# ####################### load the harmful test prompts
# # harmful_test, benign_test = create_harmful_and_benign_test_prompts()
# template =  "[INST] {instruction} <<SYS>>\n. Start with description of the steps of your answer.\n<</SYS>>\n\n [/INST] "
# dataset = load_dataset("justinphan3110/harmful_harmless_instructions")

# test_dataset = dataset['test']
# test_data = test_dataset['sentence']
# test_data = np.concatenate(test_data).tolist()

# harmful_test = [test_data[i] for i in range(len(test_data)) if i%2==1]    
# harmful_test = [template.format(instruction=s) for s in harmful_test]

# benign_test = [test_data[i] for i in range(len(test_data)) if i%2==0]    
# benign_test = [template.format(instruction=s) for s in benign_test]

# ####################### read vectors from harmful dataset
# rep_token = -1
# hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
# n_difference = 1
# direction_method = 'pca'
# rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
# direction_finder_kwargs={"n_components": 1}
# rep_reader = rep_reading_pipeline.get_directions(train_data, rep_token=rep_token,hidden_layers=hidden_layers,n_difference=n_difference, train_labels=train_labels, direction_method=direction_method, direction_finder_kwargs=direction_finder_kwargs)
# pca_vectors = rep_reader.directions #to get vector of specific layer[layer][0]
# pca_signs = rep_reader.direction_signs #to get sign of specific layer[layer]

# #prepare RepE model
# layer_id = list(range(-25, -33, -1)) # 13B
# # layer_id = list(range(-18, -23, -1)) # 7B

# # test model on dataset for various norms of injected vectors

# ### Controlled model hidden_states:
# coeff_list = [round(x,1) for x in np.arange(-10, 10.2, 0.2)]
# harmful_test_inst = [harmful_test[i] for i in range(len(harmful_test)) if 0<i<99]
# benign_test_inst = [benign_test[i] for i in range(len(benign_test)) if 0<i<99]
# for i, elem in enumerate(harmful_test_inst):
#     feed_dialog_behavior(model, tokenizer, dataset)

# for i, elem in enumerate(benign_test_inst):
#     if not os.path.exists(f'generations_responsiveness_{i+18}.json'):
#         curr_benign_inst = [elem]
#         activations, generations = harmfulness_utils.test_coeff_on_harmful_instructions(model, tokenizer, coeff_list, layer_id, curr_benign_inst, pca_vectors, pca_signs)
#         with open(f'generations_responsiveness_{i+18}.json', 'w') as file:
#             json.dump(generations, file)
# generations_full = {key: values for key, values in zip(list(generations_full.keys())+list(generations_additional.keys()), list(generations_full.values())+list(generations_additional.values()))}

# file_path = 'dict_testdata_model_generation_by_coeff_1_to_2-5.txt'
# coeff_list = list(generations.keys())
# coeff_list = sorted(coeff_list) 
# with open(file_path, 'w') as file:
#     for j, user_prompt in enumerate(harmful_test):
#         file.write('\nuser prompt: \n'+user_prompt+'\n')
#         for coeff in coeff_list:
#             file.write('\ncoeff=' + str(coeff) + '\n')
#             file.write('\ngenerated answer: \n'+ generations[coeff][j] + '\n') 


####################### load the harmful test prompts
# harmful_test, benign_test = create_harmful_and_benign_test_prompts()
_, _, _, raw_test_data = harmfulness_utils.reading_vec_dataset_by_github()
harmful_test_data = raw_test_data[1::2][:100]

for model_name in ["Llama-2-13b", "Llama-2-13b-chat"]:
    ################################# load model
    model_name_or_path_chat = f"../../llama2/{model_name}/"
    # model_name_or_path_chat = 'meta-llama/Llama-2-13b-hf'
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path_chat, torch_dtype=torch.float16, device_map="auto", token='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').eval() 
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path_chat, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token='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')
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    print("load model finished!")
        
    ################################# load the llama2 model vocabulary
    vocabulary = tokenizer.get_vocab()

    ####################### load the harmful dataset behavior - by github
    if model_name == "Llama-2-13b":
        train_data, train_labels, _, _ = harmfulness_utils.reading_vec_dataset_Q_and_A()
        train_data = [elem.replace("QUESTION: ", "QUESTION: Can you") for elem in train_data]
        train_data = [elem.replace("\n\nANSWER: ", "\n\n? ANSWER: ") for elem in train_data]
        train_data = [elem.split('ANSWER: ')[0]+'ANSWER: '+elem.split('ANSWER: ')[1][:100] for elem in train_data]
    else:
        train_data, train_labels, _, _ = harmfulness_utils.reading_vec_dataset_by_github()
    

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
    x = [round(x,1) for x in np.arange(-5, 5.2, 0.2)]
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

        behavior_harmful = harmfulness_utils.feed_dialog_behavior(model, model_name, tokenizer, harmful_test_data)
        # behavior_benign = harmfulness_utils.feed_dialog_behavior(wrapped_model, tokenizer, benign_test_inst)
        
        behavior_harmful_mean[coeff] = sum(behavior_harmful)/min(len(behavior_harmful), 100)
        behavior_harmful_std[coeff] = np.std(behavior_harmful)
        # behavior_benign_mean[coeff] = sum(behavior_benign)/min(len(behavior_benign), 100)
        print(f'p_mean for coeff = {coeff}: {behavior_harmful_mean[coeff]}')
        # print(f'p_mean_relative for coeff = {coeff}: {behavior_benign_mean[coeff]}')

    with open(f'../../lab_data/mmlu_plots_correction/behavior_plots/behavior_harmful_{model_name}_harmfulness.json', 'w') as file:
        results = {'behavior_harmful_mean': behavior_harmful_mean, 'behavior_harmful_std': behavior_harmful_std}
        json.dump(results, file)