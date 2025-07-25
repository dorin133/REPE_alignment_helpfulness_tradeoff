import random
from tqdm import tqdm
import torch
import os
import math
from repe.rep_control_reading_vec import WrappedReadingVecModel
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
from datasets import load_dataset, load_from_disk
import argparse
from typing import List, Dict, Any



class ReadingVectors_Harmfulness:
    
    def __init__(self, args):
        self.args = args
        self.reading_vec_dataset = "justinphan3110/harmful_harmless_instructions"
        
        
    def load_reading_vec_dataset(self):
        train_data, train_labels, test_data = self._reading_vec_dataset_chat_model()
        data = {"train_data": train_data, "train_labels": train_labels, "test_data": test_data}
        return train_data, train_labels, test_data       
        
    def _reading_vec_dataset_chat_model(self):
        
        dataset = load_dataset(self.reading_vec_dataset, cache_dir=None)

        train_dataset, test_dataset = dataset['train'], dataset['test'] if 'test' in dataset else dataset['train']

        train_data, train_labels = train_dataset['sentence'], train_dataset['label']
        test_data = test_dataset['sentence']

        train_data = np.concatenate(train_data).tolist()
        test_data = np.concatenate(test_data).tolist()
        
        template = "Request: Can you {instruction}? Answer:"
        train_data_Q_and_A = [template.format(instruction=s) for s in train_data]
        # train_data_Q_and_A = [self.args.template_system_and_user.format(system_prompt=f"Request: {s}", user_message="Answer: ") for s in train_data]
        test_data_by_template = [template.format(instruction=s) for s in test_data]
        
        return train_data_Q_and_A, train_labels, test_data_by_template


class Synthetic_ReadingVectors_Harmfulness:
    
    def __init__(self, args, reading_vec_dataset_save_path=None, model_name_or_path_for_generation="meta-llama/Llama-2-13b-chat-hf"):
        self.args = args
        self.model_name_or_path_for_generation = model_name_or_path_for_generation # the model used to synthetically generate the reading vectors pairs
        self.reading_vec_dataset = "justinphan3110/harmful_harmless_instructions"
        self.reading_vec_dataset_save_path = reading_vec_dataset_save_path if reading_vec_dataset_save_path is not None else f'./data/reading_vec_datasets_new/reading_vec_dataset_{args.model_name.replace("/","_")}.json'
        os.makedirs(os.path.dirname(self.reading_vec_dataset_save_path), exist_ok=True)

    def load_reading_vec_dataset(self):
        if not os.path.exists(self.reading_vec_dataset_save_path):
            os.makedirs(os.path.dirname(self.reading_vec_dataset_save_path), exist_ok=True)
            train_data, train_labels, test_data = self._reading_vec_dataset_raw_model()
            data = {"train_data": train_data, "train_labels": train_labels, "test_data": test_data}
            with open(self.reading_vec_dataset_save_path, 'w') as file:
                json.dump(data, file)
        with open(self.reading_vec_dataset_save_path, 'r') as file:
            data = json.load(file)
        train_data, train_labels, test_data = data["train_data"], data["train_labels"], data["test_data"]
        return train_data, train_labels, test_data   

    def _reading_vec_dataset_raw_model(self):
        template_raw = "{instruction}"
        template = "QUESTION: {instruction}\n\nANSWER: "
        dataset = load_dataset("justinphan3110/harmful_harmless_instructions")

        train_dataset, test_dataset = dataset['train'], dataset['test'] if 'test' in dataset else dataset['train']

        train_data, train_labels = train_dataset['sentence'], train_dataset['label']
        test_data = test_dataset['sentence']

        train_data = np.concatenate(train_data).tolist()
        train_labels_flat = np.concatenate(train_labels).tolist()
        train_data_harmful = np.array([[train_data[j], train_data[j]] for j in range(len(train_data)) if train_labels_flat[j] == False])
        train_data_harmful = np.concatenate(train_data_harmful).tolist()
        test_data = np.concatenate(test_data).tolist()

        train_data_Q_and_A = [template.format(instruction=s) for s in train_data_harmful]
        test_data_by_template = [template.format(instruction=s) for s in test_data]

        model_chat = AutoModelForCausalLM.from_pretrained(self.model_name_or_path_for_generation, torch_dtype=torch.float16, device_map="auto", token=True).eval()
        use_fast_tokenizer = "LlamaForCausalLM" not in model_chat.config.architectures
        tokenizer_chat = AutoTokenizer.from_pretrained(self.model_name_or_path_for_generation, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
        tokenizer_chat.pad_token_id = 0 if tokenizer_chat.pad_token_id is None else tokenizer_chat.pad_token_id
        tokenizer_chat.bos_token_id = 1
        print("load model finished!")

        reading_vec_chat_model = ReadingVectors_Harmfulness(self.args)
        train_data, train_labels, test_data = reading_vec_chat_model.load_reading_vec_dataset()
        rep_token = -1
        hidden_layers = list(range(-1, -model_chat.config.num_hidden_layers, -1))
        n_difference = 1
        direction_method = 'pca'
        rep_reading_pipeline = pipeline("rep-reading", model=model_chat, tokenizer=tokenizer_chat)
        direction_finder_kwargs={"n_components": 1}

        rep_reader = rep_reading_pipeline.get_directions(train_data, rep_token=rep_token, hidden_layers=hidden_layers, n_difference=n_difference, train_labels=train_labels, direction_method=direction_method, direction_finder_kwargs=direction_finder_kwargs)

        pca_vectors = rep_reader.directions #to get vector of specific layer[layer][0]
        pca_signs = rep_reader.direction_signs #to get sign of specific layer[layer]

        #prepare RepE model
        block_name = "decoder_block"

        layer_id = list(range(-25, -33, -1)) # 13B
        if "7b" in self.model_name_or_path_for_generation or "8B" in self.model_name_or_path_for_generation:
            layer_id = list(range(-18, -23, -1))

        wrapped_model = WrappedReadingVecModel(model_chat, tokenizer_chat)
        wrapped_model.unwrap()
        wrapped_model.wrap_block(layer_id, block_name=block_name)

        for j, instruction in enumerate(train_data_harmful):
            coeff = -1.0 if train_labels_flat[j] == True else 2.0
            if 'Llama-3' in self.model_name_or_path_for_generation:
                coeff = -0.5 if train_labels_flat[j] == True else 0.9
            activations = {}
            prompt = torch.unsqueeze(torch.tensor(tokenizer_chat.encode(instruction)),dim=0)
            for layer in layer_id:
                v = torch.tensor(pca_vectors[layer]*pca_signs[layer][0])
                v = (v / torch.norm(v)).cpu()
                activations[layer] = torch.tensor(coeff * v).to(model_chat.device).half()
            wrapped_model.reset()
            wrapped_model.set_controller(layer_id, activations, 'decoder_block')
            with torch.no_grad():
                outputs = model_chat.generate(prompt.cuda(), max_new_tokens=32, do_sample=False).detach().cpu()
                generation = tokenizer_chat.decode(outputs[0], skip_special_tokens=True).replace(instruction, "")
                train_data_Q_and_A[j] = f'{train_data_Q_and_A[j]}{generation}'

        return train_data_Q_and_A, train_labels, test_data_by_template
    
    
class ReadingVectors_Fairness:
    
    def __init__(self, args):
        self.args = args
        self.reading_vec_dataset_path = 'stereoset'
        self.reading_vec_dataset_name = 'intrasentence'
        
        
    def load_reading_vec_dataset(self):
        train_data, train_labels, test_data = self._reading_vec_dataset_chat_model()
        data = {"train_data": train_data, "train_labels": train_labels, "test_data": test_data}
        return train_data, train_labels, test_data

    def _reading_vec_dataset_chat_model(self, ntrain=150, bias='race'):
        def get_scenarios(example):
            data = example['sentences']
            # gold_label is an array with 0,1,2 in which 0 correspond to anti_stereotype and 1 correspond to stereotype
            anti_stereotype_idx = data['gold_label'].index(0)
            stereotype_idx = data['gold_label'].index(1)
            scenario = ""
            if 'BLANK' not in example['context']:
                scenario += example['context'] + " "
            return (
                scenario + data['sentence'][anti_stereotype_idx],
                scenario + data['sentence'][stereotype_idx],
            )
        dataset = load_from_disk(self.reading_vec_dataset_path).shuffle(seed=1)
        dataset = dataset.filter(lambda x: x['bias_type'] == bias)
        m = min([ntrain, len(dataset)])
        new_train_labels = []
        new_train_statements = []
        alignment = False if ("chat" in self.args.model_name or "instruct" in self.args.model_name) else True
        order = [False, True] if alignment else [True, False]
        for d in dataset.select(range(m)):
            anti_steotype, stereotype = get_scenarios(d)
            non_biased_statement = f"Question: Consider the statement: {stereotype} Answer: The statement is false and racist."
            biased_statement = f"Question: Consider the statement: {stereotype} Answer: 100%"
            new_train_statements.append(biased_statement)
            new_train_statements.append(non_biased_statement)
            new_train_labels.append(order.copy())

        dataset = load_from_disk(self.reading_vec_dataset_path).shuffle(seed=1)
        dataset = dataset.filter(lambda x: x['bias_type'] == "race")
        # template_test = "Yes or No: Do you agree with the following statement?\nStatement: {statement}\nANSWER: "
        template_test = "QUESTION: Do you agree with the following statement: {statement} ANSWER: "
        test_dataset = dataset.select(range(ntrain,len(dataset)))
        test_data = []
        for d in test_dataset:
            anti_steotype, stereotype = get_scenarios(d)
            current_group = [anti_steotype, stereotype]
            test_data.extend(current_group)

        test_data = [template_test.format(statement=s) for s in test_data]
        test_labels = [1, 0] * (len(test_data) // 2)

        return new_train_statements, new_train_labels, test_data
    
class Synthetic_ReadingVectors_Fairness:
    
    def __init__(self, args, reading_vec_dataset_save_path=None, model_name_or_path_for_generation="meta-llama/Llama-2-13b-chat-hf"):
        self.args = args
        self.reading_vec_dataset_path = 'stereoset'
        self.reading_vec_dataset_name = 'intrasentence'
        self.model_name_or_path_for_generation = model_name_or_path_for_generation # the model used to synthetically generate the reading vectors pairs
        self.reading_vec_dataset_save_path = reading_vec_dataset_save_path
        os.makedirs(os.path.dirname(self.reading_vec_dataset_save_path), exist_ok=True)

    def load_reading_vec_dataset(self):
        if not os.path.exists(self.reading_vec_dataset_save_path):
            os.makedirs(os.path.dirname(self.reading_vec_dataset_save_path), exist_ok=True)
            train_data, train_labels, test_data = self._reading_vec_dataset_raw_model()
            data = {"train_data": train_data, "train_labels": train_labels, "test_data": test_data}
            with open(self.reading_vec_dataset_save_path, 'w') as file:
                json.dump(data, file)
        with open(self.reading_vec_dataset_save_path, 'r') as file:
            data = json.load(file)
            train_data, train_labels, test_data = data["train_data"], data["train_labels"], data["test_data"]
        return train_data, train_labels, test_data   
    

    def _reading_vec_dataset_raw_model(self):

        def get_scenarios(example):
            data = example['sentences']
            # gold_label is an array with 0,1,2 in which 0 correspond to anti_stereotype and 1 correspond to stereotype
            anti_stereotype_idx = data['gold_label'].index(0)
            stereotype_idx = data['gold_label'].index(1)
            scenario = ""
            if 'BLANK' not in example['context']:
                scenario += example['context'] + " "
            return (
                scenario + data['sentence'][anti_stereotype_idx],
                scenario + data['sentence'][stereotype_idx],
            )
            
        ntrain=150
        dataset = load_from_disk(self.reading_vec_dataset_path).shuffle(seed=1)
        dataset = dataset.filter(lambda x: x['bias_type'] == 'race')
        m = min([ntrain, len(dataset)])
        train_labels_flat = []
        train_data_Q_and_A = []
        alignment = False if ("chat" in self.args.model_name or "instruct" in self.args.model_name) else True
        order = [False, True] if alignment else [True, False]
        for d in dataset.select(range(m)):
            anti_steotype, stereotype = get_scenarios(d)
            non_biased_statement = f"QUESTION: Consider the statement: {stereotype} ANSWER: "
            biased_statement = f"QUESTION: Consider the statement: {stereotype} ANSWER: "
            train_data_Q_and_A.append(biased_statement)
            train_data_Q_and_A.append(non_biased_statement)
            train_labels_flat.append(order.copy())

        dataset = load_from_disk(self.reading_vec_dataset_path).shuffle(seed=1)
        dataset = dataset.filter(lambda x: x['bias_type'] == "fairness")

        template_test = "QUESTION: Do you agree with the following statement: {statement} ANSWER: "
        test_dataset = dataset.select(range(ntrain, len(dataset)))
        test_data = []
        for d in test_dataset:
            anti_steotype, stereotype = get_scenarios(d)
            current_group = [anti_steotype, stereotype]
            test_data.extend(current_group)

        test_data_by_template = [template_test.format(statement=s) for s in test_data]

        model_chat = AutoModelForCausalLM.from_pretrained(self.model_name_or_path_for_generation, torch_dtype=torch.float16, device_map="auto", token=True).eval()
        use_fast_tokenizer = "LlamaForCausalLM" not in model_chat.config.architectures
        tokenizer_chat = AutoTokenizer.from_pretrained(self.model_name_or_path_for_generation, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
        tokenizer_chat.pad_token_id = 0 if tokenizer_chat.pad_token_id is None else tokenizer_chat.pad_token_id
        tokenizer_chat.bos_token_id = 1
        print("load model finished!")

        reading_vec_chat_model = ReadingVectors_Fairness(self.args)
        train_data, train_labels, _ = reading_vec_chat_model.load_reading_vec_dataset()
        rep_token = -1
        hidden_layers = list(range(-1, -model_chat.config.num_hidden_layers, -1))
        n_difference = 1
        direction_method = 'pca'
        rep_reading_pipeline = pipeline("rep-reading", model=model_chat, tokenizer=tokenizer_chat)
        direction_finder_kwargs={"n_components": 1}

        rep_reader = rep_reading_pipeline.get_directions(train_data, rep_token=rep_token, hidden_layers=hidden_layers, n_difference=n_difference, train_labels=train_labels, direction_method=direction_method, direction_finder_kwargs=direction_finder_kwargs)

        pca_vectors = rep_reader.directions #to get vector of specific layer[layer][0]
        pca_signs = rep_reader.direction_signs #to get sign of specific layer[layer]

        #prepare RepE model
        block_name = "decoder_block"

        layer_id = list(range(-25, -33, -1)) # 13B
        if "7b" in self.model_name_or_path_for_generation or "8B" in self.model_name_or_path_for_generation:
            layer_id = list(range(-18, -23, -1))

        wrapped_model = WrappedReadingVecModel(model_chat, tokenizer_chat)
        wrapped_model.unwrap()
        wrapped_model.wrap_block(layer_id, block_name=block_name)

        for j, instruction in enumerate(train_data_Q_and_A):
            coeff = -1.0 if train_labels_flat[j] == True else 2.5
            if 'Llama-3' in self.model_name_or_path_for_generation:
                coeff = -0.8 if train_labels_flat[j] == True else 1.2
            activations = {}
            prompt = torch.unsqueeze(torch.tensor(tokenizer_chat.encode(instruction)),dim=0)
            for layer in layer_id:
                v = torch.tensor(pca_vectors[layer]*pca_signs[layer][0])
                v = (v / torch.norm(v)).cpu()
                activations[layer] = torch.tensor(coeff * v).to(model_chat.device).half()
            wrapped_model.reset()
            wrapped_model.set_controller(layer_id, activations, 'decoder_block')
            with torch.no_grad():
                outputs = model_chat.generate(prompt.cuda(), max_new_tokens=32, do_sample=False).detach().cpu()
                generation = tokenizer_chat.decode(outputs[0], skip_special_tokens=True).replace(instruction, "")
                train_data_Q_and_A[j] = f'{train_data_Q_and_A[j]}{generation}'

        return train_data_Q_and_A, train_labels, test_data_by_template
    