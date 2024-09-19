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
from datasets import load_dataset
import argparse
from typing import List, Dict, Any


class ReadingVectorsChatModel:
    
    def __init__(self, args):
        self.args = args
        self.reading_vec_dataset = "justinphan3110/harmful_harmless_instructions"
        self.reading_vec_dataset_save_path = f'./data/reading_vec_datasets/reading_vec_dataset_{args.model_name.replace("/","_")}.json'
        os.makedirs(os.path.dirname(self.reading_vec_dataset_save_path), exist_ok=True)
        
    def load_reading_vec_dataset(self):
        if not os.path.exists(self.reading_vec_dataset_save_path):
            train_data, train_labels, test_data_by_template = self._reading_vec_dataset_chat_model()
            data = {"train_data": train_data, "train_labels": train_labels, "test_data_by_template": test_data_by_template}
            with open(self.reading_vec_dataset_save_path, 'w') as file:
                json.dump(data, file)
        with open(self.reading_vec_dataset_save_path, 'r') as file:
            data = json.load(file)
            train_data, train_labels, test_data_by_template = data["train_data"], data["train_labels"], data["test_data_by_template"]
        return train_data, train_labels, test_data_by_template       
        
    def _reading_vec_dataset_chat_model(self):
        
        template = self.args.template_user
        dataset = load_dataset(self.reading_vec_dataset, cache_dir=None)

        train_dataset, test_dataset = dataset['train'], dataset['test'] if 'test' in dataset else dataset['train']

        train_data, train_labels = train_dataset['sentence'], train_dataset['label']
        test_data = test_dataset['sentence']

        train_data = np.concatenate(train_data).tolist()
        test_data = np.concatenate(test_data).tolist()

        train_data = [template.format(user_message=s) for s in train_data]
        test_data_by_template = [template.format(user_message=s) for s in test_data]
        
        return train_data, train_labels, test_data_by_template
    
class ReadingVectorsRawModel:
    
    def __init__(self, args, model_name_or_path_for_generation="meta-llama/Llama-2-13b-chat-hf"):
        self.args = args
        self.model_name_or_path_for_generation = model_name_or_path_for_generation # the model used to synthetically generate the reading vectors pairs
        self.reading_vec_dataset = "justinphan3110/harmful_harmless_instructions"
        self.reading_vec_dataset_save_path = f'./data/reading_vec_datasets/reading_vec_dataset_{args.model_name.replace("/","_")}.json'
        os.makedirs(os.path.dirname(self.reading_vec_dataset_save_path), exist_ok=True)

    def load_reading_vec_dataset(self):
        if not os.path.exists(self.reading_vec_dataset_save_path):
            train_data, train_labels, test_data_by_template = self._reading_vec_dataset_raw_model()
            data = {"train_data": train_data, "train_labels": train_labels, "test_data_by_template": test_data_by_template}
            with open(self.reading_vec_dataset_save_path, 'w') as file:
                json.dump(data, file)
        with open(self.reading_vec_dataset_save_path, 'r') as file:
            data = json.load(file)
            train_data, train_labels, test_data_by_template = data["train_data"], data["train_labels"], data["test_data_by_template"]
        return train_data, train_labels, test_data_by_template   

    def _reading_vec_dataset_raw_model(self):
        dataset = load_dataset(self.reading_vec_dataset, cache_dir=None)

        train_dataset, test_dataset = dataset['train'], dataset['test'] if 'test' in dataset else dataset['train']

        train_data, train_labels = train_dataset['sentence'], train_dataset['label']
        test_data = test_dataset['sentence']

        train_data_flat = np.concatenate(train_data).tolist()
        train_labels_flat = np.concatenate(train_labels).tolist()
        train_data_harmful = np.array([[train_data_flat[j], train_data_flat[j]] for j in range(len(train_data_flat)) if train_labels_flat[j] == False])
        train_data_harmful = np.concatenate(train_data_harmful).tolist()
        test_data = np.concatenate(test_data).tolist()

        train_data_Q_and_A = [self.args.template_system_and_user.format(system_prompt=f"Request: {s}", user_message="Answer:") for s in train_data_harmful]
        test_data_by_template = [self.args.template_system_and_user.format(system_prompt=f"Request: {s}", user_message="Answer:") for s in test_data]

        if not os.path.exists(self.reading_vec_dataset_save_path):
            model_chat = AutoModelForCausalLM.from_pretrained(self.model_name_or_path_for_generation, torch_dtype=torch.float16, device_map="auto", token=True).eval()
            use_fast_tokenizer = "LlamaForCausalLM" not in model_chat.config.architectures
            tokenizer_chat = AutoTokenizer.from_pretrained(self.model_name_or_path_for_generation, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
            tokenizer_chat.pad_token_id = 0 if tokenizer_chat.pad_token_id is None else tokenizer_chat.pad_token_id
            tokenizer_chat.bos_token_id = 1
            print("load model finished!")

            reading_vec_chat_model = ReadingVectorsChatModel(self.args)
            train_data, _, test_data_by_template = reading_vec_chat_model.load_reading_vec_dataset()
            rep_token = -1
            hidden_layers = list(range(-1, -model_chat.config.num_hidden_layers, -1))
            n_difference = 1
            direction_method = 'pca'
            rep_reading_pipeline = pipeline("rep-reading", model=model_chat, tokenizer=tokenizer_chat)
            direction_finder_kwargs={"n_components": 1}

            rep_reader = rep_reading_pipeline.get_directions(train_data, rep_token=rep_token,hidden_layers=hidden_layers,n_difference=n_difference, train_labels=train_labels, direction_method=direction_method, direction_finder_kwargs=direction_finder_kwargs)

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
                coeff = -1.0 if train_labels_flat[j] == True else 2.5
                if 'Llama-3' in self.model_name_or_path_for_generation:
                    coeff = -0.9 if train_labels_flat[j] == True else 0.8
                activations = {}
                prompt = torch.unsqueeze(torch.tensor(tokenizer_chat.encode(instruction)),dim=0)
                for layer in layer_id:
                    v = torch.tensor(pca_vectors[layer]*pca_signs[layer][0])
                    v = (v / torch.norm(v)).cpu()
                    activations[layer] = torch.tensor(coeff * v).to(model_chat.device).half()
                wrapped_model.reset()
                wrapped_model.set_controller(layer_id, activations, 'decoder_block')
                with torch.no_grad():
                    outputs = model_chat.generate(prompt.cuda(), max_new_tokens=64, do_sample=False).detach().cpu()
                    generation = tokenizer_chat.decode(outputs[0], skip_special_tokens=True).replace(instruction, "")
                    train_data_Q_and_A[j] = f'{train_data_Q_and_A[j]}{generation}'

        return train_data_Q_and_A, train_labels, test_data_by_template