import torch
import numpy as np
from transformers import (DataCollatorWithPadding, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer)
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, LoftQConfig
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import pdb
import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dir = '/cs/labs/shashua/binyamin/models/'
model_path = os.path.join(models_dir, "Meta-Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, token=True).eval()
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast_tokenizer,
                                          padding_side="left", legacy=False, token=True)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1
print("load model finished!")


def filter_function(example):
    return example['is_response_0_safe'] != example['is_response_1_safe']


def preprocess_function(example):
    output_dict = {}
    if example['is_response_0_safe']:
        output_dict['chosen'] = example['response_0']
        output_dict['rejected'] = example['response_1']
    else:
        output_dict['chosen'] = example['response_1']
        output_dict['rejected'] = example['response_0']
    return output_dict


dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
train_dataset, test_dataset = dataset['train'], dataset['test']
train_dataset = train_dataset.filter(filter_function)
test_dataset = test_dataset.filter(filter_function)

train_dataset = train_dataset.map(preprocess_function)
test_dataset = test_dataset.map(preprocess_function)

loftq_config = LoftQConfig(loftq_bits=4)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=2,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj"],
    loftq_config=loftq_config,
)

model = get_peft_model(model, peft_config)
model.to(device)

batch_size = 1
per_device_batch_size = batch_size // 2
num_epochs = 1

total_steps = math.ceil((len(list(train_dataset)) * num_epochs) / batch_size)
save_steps = math.ceil(total_steps / 3)

training_args = DPOConfig(
    output_dir="./lora_finetuned_model",
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    logging_dir="./logs",
    save_steps=save_steps,  # Save the model every 100 steps
    gradient_accumulation_steps=8,
    fp16=True,  # Enable mixed precision training
)

dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    max_prompt_length=128,
    max_length=32,
)

dpo_trainer.train()
