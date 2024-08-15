import torch
import numpy as np
from transformers import (DataCollatorWithPadding, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer)
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer
import pdb
import math
import os


models_dir = '/cs/labs/shashua/binyamin/models/'
model_path = os.path.join(models_dir, "Meta-Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                             device_map="auto", token=True).eval()
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast_tokenizer,
                                          padding_side="left", legacy=False, token=True)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1
print("load model finished!")


def filter_function(example):
    return example['is_response_0_safe'] != example['is_response_1_safe']


dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
train_dataset, test_dataset = dataset['train'], dataset['test']
train_dataset = train_dataset.filter(filter_function)
test_dataset = test_dataset.filter(filter_function)

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    # target_modules=["q_proj", "k_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)
device = torch.device("cuda")
model.to(device)

batch_size = 8
per_device_batch_size = batch_size // 2
num_epochs = 5

# need this to fix padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
eval_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)

total_steps = math.ceil((len(list(train_dataset)) * num_epochs) / batch_size)
save_steps = math.ceil(total_steps / 12)

training_args = TrainingArguments(
    output_dir="./lora_finetuned_model",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=per_device_batch_size,
    weight_decay=0.01,
    logging_dir="./logs",
    save_steps=save_steps,  # Save the model every 100 steps
    gradient_accumulation_steps=4,
)

dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_prompt_length=512,
    max_length=2048,
)

dpo_trainer.train()
