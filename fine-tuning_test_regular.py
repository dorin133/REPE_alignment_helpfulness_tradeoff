import torch
import numpy as np
from transformers import (DataCollatorWithPadding, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, LoftQConfig
from datasets import load_dataset
import pdb
import math
import os
import wandb

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
    prompt = f"Question: {example['prompt']}\n Answer: "
    if example['is_response_0_safe']:
        prompt += example['response_0']
    else:
        prompt += example['response_1']

    output = tokenizer(prompt, truncation=True, max_length=512)

    return output

NUM_EXAMPLES = 500
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
train_dataset, test_dataset = dataset['train'], dataset['test']
train_dataset = train_dataset.filter(filter_function).select(range(NUM_EXAMPLES))
test_dataset = test_dataset.filter(filter_function).select(range(NUM_EXAMPLES // 2))

train_dataset = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(preprocess_function, remove_columns=test_dataset.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

loftq_config = LoftQConfig(loftq_bits=4)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=16,
    lora_dropout=0.01,
    # target_modules=["q_proj", "k_proj", "v_proj"],
    # loftq_config=loftq_config,
)

model = get_peft_model(model, peft_config)
model.to(device)

training_args = TrainingArguments(
    output_dir="./lora_finetuned_model_15_09",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=0.1,
    gradient_accumulation_steps=16,
    fp16=True,
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=0.01,
    remove_unused_columns=False,
    learning_rate=1e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
