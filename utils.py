import json
import os
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import gc
import numpy as np
import random
from peft import LoraConfig, get_peft_model, PeftModel

question_template = \
"""[INST] <<SYS>>
You are a helpful, respectful and honest assistant. You will be given a python function to complete.
Complete the function correctly. Separate the code of the function from the rest of your message.
Avoid unnecessary indentation in your answer. Only give one answer. 
<</SYS>>
{user_prompt} [/INST]"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_json_if_exists(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Error: {file_path} is not a valid JSON file.")
            return dict()
    else:
        print(f"Error: {file_path} does not exist.")
        return dict()


def sample_model(model, tokenizer, question, num_samples=32, batch_size=2, question_template_for_sample=None,
                 device=device):
    if question_template_for_sample is None:
        question_template_for_sample = question_template
    prompt = question_template_for_sample.format(user_prompt=question['prompt'])
    q_encoding = tokenizer.encode_plus(prompt, return_tensors="pt", padding=True)
    input_ids = q_encoding['input_ids'].to(device)
    attn_mask = q_encoding['attention_mask'].to(device)
    num_batches = num_samples // batch_size

    all_answers = []
    for j in range(num_batches):
        with torch.inference_mode():
            outputs = model.generate(input_ids, max_new_tokens=1800, temperature=1.0, do_sample=True,
                                     top_p=0.95, attention_mask=attn_mask,
                                     return_dict_in_generate=True,
                                     pad_token_id=tokenizer.pad_token_id, num_return_sequences=batch_size)
        # decode the input only
        partial_given_answers = [tokenizer.decode(output_sequence[:input_ids.shape[1]], skip_special_tokens=True) for
                                 output_sequence in outputs.sequences]
        # decode the entire output, and remove the input from it
        curr_answers = [
            tokenizer.decode(outputs.sequences[i], skip_special_tokens=True).replace(partial_given_answers[i],
                                                                                     "").replace(
                '<s>', "").replace('</s>', "") for i in range(len(partial_given_answers))]
        all_answers += curr_answers

    return all_answers

def load_model(model_path):
    base_model = os.path.join('/cs/labs/shashua/binyamin/models/', "Meta-Llama-3.1-8B")
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, token=True,
                                                 local_files_only=True, cache_dir=None, use_cache=False).eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", legacy=False, token=True,
                                              local_files_only=True, cache_dir=None, use_cache=False)

    # Load the fine-tuned LoRA weights
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()
    model.to(device)
    return model, tokenizer

def clear_memory(*objects):
    # Delete any objects passed as arguments
    for obj in objects:
        del obj

    # Collect garbage
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_checkpoint_models(model_dir, base_model = None):
    if base_model is None:
        base_model = os.path.join('/cs/labs/shashua/binyamin/models/', "Meta-Llama-3.1-8B")

    model_subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    model_subdirs = sorted(model_subdirs, key=lambda x: int(x.split('-')[-1]))
    model_subdirs.insert(0, base_model)
    return model_subdirs


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_color(text, color):
    print(color + text + bcolors.ENDC)
