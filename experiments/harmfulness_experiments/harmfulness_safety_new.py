import csv
import tqdm
import torch
import json
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from tqdm import tqdm
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from repe.rep_control_reading_vec import WrappedReadingVecModel
from experiments.GenArgs import GenerationArgsSafety
from experiments.WrapModel import WrapModel
from experiments.generate_reading_vectors import ReadingVectorsChatModel_Harmfulness, ReadingVectorsRawModel_Harmfulness
# from harmfulness_experiments.harmfulness_utils import reading_vec_dataset_raw_model
from repe import repe_pipeline_registry
repe_pipeline_registry()
from experiments.utils import generate_responses, feed_forward_responses, feed_dialog_behavior
from experiments.utils import load_test_dataset
from datasets import Dataset

args = GenerationArgsSafety()
print(args)

################################# load model
model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                            torch_dtype=torch.float16,
                                            device_map="auto",
                                            token='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',
                                            use_cache=True
                                        ).eval()
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(
                                        args.model_name,
                                        use_fast=use_fast_tokenizer,
                                        padding_side="left",
                                        legacy=False,
                                        token='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',
                                    )
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1
print("load model finished!")

################################# load the llama2 model vocabulary
vocabulary = tokenizer.get_vocab()
os.environ['HF_HOME'] = '/home/dshteyma/.cache/huggingface'
################################# load the harmful dataset behavior
if "chat" in args.model_name or "Llama-3" in args.model_name:
    # RLHF aligned model
    reading_vecs = ReadingVectorsChatModel_Harmfulness(args)    
else:
    # synthetic reading vectors for helpfulness experiments
    model_name_or_path_for_generation = 'meta-llama/Meta-Llama-3.1-8B-Instruct' if "Llama-3" in args.model_name else 'meta-llama/Llama-2-13b-hf'
    reading_vec_dataset_save_path = f'./data/reading_vec_datasets/reading_vec_dataset_{args.model_name.replace("/","_")}.json'
    reading_vecs = ReadingVectorsRawModel_Harmfulness(args, reading_vec_dataset_save_path, model_name_or_path_for_generation)
    
train_data, train_labels, test_data = reading_vecs.load_reading_vec_dataset()
if "chat" in args.model_name or "Llama-3" in args.model_name:
    for i, elem in enumerate(train_data):
        if train_labels[i//2][i%2]==False:
            train_data[i] = elem.replace("\n\nAnswer: ", "\n\nAnswer: Don't do it, it's illegal and harmful")
        else:
            train_data[i] = elem.replace("\n\nAnswer: ", "\n\nAnswer: Sure! Here's a solution")

harmful_test_data = test_data[1::2][:100] # use only the first 100 harmful examples
# harmful_test_data = [elem.replace("\n\nAnswer: ", "\n\nCan you assist with the request? Answer Yes or No: ") for elem in harmful_test_data]
dataset_names = args.dataset_names.split(',') if args.dataset_names != None else ['harmful_test_data']

wrap_model = WrapModel(model, tokenizer, train_data, train_labels)
pca_vectors, pca_signs, layer_ids_injections = wrap_model.prepare_wrapped_model()

for dataset_name in dataset_names: 
    if dataset_name == 'harmful_test_data':
        dataset = Dataset.from_dict({'input': harmful_test_data})
    else:
        dataset = load_test_dataset(dataset_path=args.dataset_path, dataset_name=dataset_name)
    #test model on dataset for various norms of injected vectors
    x = list(np.round(np.arange(args.start_coeff, args.end_coeff, args.coeff_step), 1))
    behavior_harmful_mean = {key: 0 for key in x}
    behavior_harmful_std = {key: 0 for key in x}
    all_answers_dict = {key: {} for key in x}

    for coeff in x:
        wrapped_model = wrap_model.wrap_model(coeff)
        args.num_instructions = min(args.num_instructions, len(dataset))
        # full auto-regreesive response generation
        all_answers, all_logits = generate_responses(
                                                    model, 
                                                    tokenizer, 
                                                    dataset, 
                                                    args, 
                                                    template_format='default',
                                                    do_sample=False
                                                ) 
        # Only one forward pass to get the first logits
        all_logits_forward_pass = feed_forward_responses(model, tokenizer, dataset, args, template_format='default')
        behavior_all_samples = feed_dialog_behavior(tokenizer, dataset, args, all_answers)
        
        behavior_harmful_mean[coeff] = np.nanmean(np.nanmean(behavior_all_samples, axis=0))        
        behavior_harmful_std[coeff] = np.nanmean(np.nanstd(behavior_all_samples, axis=0))
        print(f'\np_mean for coeff = {coeff}: {behavior_harmful_mean[coeff]}')
        print(f'\nacc_std for coeff {coeff}: {behavior_harmful_std[coeff]}')

        os.makedirs(args.output_dir, exist_ok=True)
        with open(f'{args.output_dir}/safety_harmfulness_{args.model_name.replace("/","_")}_stats_sample.json', 'w') as file:
            results = {'behavior_harmful_mean': behavior_harmful_mean, 'behavior_harmful_std': behavior_harmful_std}
            json.dump(results, file)
            
        all_answers_dict[coeff] = all_answers
        with open(f'{args.output_dir}/safety_harmfulness_{args.model_name.replace("/","_")}_answers_sample.json', 'w') as file:
            json.dump(all_answers_dict, file)
            
    
    