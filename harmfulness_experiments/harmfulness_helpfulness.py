"""
Harmfulness vs Helpfulness Evaluation Script

This script evaluates the trade-off between harmfulness and helpfulness in language models 
using Representation Engineering (REPE). It measures model performance on MMLU datasets 
with various coefficient interventions to analyze how safety interventions affect helpfulness.
"""

import csv
import torch
import json
import os
import pandas as pd
import numpy as np
import sys
import pickle
import math
import argparse
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from datasets import load_dataset
import torch.nn.functional as F

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from repe.rep_control_reading_vec import WrappedReadingVecModel
from utils.GenArgs import GenerationArgsHelpfulness
from utils.WrapModel import WrapModel
from utils.generate_reading_vectors import Synthetic_ReadingVectors_Harmfulness, ReadingVectors_Harmfulness
from utils.utils import generate_responses, feed_mmlu_helpfulness, feed_forward_responses
from utils.utils import load_test_dataset, get_logits_dict_and_probs, get_norms_and_projections
from repe import repe_pipeline_registry

# Register REPE pipeline components
repe_pipeline_registry()


def setup_model_and_tokenizer(model_name: str) -> Tuple[Any, Any]:
    """
    Load and configure the language model and tokenizer.
    
    Args:
        model_name (str): Name or path of the model to load
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Load model with appropriate configuration for inference
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True
    ).eval()
    
    # Configure tokenizer based on model architecture
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast_tokenizer,
        padding_side="left",
        legacy=False,
    )
    
    # Set up tokenizer special tokens
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    
    print("Model loading completed!")
    return model, tokenizer


def setup_reading_vectors(args: GenerationArgsHelpfulness) -> Tuple[List, List, Any]:
    """
    Set up reading vectors for representation engineering based on configuration.
    
    Args:
        args: Configuration arguments containing experiment parameters
        
    Returns:
        Tuple of (train_data, train_labels, test_data)
    """
    if args.is_synth_reading_vectors:
        # Use synthetic reading vectors for helpfulness experiments
        model_name_for_generation = (
            'meta-llama/Meta-Llama-3.1-8B-Instruct' if "Llama-3" in args.model_name 
            else 'meta-llama/Llama-2-13b-chat-hf'
        )
        
        reading_vec_dataset_save_path = (
            f'/path/to/data/reading_vec_datasets/reading_vec_dataset_'
            f'{args.model_name.replace("/","_")}.json'
        )
        
        reading_vecs = Synthetic_ReadingVectors_Harmfulness(
            args, 
            reading_vec_dataset_save_path, 
            model_name_for_generation
        )
    else:
        # Use RLHF aligned model reading vectors
        reading_vecs = ReadingVectors_Harmfulness(args)
    
    return reading_vecs.load_reading_vec_dataset()


def initialize_result_containers(coefficient_range: List[float]) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict]:
    """
    Initialize dictionaries to store experimental results.
    
    Args:
        coefficient_range: List of coefficient values to test
        
    Returns:
        Tuple of result dictionaries for various metrics
    """
    acc_mean = {key: 0 for key in coefficient_range}
    acc_std = {key: 0 for key in coefficient_range}
    p_mean = {key: 0 for key in coefficient_range}
    p_mean_relative = {key: 0 for key in coefficient_range}
    p_std = {key: 0 for key in coefficient_range}
    p_std_relative = {key: 0 for key in coefficient_range}
    all_answers_dict = {coeff: {} for coeff in coefficient_range}
    vector_norms = {coeff: {} for coeff in coefficient_range}
    projections = {coeff: {} for coeff in coefficient_range}
    
    return (acc_mean, acc_std, p_mean, p_mean_relative, p_std, p_std_relative, 
            all_answers_dict, vector_norms, projections)


def compute_baseline_metrics(model: Any, tokenizer: Any, dataset: Any, 
                           args: GenerationArgsHelpfulness) -> Tuple[Dict, Any]:
    """
    Compute baseline metrics without REPE intervention (coefficient = 0).
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        dataset: Test dataset
        args: Configuration arguments
        
    Returns:
        Tuple of (baseline_logits_dict, baseline_best_indices)
    """
    print("Computing baseline metrics (no REPE intervention)...")
    
    # Get logits without any intervention
    baseline_logits = feed_forward_responses(
        model, tokenizer, dataset, args, 
        template_format='mmlu', batch_size=16
    )
    
    # Extract logit dictionary and best answer indices
    baseline_logits_dict, baseline_best_indices = get_logits_dict_and_probs(
        args, dataset, baseline_logits
    )
    
    return baseline_logits_dict, baseline_best_indices


def evaluate_coefficient(model: Any, tokenizer: Any, dataset: Any, 
                        wrap_model: WrapModel, args: GenerationArgsHelpfulness,
                        coeff: float, baseline_logits_dict: Dict, 
                        baseline_best_indices: Any) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    """
    Evaluate model performance at a specific coefficient value.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        dataset: Test dataset
        wrap_model: Model wrapper for REPE interventions
        args: Configuration arguments
        coeff: Coefficient value for REPE intervention
        baseline_logits_dict: Baseline logits without intervention
        baseline_best_indices: Baseline best answer indices
        
    Returns:
        Tuple of evaluation metrics and results
    """
    print(f"Evaluating coefficient: {coeff}")
    
    # Wrap model with current coefficient
    wrapped_model = wrap_model.wrap_model(coeff)
    
    # Ensure we don't exceed dataset size
    args.num_instructions = min(args.num_instructions, len(dataset))
    
    # Generate full auto-regressive responses
    all_answers, all_logits = generate_responses(
        model, 
        tokenizer, 
        dataset, 
        args, 
        template_format='mmlu',
        batch_size=16,
        do_sample=getattr(args, 'do_sample', False)
    )
    
    # Get first-token logits via single forward pass
    all_logits_forward_pass = feed_forward_responses(
        model, tokenizer, dataset, args, 
        template_format='mmlu', batch_size=16
    )
    
    # Compute vector norms and projections
    vector_norms, norms_stds, projections = get_norms_and_projections(
        args, dataset, tokenizer, all_logits_forward_pass, 
        baseline_logits_dict, baseline_best_indices
    )
    
    # Evaluate MMLU helpfulness metrics
    probs_samples, p_relative_samples, acc_samples = feed_mmlu_helpfulness(
        tokenizer, 
        dataset, 
        args, 
        all_answers, 
        all_logits, 
        all_logits_forward_pass,
        batch_size=16
    )
    
    return (probs_samples, p_relative_samples, acc_samples, 
            all_answers, vector_norms, projections)


def compute_statistics(probs_samples: List, p_relative_samples: List, 
                      acc_samples: List) -> Tuple[float, float, float, float, float, float]:
    """
    Compute mean and standard deviation statistics from sample results.
    
    Args:
        probs_samples: Probability samples across runs
        p_relative_samples: Relative probability samples
        acc_samples: Accuracy samples
        
    Returns:
        Tuple of (p_mean, p_std, p_mean_relative, p_std_relative, acc_mean, acc_std)
    """
    # Convert NaN values to 0 for stable computation
    probs_clean = np.nan_to_num(probs_samples, nan=0.0)
    p_relative_clean = np.nan_to_num(p_relative_samples, nan=0.0)
    acc_clean = np.nan_to_num(acc_samples, nan=0.0)
    
    # Compute means
    p_mean = np.mean(np.mean(probs_clean, axis=0))
    p_mean_relative = np.mean(np.mean(p_relative_clean, axis=0))
    acc_mean = np.mean(np.mean(acc_clean, axis=0))
    
    # Compute standard deviations
    p_std = np.std(np.mean(probs_clean, axis=0))
    p_std_relative = np.std(np.mean(p_relative_clean, axis=0))
    acc_std = np.std(np.mean(acc_clean, axis=0))
    
    return p_mean, p_std, p_mean_relative, p_std_relative, acc_mean, acc_std


def save_results(results: Dict, dataset_name: str, model_name: str, 
                output_dir: str, result_type: str) -> None:
    """
    Save experimental results to JSON or pickle files.
    
    Args:
        results: Results dictionary to save
        dataset_name: Name of the dataset
        model_name: Name of the model
        output_dir: Output directory path
        result_type: Type of results ('stats', 'answers', 'proj', 'vec_norms')
    """
    # Create output directory if it doesn't exist
    dataset_output_dir = f'{output_dir}/{dataset_name}/'
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Sanitize model name for filename
    safe_model_name = model_name.replace("/", "_")
    
    # Determine file extension and save method
    if result_type in ['proj', 'vec_norms']:
        filename = f'{dataset_output_dir}/helpfulness_harmfulness_{safe_model_name}_{result_type}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    else:
        filename = f'{dataset_output_dir}/helpfulness_harmfulness_{safe_model_name}_{result_type}_sample.json'
        with open(filename, 'w') as file:
            if result_type == 'stats':
                json.dump(results, file, indent=2)
            else:
                json.dump(results, file, indent=2)
    
    print(f"Saved {result_type} results to: {filename}")


def print_coefficient_results(coeff: float, p_mean: float, p_std: float, 
                            p_mean_relative: float, p_std_relative: float, 
                            acc_mean: float, acc_std: float) -> None:
    """
    Print formatted results for a specific coefficient.
    
    Args:
        coeff: Coefficient value
        p_mean: Mean probability
        p_std: Standard deviation of probability
        p_mean_relative: Mean relative probability
        p_std_relative: Standard deviation of relative probability
        acc_mean: Mean accuracy
        acc_std: Standard deviation of accuracy
    """
    print(f'Results for coefficient {coeff}:')
    print(f'  Probability - Mean: {p_mean:.4f}, Std: {p_std:.4f}')
    print(f'  Relative Prob - Mean: {p_mean_relative:.4f}, Std: {p_std_relative:.4f}')
    print(f'  Accuracy - Mean: {acc_mean:.4f}, Std: {acc_std:.4f}')
    print('-' * 50)


def main():
    """
    Main execution function for the harmfulness vs helpfulness evaluation experiment.
    """
    # Parse command line arguments
    args = GenerationArgsHelpfulness()
    print("Experiment Configuration:")
    print(args)
    print("=" * 60)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    
    # Get model vocabulary for analysis
    vocabulary = tokenizer.get_vocab()
    
    # Setup reading vectors for representation engineering
    train_data, train_labels, _ = setup_reading_vectors(args)
    
    # Initialize model wrapper for REPE interventions
    wrap_model = WrapModel(model, tokenizer, train_data, train_labels)
    pca_vectors, pca_signs, layer_ids_injections = wrap_model.prepare_wrapped_model()
    
    # Parse dataset names from arguments
    dataset_names = args.dataset_names.split(',')
    print(f"Processing {len(dataset_names)} dataset(s): {dataset_names}")
    
    # Generate coefficient range for testing
    coefficient_range = list(np.round(np.arange(args.start_coeff, args.end_coeff, args.coeff_step), 2))
    print(f"Testing coefficient range: {coefficient_range[0]} to {coefficient_range[-1]} (step: {args.coeff_step})")
    
    # Process each dataset
    for dataset_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Load test dataset
        dataset = load_test_dataset(dataset_path=args.dataset_path, dataset_name=dataset_name)
        
        # Initialize result containers
        (acc_mean, acc_std, p_mean, p_mean_relative, p_std, p_std_relative, 
         all_answers_dict, vector_norms, projections) = initialize_result_containers(coefficient_range)
        
        # Compute baseline metrics (no intervention)
        baseline_logits_dict, baseline_best_indices = compute_baseline_metrics(
            model, tokenizer, dataset, args
        )
        
        # Evaluate each coefficient
        for coeff in coefficient_range:
            try:
                # Evaluate current coefficient
                (probs_samples, p_relative_samples, acc_samples, 
                 all_answers, coeff_vector_norms, coeff_projections) = evaluate_coefficient(
                    model, tokenizer, dataset, wrap_model, args, coeff,
                    baseline_logits_dict, baseline_best_indices
                )
                
                # Compute statistics
                (p_mean[coeff], p_std[coeff], p_mean_relative[coeff], 
                 p_std_relative[coeff], acc_mean[coeff], acc_std[coeff]) = compute_statistics(
                    probs_samples, p_relative_samples, acc_samples
                )
                
                # Store additional results
                all_answers_dict[coeff] = all_answers
                vector_norms[coeff] = coeff_vector_norms
                projections[coeff] = coeff_projections
                
                # Print results for current coefficient
                print_coefficient_results(
                    coeff, p_mean[coeff], p_std[coeff], 
                    p_mean_relative[coeff], p_std_relative[coeff], 
                    acc_mean[coeff], acc_std[coeff]
                )
                
            except Exception as e:
                print(f"Error processing coefficient {coeff}: {e}")
                continue
        
        # Save all results for this dataset
        stats_results = {
            'acc_mean': acc_mean, 
            'acc_std': acc_std, 
            'p_mean': p_mean, 
            'p_std': p_std, 
            'p_mean_relative': p_mean_relative, 
            'p_std_relative': p_std_relative
        }
        
        save_results(stats_results, dataset_name, args.model_name, args.output_dir, 'stats')
        save_results(all_answers_dict, dataset_name, args.model_name, args.output_dir, 'answers')
        save_results(projections, dataset_name, args.model_name, args.output_dir, 'proj')
        save_results(vector_norms, dataset_name, args.model_name, args.output_dir, 'vec_norms')
        
        print(f"Completed processing dataset: {dataset_name}")
    
    print(f"\n{'='*60}")
    print("Experiment completed successfully!")
    print(f"All results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()