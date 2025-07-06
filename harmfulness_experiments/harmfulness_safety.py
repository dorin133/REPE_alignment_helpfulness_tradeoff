"""
Harmfulness Safety Evaluation Script

This script evaluates the safety behavior of language models using Representation Engineering (REPE).
It measures how models respond to harmful prompts with different coefficient interventions,
analyzing the trade-off between helpfulness and safety.
"""

import csv
import torch
import json
import os
import numpy as np
import sys
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from repe.rep_control_reading_vec import WrappedReadingVecModel
from utils.GenArgs import GenerationArgsSafety
from utils.WrapModel import WrapModel
from utils.generate_reading_vectors import Synthetic_ReadingVectors_Harmfulness, ReadingVectors_Harmfulness
from repe import repe_pipeline_registry
from utils.utils import generate_responses, feed_forward_responses, feed_dialog_behavior
from utils.utils import load_test_dataset

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
    
    # Load model with appropriate configuration
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


def setup_reading_vectors(args: GenerationArgsSafety) -> Tuple[List, List, List]:
    """
    Set up reading vectors for representation engineering.
    
    Args:
        args: Configuration arguments containing experiment parameters
        
    Returns:
        Tuple of (train_data, train_labels, test_data)
    """
    if args.is_synth_reading_vectors:
        # Use synthetic reading vectors for alignment experiments
        model_name_for_generation = (
            'meta-llama/Meta-Llama-3.1-8B-Instruct' if "Llama-3" in args.model_name 
            else 'meta-llama/Llama-2-13b-chat-hf'
        )
        
        reading_vec_dataset_save_path = (
            f'/path/to/data/reading_vec_datasets/reading_vec_dataset_'
            f'{args.model_name.replace("/","_").replace("-Instruct", "")}.json'
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


def process_training_data_for_chat_models(train_data: List[str], train_labels: List, 
                                        model_name: str) -> List[str]:
    """
    Process training data for chat/instruct models by adding appropriate response patterns.
    
    Args:
        train_data: List of training examples
        train_labels: Corresponding labels for training data
        model_name: Name of the model being used
        
    Returns:
        Processed training data with added response patterns
    """
    if not ('chat' in model_name or 'Instruct' in model_name):
        return train_data
    
    processed_data = []
    for i, elem in enumerate(train_data):
        # Determine response pattern based on label
        if train_labels[i//2][i%2]:
            pattern_to_add = 'Sure! Here\'s a solution for the task'
        else:
            pattern_to_add = 'I can\'t help you with this task'
        
        # Add pattern to the training example
        processed_example = elem.split("\n\nAnswer:")[0] + f"\n\nAnswer: {pattern_to_add}"
        processed_data.append(processed_example)
    
    return processed_data


def prepare_test_dataset(dataset_name: str, harmful_test_data: List[str], 
                        dataset_path: str = None) -> Dataset:
    """
    Prepare the test dataset for evaluation.
    
    Args:
        dataset_name: Name of the dataset to load
        harmful_test_data: Pre-prepared harmful test examples
        dataset_path: Optional path to dataset files
        
    Returns:
        Dataset object ready for evaluation
    """
    if dataset_name == 'harmful_test_data':
        return Dataset.from_dict({'input': harmful_test_data})
    else:
        return load_test_dataset(dataset_path=dataset_path, dataset_name=dataset_name)


def evaluate_model_safety(model: Any, tokenizer: Any, wrap_model: WrapModel, 
                         dataset: Dataset, args: GenerationArgsSafety) -> Tuple[Dict, Dict, Dict]:
    """
    Evaluate model safety across different coefficient values.
    
    Args:
        model: The language model to evaluate
        tokenizer: Model tokenizer
        wrap_model: Wrapper for REPE interventions
        dataset: Test dataset
        args: Configuration arguments
        
    Returns:
        Tuple of (behavior_means, behavior_stds, all_answers)
    """
    # Generate coefficient range for testing
    coefficient_range = list(np.round(np.arange(args.start_coeff, args.end_coeff, args.coeff_step), 1))
    
    # Initialize result containers
    behavior_harmful_mean = {coeff: 0 for coeff in coefficient_range}
    behavior_harmful_std = {coeff: 0 for coeff in coefficient_range}
    all_answers_dict = {coeff: {} for coeff in coefficient_range}
    
    # Evaluate each coefficient
    for coeff in coefficient_range:
        print(f"\nEvaluating coefficient: {coeff}")
        
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
            template_format='default',
            batch_size=16,
            do_sample=False
        )
        
        # Get first-token logits via single forward pass
        all_logits_forward_pass = feed_forward_responses(
            model, 
            tokenizer, 
            dataset, 
            args, 
            template_format='default', 
            batch_size=16
        )
        
        # Analyze dialog behavior (agreement vs refusal)
        behavior_all_samples = feed_dialog_behavior(
            tokenizer, 
            dataset, 
            args, 
            all_answers, 
            batch_size=16
        )
        
        # Calculate statistics
        behavior_harmful_mean[coeff] = np.nanmean(np.nanmean(behavior_all_samples, axis=0))
        behavior_harmful_std[coeff] = np.nanmean(np.nanstd(behavior_all_samples, axis=0))
        
        print(f'Mean behavior score for coeff = {coeff}: {behavior_harmful_mean[coeff]:.4f}')
        print(f'Std behavior score for coeff = {coeff}: {behavior_harmful_std[coeff]:.4f}')
        
        # Store answers for detailed analysis
        all_answers_dict[coeff] = all_answers
    
    return behavior_harmful_mean, behavior_harmful_std, all_answers_dict


def save_experiment_results(behavior_means: Dict, behavior_stds: Dict, 
                           all_answers: Dict, model_name: str, output_dir: str) -> None:
    """
    Save experiment results to JSON files.
    
    Args:
        behavior_means: Dictionary of mean behavior scores by coefficient
        behavior_stds: Dictionary of standard deviation scores by coefficient
        all_answers: Dictionary of all generated answers by coefficient
        model_name: Name of the evaluated model
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize model name for filename
    safe_model_name = model_name.replace("/", "_")
    
    # Save statistics
    stats_filename = f'{output_dir}/safety_harmfulness_{safe_model_name}_stats_sample.json'
    with open(stats_filename, 'w') as file:
        results = {
            'behavior_harmful_mean': behavior_means, 
            'behavior_harmful_std': behavior_stds
        }
        json.dump(results, file, indent=2)
    
    # Save detailed answers
    answers_filename = f'{output_dir}/safety_harmfulness_{safe_model_name}_answers_sample.json'
    with open(answers_filename, 'w') as file:
        json.dump(all_answers, file, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  Statistics: {stats_filename}")
    print(f"  Answers: {answers_filename}")


def main():
    """
    Main execution function for the harmfulness safety evaluation experiment.
    """
    # Parse command line arguments
    args = GenerationArgsSafety()
    print("Experiment Configuration:")
    print(args)
    print("-" * 50)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    
    # Setup reading vectors for representation engineering
    train_data, train_labels, test_data = setup_reading_vectors(args)
    
    # Process training data for chat models if needed
    if args.is_synth_reading_vectors:
        train_data = process_training_data_for_chat_models(train_data, train_labels, args.model_name)
    
    # Prepare harmful test data (first 100 examples)
    harmful_test_data = test_data[1::2][:100]
    
    # Parse dataset names
    dataset_names = (
        args.dataset_names.split(',') if args.dataset_names 
        else ['harmful_test_data']
    )
    
    # Initialize model wrapper for REPE interventions
    wrap_model = WrapModel(model, tokenizer, train_data, train_labels)
    pca_vectors, pca_signs, layer_ids_injections = wrap_model.prepare_wrapped_model()
    
    print(f"Processing {len(dataset_names)} dataset(s): {dataset_names}")
    
    # Evaluate each dataset
    for dataset_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Prepare test dataset
        dataset = prepare_test_dataset(
            dataset_name, 
            harmful_test_data, 
            getattr(args, 'dataset_path', None)
        )
        
        # Run safety evaluation
        behavior_means, behavior_stds, all_answers = evaluate_model_safety(
            model, tokenizer, wrap_model, dataset, args
        )
        
        # Save results
        save_experiment_results(
            behavior_means, behavior_stds, all_answers, 
            args.model_name, args.output_dir
        )
    
    print(f"\n{'='*60}")
    print("Experiment completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()