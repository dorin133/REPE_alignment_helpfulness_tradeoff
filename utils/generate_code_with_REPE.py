"""
Code Generation with Representation Engineering (REPE) Module

This module generates code using language models with REPE interventions applied.
It evaluates how representation engineering affects code generation quality on
the HumanEval dataset by applying different coefficient values to control vectors.
"""

import torch
import json
import random
import os
import sys
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from datasets import load_dataset
from repe.rep_control_reading_vec import WrappedReadingVecModel
from repe import repe_pipeline_registry

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from utils.GenArgs import GenerationArgsHelpfulness
from utils.generate_reading_vectors import ReadingVectors_Fairness
from utils import read_json_if_exists, sample_model, set_seed

# Register REPE pipeline components
repe_pipeline_registry()

# Set random seed for reproducibility
random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_model_and_tokenizer(model_name_or_path: str) -> Tuple[Any, Any]:
    """
    Load and configure the language model and tokenizer.
    
    Args:
        model_name_or_path (str): Path to the model directory or model identifier
        
    Returns:
        Tuple[Any, Any]: (model, tokenizer) configured for code generation
    """
    print(f"Loading model from: {model_name_or_path}")
    
    # Load model with appropriate configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.float16, 
        token=True
    ).eval()
    model.to(device)
    
    # Configure tokenizer based on model architecture
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        use_fast=use_fast_tokenizer, 
        padding_side="left", 
        legacy=False, 
        token=True
    )
    
    # Set up tokenizer special tokens
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    
    print("Model loading completed!")
    return model, tokenizer


def load_humaneval_dataset() -> Dict[str, Dict]:
    """
    Load the HumanEval dataset for code generation evaluation.
    
    Returns:
        Dict[str, Dict]: Dictionary mapping task_id to problem data
    """
    print("Loading HumanEval dataset...")
    human_eval_data = load_dataset("openai/openai_humaneval")
    human_eval_dict = {q['task_id']: q for q in human_eval_data['test']}
    print(f"Loaded {len(human_eval_dict)} HumanEval problems")
    return human_eval_dict


def setup_reading_vectors(model: Any, tokenizer: Any) -> Tuple[Dict, Dict]:
    """
    Set up reading vectors for REPE interventions using fairness dataset.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        
    Returns:
        Tuple[Dict, Dict]: (pca_vectors, pca_signs) for REPE interventions
    """
    print("Setting up reading vectors...")
    
    # Load fairness reading vectors
    args = GenerationArgsHelpfulness()
    print(f"Args: {args}")
    
    reading_vecs = ReadingVectors_Fairness(args)
    train_data, train_labels, _ = reading_vecs.load_reading_vec_dataset()
    
    # Configure representation reading parameters
    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = 'pca'
    direction_finder_kwargs = {"n_components": 1}
    
    # Create representation reading pipeline
    rep_reading_pipeline = pipeline(
        "rep-reading", 
        model=model, 
        tokenizer=tokenizer, 
        device=device
    )
    
    # Compute reading directions
    rep_reader = rep_reading_pipeline.get_directions(
        train_data, 
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        n_difference=n_difference, 
        train_labels=train_labels, 
        direction_method=direction_method, 
        direction_finder_kwargs=direction_finder_kwargs
    )
    
    pca_vectors = rep_reader.directions
    pca_signs = rep_reader.direction_signs
    
    print("Reading vectors setup completed!")
    return pca_vectors, pca_signs


def setup_wrapped_model(model: Any, tokenizer: Any, layer_ids: List[int]) -> WrappedReadingVecModel:
    """
    Set up the wrapped model for REPE interventions.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        layer_ids: List of layer IDs for intervention
        
    Returns:
        WrappedReadingVecModel: Configured wrapped model
    """
    print("Setting up wrapped model for REPE interventions...")
    
    block_name = "decoder_block"
    wrapped_model = WrappedReadingVecModel(model, tokenizer)
    wrapped_model.unwrap()
    wrapped_model.wrap_block(layer_ids, block_name=block_name)
    
    print("Wrapped model setup completed!")
    return wrapped_model


def get_layer_ids_for_model_size(model: Any) -> List[int]:
    """
    Determine appropriate layer IDs for intervention based on model size.
    
    Args:
        model: The language model
        
    Returns:
        List[int]: List of layer IDs for intervention
    """
    # Calculate model size in billions of parameters
    model_size = sum(p.numel() for p in model.parameters()) // 1000000000
    
    if model_size <= 8:  # 7B or 8B models
        return list(range(-18, -23, -1))
    else:  # 13B models
        return list(range(-25, -33, -1))


def load_existing_generations(generation_path: str) -> Dict[float, Dict]:
    """
    Load existing code generations from file if available.
    
    Args:
        generation_path (str): Path to the generation results file
        
    Returns:
        Dict[float, Dict]: Dictionary mapping coefficients to generations
    """
    generation_dict_string_keys = read_json_if_exists(generation_path)
    
    if generation_dict_string_keys:
        print("Loaded existing generation dict with:")
        for key in generation_dict_string_keys:
            for inner_key in generation_dict_string_keys[key]:
                print(f"{key}: {inner_key}")
        print("-" * 50)
        
        # Convert string keys to float for coefficient values
        generation_dict = {
            float(key): generation_dict_string_keys[key] 
            for key in generation_dict_string_keys
        }
    else:
        generation_dict = {}
        print("No existing generation file found, starting fresh.")
    
    return generation_dict


def apply_repe_intervention(wrapped_model: WrappedReadingVecModel, 
                           model: Any, 
                           layer_ids: List[int],
                           pca_vectors: Dict, 
                           pca_signs: Dict, 
                           coeff: float) -> None:
    """
    Apply REPE intervention with specified coefficient to the wrapped model.
    
    Args:
        wrapped_model: Wrapped model for interventions
        model: Base language model
        layer_ids: List of layer IDs for intervention
        pca_vectors: PCA vectors by layer
        pca_signs: PCA signs by layer
        coeff: Coefficient for intervention strength
    """
    activations = {}
    
    for layer in layer_ids:
        # Compute intervention vector for this layer
        v = torch.tensor(pca_vectors[layer] * pca_signs[layer][0])
        v = (v / torch.norm(v)).cpu()
        activations[layer] = torch.tensor(coeff * v).to(model.device).half()
    
    # Apply interventions to the wrapped model
    wrapped_model.reset()
    wrapped_model.set_controller(layer_ids, activations, "decoder_block")


def generate_code_with_coefficients(wrapped_model: WrappedReadingVecModel,
                                   model: Any,
                                   tokenizer: Any,
                                   human_eval_dict: Dict[str, Dict],
                                   layer_ids: List[int],
                                   pca_vectors: Dict,
                                   pca_signs: Dict,
                                   coefficient_range: List[float],
                                   generation_path: str,
                                   num_samples: int = 16,
                                   batch_size: int = 4) -> Dict[float, Dict]:
    """
    Generate code for HumanEval problems with different REPE coefficients.
    
    Args:
        wrapped_model: Wrapped model for interventions
        model: Base language model
        tokenizer: Model tokenizer
        human_eval_dict: HumanEval dataset dictionary
        layer_ids: List of layer IDs for intervention
        pca_vectors: PCA vectors by layer
        pca_signs: PCA signs by layer
        coefficient_range: List of coefficients to test
        generation_path: Path to save results
        num_samples: Number of samples per problem
        batch_size: Batch size for generation
        
    Returns:
        Dict[float, Dict]: Generated code by coefficient and task
    """
    # Load existing generations to resume if needed
    generation_dict = load_existing_generations(generation_path)
    
    for i, coeff in enumerate(coefficient_range):
        print(f"Processing coefficient: {coeff}")
        
        # Apply REPE intervention for current coefficient
        apply_repe_intervention(
            wrapped_model, model, layer_ids, 
            pca_vectors, pca_signs, coeff
        )
        
        # Initialize coefficient entry if not exists
        if coeff not in generation_dict:
            generation_dict[coeff] = {}
        
        # Generate code for each HumanEval problem
        for j, task_id in enumerate(human_eval_dict):
            # Skip if already generated
            if task_id in generation_dict[coeff]:
                continue
            
            print(f"  Generating for task: {task_id}")
            question = human_eval_dict[task_id]
            
            # Generate code samples
            generation_dict[coeff][task_id] = sample_model(
                model, tokenizer, question, 
                num_samples=num_samples, 
                batch_size=batch_size
            )
            
            # Save progress periodically
            if j % 5 == 0:
                with open(generation_path, 'w') as file:
                    json.dump(generation_dict, file, indent=2)
                print(f"    Progress saved after {j+1} tasks")
    
    # Final save
    with open(generation_path, 'w') as file:
        json.dump(generation_dict, file, indent=2)
    print(f"All generations completed and saved to: {generation_path}")
    
    return generation_dict


def main():
    """
    Main execution function for code generation with REPE interventions.
    """
    print("=" * 60)
    print("Code Generation with REPE Interventions")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Configuration
    model_name_or_path = "/path/to/models/llama-2-13b-chat/"
    generation_path = '/path/to/outputs/code_generations/code_generations_results_fairness_bias.json'
    
    # Coefficient range for testing
    coefficient_range = [
        -3.0, -2.5, -2.0, -1.5, -1.4, -1.2, -1.0, -0.8, -0.6, -0.5, 
        -0.4, -0.2, 0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 
        2.0, 2.5, 3.0
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(generation_path), exist_ok=True)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name_or_path)
    
    # Load HumanEval dataset
    human_eval_dict = load_humaneval_dataset()
    
    # Get model vocabulary (for potential future use)
    vocabulary = tokenizer.get_vocab()
    print(f"Vocabulary size: {len(vocabulary)}")
    
    # Setup reading vectors for REPE
    pca_vectors, pca_signs = setup_reading_vectors(model, tokenizer)
    
    # Determine layer IDs based on model size
    layer_ids = get_layer_ids_for_model_size(model)
    print(f"Using layer IDs for intervention: {layer_ids}")
    
    # Setup wrapped model for REPE
    wrapped_model = setup_wrapped_model(model, tokenizer, layer_ids)
    
    # Generate code with different coefficients
    generation_dict = generate_code_with_coefficients(
        wrapped_model, model, tokenizer, human_eval_dict,
        layer_ids, pca_vectors, pca_signs, coefficient_range,
        generation_path, num_samples=16, batch_size=4
    )
    
    print("=" * 60)
    print("Code generation experiment completed!")
    print(f"Results saved to: {generation_path}")
    print(f"Generated code for {len(coefficient_range)} coefficients")
    print(f"Processed {len(human_eval_dict)} HumanEval problems")
    print("=" * 60)


if __name__ == "__main__":
    main()