"""
Separation by Dimensionality Reduction Analysis Script

This script analyzes the separation of harmful vs. benign behaviors in language models
using dimensionality reduction techniques. It extracts logits from models with various
REPE intervention coefficients to study the representational space of harmful content.
"""

import torch
import json
import os
import sys
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple, Any
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoConfig
from repe.rep_control_reading_vec import WrappedReadingVecModel
from repe import repe_pipeline_registry

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.GenArgs import GenerationArgsHelpfulness
from utils.generate_reading_vectors import Synthetic_ReadingVectors_Harmfulness, ReadingVectors_Harmfulness

# Register REPE pipeline components
repe_pipeline_registry()


def setup_model_and_tokenizer(model_name: str, model_path: str) -> Tuple[Any, Any]:
    """
    Load and configure the language model and tokenizer.
    
    Args:
        model_name (str): Name of the model (e.g., "Llama-2-13b")
        model_path (str): Path to the model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Load model with appropriate configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        token=True
    ).eval()
    
    # Configure tokenizer based on model architecture
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
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


def prepare_training_data(model_name: str, train_data: List[str], use_synthetic: bool = False) -> Tuple[List[str], str]:
    """
    Prepare training data based on model type (raw vs chat).
    
    Args:
        model_name (str): Name of the model to determine data preparation strategy
        train_data (List[str]): Raw training data from reading vectors
        use_synthetic (bool): Whether synthetic reading vectors were used
        
    Returns:
        Tuple of (processed_train_data, save_data_dir)
    """
    if model_name == "Llama-2-13b" and not use_synthetic:
        # Raw model - process data for Q&A format
        # Convert chat format to Q&A format for raw models
        processed_train_data = []
        for elem in train_data:
            if "Request: Can you" in elem:
                # Convert to Q&A format
                processed_elem = elem.replace("Request: Can you ", "QUESTION: Can you ")
                processed_elem = processed_elem.replace("? Answer:", "?\n\nANSWER: ")
                
                # Limit answer length for raw models
                if 'ANSWER: ' in processed_elem:
                    parts = processed_elem.split('ANSWER: ')
                    answer_part = parts[1][:100] if len(parts) > 1 else ""
                    processed_elem = parts[0] + 'ANSWER: ' + answer_part
                
                processed_train_data.append(processed_elem)
            else:
                processed_train_data.append(elem)
        
        save_data_dir = 'separation_by_dr_raw_model'
    else:
        # Chat model - keep original format
        processed_train_data = train_data
        save_data_dir = 'separation_by_dr_chat_model'
    
    return processed_train_data, save_data_dir


def compute_representation_vectors(model: Any, tokenizer: Any, train_data: List[str], 
                                 train_labels: List) -> Tuple[Dict, Dict]:
    """
    Compute PCA representation vectors from training data using REPE.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        train_data: Training examples for vector computation
        train_labels: Labels for training examples
        
    Returns:
        Tuple of (pca_vectors, pca_signs) dictionaries
    """
    print("Computing representation vectors...")
    
    # Configuration for representation reading
    rep_token = -1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    n_difference = 1
    direction_method = 'pca'
    direction_finder_kwargs = {"n_components": 1}
    
    # Create representation reading pipeline
    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    
    # Compute directions using PCA
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
    
    print("Representation vectors computed successfully!")
    return pca_vectors, pca_signs


def setup_wrapped_model(model: Any, tokenizer: Any, layer_ids: List[int]) -> WrappedReadingVecModel:
    """
    Set up the wrapped model for REPE interventions.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        layer_ids: List of layer IDs for intervention
        
    Returns:
        WrappedReadingVecModel configured for interventions
    """
    wrapped_model = WrappedReadingVecModel(model, tokenizer)
    wrapped_model.unwrap()
    wrapped_model.wrap_block(layer_ids, block_name='decoder_block')
    return wrapped_model


def extract_baseline_logits(wrapped_model: WrappedReadingVecModel, tokenizer: Any,
                          test_data: List[str], layer_ids: List[int], 
                          pca_vectors: Dict, pca_signs: Dict, 
                          save_dir: str, num_instructions: int = 10) -> None:
    """
    Extract baseline logits (coefficient = 0) for test instructions.
    
    Args:
        wrapped_model: Wrapped model for interventions
        tokenizer: Model tokenizer
        test_data: Test dataset
        layer_ids: Layer IDs for intervention
        pca_vectors: PCA vectors by layer
        pca_signs: PCA signs by layer
        save_dir: Directory to save results
        num_instructions: Number of instructions to process
    """
    print("Extracting baseline logits (coefficient = 0)...")
    
    # Ensure output directory exists
    output_dir = f'/path/to/experiment/outputs/{save_dir}'
    os.makedirs(output_dir, exist_ok=True)
    
    coeff = 0  # Baseline coefficient
    
    for j, instruction in enumerate(test_data[1:2*num_instructions:2]):
        if j >= num_instructions:
            break
            
        # Encode instruction
        prompt = torch.unsqueeze(torch.tensor(tokenizer.encode(instruction)), dim=0)
        
        # Prepare activations for baseline (coefficient = 0)
        activations = {}
        for layer in layer_ids:
            v = torch.tensor(pca_vectors[layer] * pca_signs[layer][0])
            v = (v / torch.norm(v)).cpu()
            activations[layer] = torch.tensor(coeff * v).to(wrapped_model.model.device).half()
        
        # Apply intervention and extract logits
        wrapped_model.reset()
        wrapped_model.set_controller(layer_ids, activations, 'decoder_block')
        
        with torch.no_grad():
            logits_gt = wrapped_model(input_ids=prompt.cuda()).logits[0, -1]
            logits_gt = logits_gt.to(dtype=torch.float32)
        
        # Save baseline logits
        output_file = f'{output_dir}/logits_iter_gt_inst_{j}.json'
        with open(output_file, 'w') as file:
            json.dump(logits_gt.tolist(), file)
    
    print(f"Baseline logits saved to: {output_dir}")


def extract_coefficient_logits(wrapped_model: WrappedReadingVecModel, tokenizer: Any,
                             test_data: List[str], layer_ids: List[int],
                             pca_vectors: Dict, pca_signs: Dict,
                             coefficient_range: List[float], save_dir: str,
                             num_instructions: int = 10) -> None:
    """
    Extract logits for various coefficient values across test instructions.
    
    Args:
        wrapped_model: Wrapped model for interventions
        tokenizer: Model tokenizer
        test_data: Test dataset
        layer_ids: Layer IDs for intervention
        pca_vectors: PCA vectors by layer
        pca_signs: PCA signs by layer
        coefficient_range: Range of coefficients to test
        save_dir: Directory to save results
        num_instructions: Number of instructions to process
    """
    print(f"Extracting logits for {len(coefficient_range)} coefficients...")
    
    # Ensure output directory exists
    output_dir = f'/path/to/experiment/outputs/{save_dir}'
    os.makedirs(output_dir, exist_ok=True)
    
    for j, instruction in enumerate(test_data[1:2*num_instructions:2]):
        if j >= num_instructions:
            break
            
        print(f"Processing instruction {j+1}/{num_instructions}")
        
        # Encode instruction
        prompt = torch.unsqueeze(torch.tensor(tokenizer.encode(instruction)), dim=0)
        
        # Process each coefficient
        for i, coeff in tqdm(enumerate(coefficient_range), total=len(coefficient_range)):
            # Prepare activations for current coefficient
            activations = {}
            for layer in layer_ids:
                v = torch.tensor(pca_vectors[layer] * pca_signs[layer][0])
                v = (v / torch.norm(v)).cpu()
                activations[layer] = torch.tensor(coeff * v).to(wrapped_model.model.device).half()
            
            # Apply intervention and extract logits
            wrapped_model.reset()
            wrapped_model.set_controller(layer_ids, activations, 'decoder_block')
            
            with torch.no_grad():
                logits = wrapped_model(input_ids=prompt.cuda()).logits[0, -1]
                logits = logits.to(dtype=torch.float32)
            
            # Save logits for this coefficient and instruction
            output_file = f'{output_dir}/logits_iter_{i}_inst_{j}.json'
            with open(output_file, 'w') as file:
                json.dump(logits.tolist(), file)
    
    print(f"Coefficient logits saved to: {output_dir}")


def get_layer_ids_for_model_size(model: Any) -> List[int]:
    """
    Determine appropriate layer IDs for intervention based on model size.
    
    Args:
        model: The language model
        
    Returns:
        List of layer IDs for intervention
    """
    # Calculate model size in billions of parameters
    model_size = sum(p.numel() for p in model.parameters()) // 1000000000
    
    if model_size <= 8:  # 7B or 8B models
        return list(range(-18, -23, -1))
    else:  # 13B models
        return list(range(-25, -33, -1))

def main():
    """
    Main execution function for dimensionality reduction separation analysis.
    """
    # Configuration
    model_name = "Llama-2-13b"  # OPTIONS: "Llama-2-13b-chat" / "Llama-2-13b"
    model_path = "/path/to/models/llama2/Llama-2-13b/"
    num_test_instructions = 10  # Number of test instructions to process
    use_synthetic_reading_vectors = False  # Set to True to use synthetic reading vectors
    
    print("=" * 60)
    print("Dimensionality Reduction Separation Analysis")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, model_path)
    
    # Create args for reading vector setup (matching the interface from harmfulness_helpfulness.py)
    args = GenerationArgsHelpfulness()
    
    # Setup reading vectors for representation engineering (same as harmfulness_helpfulness.py)
    train_data, train_labels, test_data = setup_reading_vectors(args)
    
    # Prepare training data based on model type
    processed_train_data, save_data_dir = prepare_training_data(model_name, train_data, use_synthetic_reading_vectors)
    
    print(f"Using {'synthetic' if use_synthetic_reading_vectors else 'RLHF'} reading vectors")
    print(f"Training data examples: {len(processed_train_data)}")
    print(f"Test data examples: {len(test_data)}")
    
    # Compute representation vectors
    pca_vectors, pca_signs = compute_representation_vectors(
        model, tokenizer, processed_train_data, train_labels
    )
    
    # Determine layer IDs for intervention
    layer_ids = get_layer_ids_for_model_size(model)
    print(f"Using layer IDs for intervention: {layer_ids}")
    
    # Set up wrapped model
    wrapped_model = setup_wrapped_model(model, tokenizer, layer_ids)
    
    # Define coefficient range for analysis
    coefficient_range = [i for i in np.arange(-10, 10.2, 0.2)]
    print(f"Testing coefficient range: {coefficient_range[0]} to {coefficient_range[-1]} (step: 0.2)")
    
    # Extract baseline logits (coefficient = 0)
    extract_baseline_logits(
        wrapped_model, tokenizer, test_data, layer_ids,
        pca_vectors, pca_signs, save_data_dir, num_test_instructions
    )
    
    # Extract logits for various coefficients
    extract_coefficient_logits(
        wrapped_model, tokenizer, test_data, layer_ids,
        pca_vectors, pca_signs, coefficient_range, save_data_dir,
        num_test_instructions
    )
    
    print("=" * 60)
    print("Analysis completed successfully!")
    print(f"Results saved to: /path/to/experiment/outputs/{save_data_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()