"""
Generation Arguments Module

This module provides argument parsing and configuration classes for different types
of language model experiments including helpfulness, safety, and quality evaluations.
It handles model-specific prompt templates and validation of experiment parameters.
"""

import argparse
from typing import List, Union


def parse_comma_separated(value: str) -> str:
    """
    Parse and validate a comma-separated string containing dataset names.
    
    Args:
        value (str): Comma-separated string of dataset names
        
    Returns:
        str: The original value if all choices are valid
        
    Raises:
        argparse.ArgumentTypeError: If any dataset name is invalid
    """
    choices = [
        'high_school_computer_science', 
        'medical_genetics', 
        'international_law', 
        'clinical_knowledge'
    ]
    values = value.split(',')
    
    # Validate each item against allowed choices
    for v in values:
        if v not in choices:
            raise argparse.ArgumentTypeError(
                f"Invalid choice: '{v}'. Choices are {', '.join(choices)}."
            )
    
    return value


def model_name_verify(value: str) -> str:
    """
    Parse and validate the provided model name against supported models.
    
    Args:
        value (str): Model name to validate
        
    Returns:
        str: The validated model name
        
    Raises:
        argparse.ArgumentTypeError: If model name is not in supported list
    """
    choices = [
        'meta-llama/Meta-Llama-3.1-8B', 
        'meta-llama/Meta-Llama-3.1-8B-Instruct', 
        'meta-llama/Llama-2-13b-hf', 
        'meta-llama/Llama-2-13b-chat-hf'
    ]
    
    if value not in choices:
        raise argparse.ArgumentTypeError(
            f"Invalid choice: '{value}'. Choices are {', '.join(choices)}."
        )
    
    return value


def prompt_template_system_and_user(model_name: str) -> str:
    """
    Generate the appropriate prompt template for system and user messages based on model type.
    
    Args:
        model_name (str): Name of the model to get template for
        
    Returns:
        str: Formatted prompt template with placeholders for system_prompt and user_message
    """
    if "Llama-3" in model_name and "Instruct" in model_name:
        return (
            "<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "{user_message}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        )
    elif "Llama-3" in model_name:
        return "{system_prompt}\n\n{user_message}"
    elif "Llama-2" in model_name and "chat" in model_name:
        return "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"
    
    # Default template for other models
    return "{system_prompt}\n\n{user_message}"


def prompt_template_user(model_name: str) -> str:
    """
    Generate the appropriate prompt template for user-only messages based on model type.
    
    Args:
        model_name (str): Name of the model to get template for
        
    Returns:
        str: Formatted prompt template with placeholder for user_message
    """
    if "Llama-3" in model_name and "Instruct" in model_name:
        return (
            "<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>"
        )
    elif "Llama-3" in model_name:
        return "{user_message}"
    elif "Llama-2" in model_name and "chat" in model_name:
        return "[INST]\n\n{user_message} [/INST]"
    
    # Default template for other models
    return "{user_message}"


class GenerationArgsHelpfulness:
    """
    Configuration class for helpfulness evaluation experiments.
    
    This class handles argument parsing and configuration for experiments that evaluate
    the helpfulness of language models, typically using MMLU datasets and measuring
    performance across different REPE coefficient values.
    """
    
    def __init__(self):
        """Initialize helpfulness experiment arguments with command line parsing."""
        parser = argparse.ArgumentParser(
            description="Argument parser for helpfulness evaluation experiments"
        )
        
        # Model configuration
        parser.add_argument(
            '--model_name', 
            default='meta-llama/Meta-Llama-3.1-8B-Instruct', 
            type=model_name_verify, 
            help='Name or path of the model to evaluate'
        )
        
        # Dataset configuration
        parser.add_argument(
            '--dataset_path', 
            default='lukaemon/mmlu', 
            type=str, 
            help='Path or name of the dataset to use for evaluation'
        )
        parser.add_argument(
            '--dataset_names', 
            default='international_law,medical_genetics,clinical_knowledge,high_school_computer_science', 
            type=parse_comma_separated, 
            help='Comma-separated list of dataset names to evaluate'
        )
        
        # Coefficient range configuration
        parser.add_argument(
            '--start_coeff', 
            default=-4.0, 
            type=float, 
            help='Starting coefficient for REPE intervention range'
        )
        parser.add_argument(
            '--end_coeff', 
            default=3.2, 
            type=float, 
            help='Ending coefficient for REPE intervention range'
        )
        parser.add_argument(
            '--coeff_step', 
            default=0.5, 
            type=float, 
            help='Step size for coefficient range'
        )
        
        # Experiment parameters
        parser.add_argument(
            '--num_instructions', 
            default=16, 
            type=int, 
            help='Number of instructions to process per experiment'
        )
        parser.add_argument(
            '--num_samples', 
            default=1, 
            type=int, 
            help='Number of samples to generate per instruction'
        )
        
        # Reading vectors configuration
        parser.add_argument(
            '--is_synth_reading_vectors', 
            action='store_true', 
            help='Use synthetic reading vectors instead of dataset-derived ones'
        )
        
        # Output configuration
        parser.add_argument(
            '--output_dir', 
            default=None, 
            type=str, 
            help='Directory to save experiment results'
        )
        
        # Parse arguments and store as instance variables
        args = parser.parse_args()
        self._initialize_from_args(args)
    
    def _initialize_from_args(self, args: argparse.Namespace) -> None:
        """
        Initialize instance variables from parsed arguments.
        
        Args:
            args: Parsed command line arguments
        """
        self.model_name = args.model_name
        self.dataset_path = args.dataset_path
        self.dataset_names = args.dataset_names
        self.start_coeff = args.start_coeff
        self.end_coeff = args.end_coeff
        self.coeff_step = args.coeff_step
        self.num_instructions = args.num_instructions
        self.num_samples = args.num_samples
        self.is_synth_reading_vectors = args.is_synth_reading_vectors
        self.output_dir = args.output_dir
        
        # Generate model-specific templates
        self.template_system_and_user = prompt_template_system_and_user(self.model_name)
        self.template_user = prompt_template_user(self.model_name)
        
        # Sampling configuration (disabled for reproducibility)
        self.do_sample = False
    
    def __str__(self) -> str:
        """Return a formatted string representation of all configuration parameters."""
        return (
            f"Model Name or Path: {self.model_name}\n"
            f"Dataset Path: {self.dataset_path}\n"
            f"Dataset Names: {self.dataset_names}\n"
            f"Start Coeff: {self.start_coeff}\n"
            f"End Coeff: {self.end_coeff}\n"
            f"Coeff Step: {self.coeff_step}\n"
            f"Number of Instructions: {self.num_instructions}\n"
            f"Number of Samples: {self.num_samples}\n"
            f"Is Synthetic Reading Vectors: {self.is_synth_reading_vectors}\n"
            f"Output directory: {self.output_dir}\n"
            f"Template System and User: {self.template_system_and_user}\n"
            f"Template User: {self.template_user}\n"
            f"Do Sample: {self.do_sample}\n"
        )


class GenerationArgsSafety:
    """
    Configuration class for safety evaluation experiments.
    
    This class handles argument parsing and configuration for experiments that evaluate
    the safety behavior of language models when responding to potentially harmful prompts,
    measuring the impact of REPE interventions on safety.
    """
    
    def __init__(self):
        """Initialize safety experiment arguments with command line parsing."""
        parser = argparse.ArgumentParser(
            description="Argument parser for safety evaluation experiments"
        )
        
        # Model configuration
        parser.add_argument(
            '--model_name', 
            default='meta-llama/Meta-Llama-3.1-8B-Instruct', 
            type=model_name_verify, 
            help='Name or path of the model to evaluate'
        )
        
        # Dataset configuration
        parser.add_argument(
            '--dataset_path', 
            default='justinphan3110/harmful_harmless_instructions', 
            # default='McGill-NLP/stereoset', 
            type=str, 
            help='Path or name of the safety dataset'
        )
        parser.add_argument(
            '--dataset_names', 
            default=None, 
            type=str, 
            help='Specific dataset configuration name (optional)'
        )
        
        # Coefficient range configuration
        parser.add_argument(
            '--start_coeff', 
            default=-4.0, 
            type=float, 
            help='Starting coefficient for REPE intervention range'
        )
        parser.add_argument(
            '--end_coeff', 
            default=4.2, 
            type=float, 
            help='Ending coefficient for REPE intervention range'
        )
        parser.add_argument(
            '--coeff_step', 
            default=0.5, 
            type=float, 
            help='Step size for coefficient range'
        )
        
        # Experiment parameters
        parser.add_argument(
            '--num_instructions', 
            default=96, 
            type=int, 
            help='Number of instructions to process per experiment'
        )
        parser.add_argument(
            '--num_samples', 
            default=1, 
            type=int, 
            help='Number of samples to generate per instruction'
        )
        
        # Reading vectors configuration
        parser.add_argument(
            '--is_synth_reading_vectors', 
            action='store_true', 
            help='Use synthetic reading vectors instead of dataset-derived ones'
        )
        
        # Output configuration
        parser.add_argument(
            '--output_dir', 
            default=None, 
            type=str, 
            help='Directory to save experiment results'
        )
        
        # Parse arguments and store as instance variables
        args = parser.parse_args()
        self._initialize_from_args(args)
    
    def _initialize_from_args(self, args: argparse.Namespace) -> None:
        """
        Initialize instance variables from parsed arguments.
        
        Args:
            args: Parsed command line arguments
        """
        self.model_name = args.model_name
        self.dataset_path = args.dataset_path
        self.dataset_names = args.dataset_names
        self.start_coeff = args.start_coeff
        self.end_coeff = args.end_coeff
        self.coeff_step = args.coeff_step
        self.num_instructions = args.num_instructions
        self.num_samples = args.num_samples
        self.is_synth_reading_vectors = args.is_synth_reading_vectors
        self.output_dir = args.output_dir
        
        # Generate model-specific templates
        self.template_system_and_user = prompt_template_system_and_user(self.model_name)
        self.template_user = prompt_template_user(self.model_name)
        
        # Sampling configuration (disabled for reproducibility)
        self.do_sample = False
    
    def __str__(self) -> str:
        """Return a formatted string representation of all configuration parameters."""
        return (
            f"Model Name or Path: {self.model_name}\n"
            f"Dataset Path: {self.dataset_path}\n"
            f"Dataset Names: {self.dataset_names}\n"
            f"Start Coeff: {self.start_coeff}\n"
            f"End Coeff: {self.end_coeff}\n"
            f"Coeff Step: {self.coeff_step}\n"
            f"Number of Instructions: {self.num_instructions}\n"
            f"Number of Samples: {self.num_samples}\n"
            f"Is Synthetic Reading Vectors: {self.is_synth_reading_vectors}\n"
            f"Output directory: {self.output_dir}\n"
            f"Template System and User: {self.template_system_and_user}\n"
            f"Template User: {self.template_user}\n"
            f"Do Sample: {self.do_sample}\n"
        )


# class GenerationArgsQuality:
#     """
#     Configuration class for quality evaluation experiments.
    
#     This class handles argument parsing and configuration for experiments that evaluate
#     the quality of language model outputs, typically using human preference datasets
#     and measuring how REPE interventions affect response quality.
#     """
    
#     def __init__(self):
#         """Initialize quality experiment arguments with command line parsing."""
#         parser = argparse.ArgumentParser(
#             description="Argument parser for quality evaluation experiments"
#         )
        
#         # Model configuration
#         parser.add_argument(
#             '--model_name', 
#             default='meta-llama/Meta-Llama-3.1-8B-Instruct', 
#             type=model_name_verify, 
#             help='Name or path of the model to evaluate'
#         )
        
#         # Dataset configuration
#         parser.add_argument(
#             '--dataset_path', 
#             default='PKU-Alignment/PKU-SafeRLHF', 
#             type=str, 
#             help='Path or name of the quality evaluation dataset'
#         )
#         parser.add_argument(
#             '--dataset_names', 
#             default=None, 
#             type=parse_comma_separated, 
#             help='Comma-separated list of dataset configurations'
#         )
        
#         # Coefficient range configuration
#         parser.add_argument(
#             '--start_coeff', 
#             default=-3.5, 
#             type=float, 
#             help='Starting coefficient for REPE intervention range'
#         )
#         parser.add_argument(
#             '--end_coeff', 
#             default=5.2, 
#             type=float, 
#             help='Ending coefficient for REPE intervention range'
#         )
#         parser.add_argument(
#             '--coeff_step', 
#             default=0.25, 
#             type=float, 
#             help='Step size for coefficient range'
#         )
        
#         # Experiment parameters
#         parser.add_argument(
#             '--num_instructions', 
#             default=48, 
#             type=int, 
#             help='Number of instructions to process per experiment'
#         )
#         parser.add_argument(
#             '--num_samples', 
#             default=1, 
#             type=int, 
#             help='Number of samples to generate per instruction'
#         )
        
#         # Reading vectors configuration
#         parser.add_argument(
#             '--is_synth_reading_vectors', 
#             action='store_true', 
#             help='Use synthetic reading vectors instead of dataset-derived ones'
#         )
        
#         # Output configuration
#         parser.add_argument(
#             '--output_dir', 
#             default=None, 
#             type=str, 
#             help='Directory to save experiment results'
#         )
        
#         # Parse arguments and store as instance variables
#         args = parser.parse_args()
#         self._initialize_from_args(args)
    
#     def _initialize_from_args(self, args: argparse.Namespace) -> None:
#         """
#         Initialize instance variables from parsed arguments.
        
#         Args:
#             args: Parsed command line arguments
#         """
#         self.model_name = args.model_name
#         self.dataset_path = args.dataset_path
#         self.dataset_names = args.dataset_names
#         self.start_coeff = args.start_coeff
#         self.end_coeff = args.end_coeff
#         self.coeff_step = args.coeff_step
#         self.num_instructions = args.num_instructions
#         self.num_samples = args.num_samples
#         self.is_synth_reading_vectors = args.is_synth_reading_vectors
#         self.output_dir = args.output_dir
        
#         # Generate model-specific templates
#         self.template_system_and_user = prompt_template_system_and_user(self.model_name)
#         self.template_user = prompt_template_user(self.model_name)
        
#         # Sampling configuration (disabled for reproducibility)
#         self.do_sample = False
    
#     def __str__(self) -> str:
#         """Return a formatted string representation of all configuration parameters."""
#         return (
#             f"Model Name or Path: {self.model_name}\n"
#             f"Dataset Path: {self.dataset_path}\n"
#             f"Dataset Names: {self.dataset_names}\n"
#             f"Start Coeff: {self.start_coeff}\n"
#             f"End Coeff: {self.end_coeff}\n"
#             f"Coeff Step: {self.coeff_step}\n"
#             f"Number of Instructions: {self.num_instructions}\n"
#             f"Number of Samples: {self.num_samples}\n"
#             f"Is Synthetic Reading Vectors: {self.is_synth_reading_vectors}\n"
#             f"Output directory: {self.output_dir}\n"
#             f"Template System and User: {self.template_system_and_user}\n"
#             f"Template User: {self.template_user}\n"
#             f"Do Sample: {self.do_sample}\n"
#         )