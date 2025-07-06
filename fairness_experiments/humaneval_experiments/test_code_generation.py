"""
HumanEval Code Generation Testing Module

This module provides functionality for testing code generation models against the HumanEval dataset.
It includes code parsing, function extraction, evaluation, and result visualization capabilities
for analyzing the performance of language models on coding tasks with various REPE coefficients.
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional, Any
from code_runner import test_humaneval_function


def parse_code_from_response(model_response: str) -> Optional[str]:
    """
    Extract code from model response that may contain markdown code blocks.
    
    Args:
        model_response (str): Raw response from language model
        
    Returns:
        Optional[str]: Extracted code or False if no code blocks found
    """
    split_model_output = model_response.split("```")
    
    if len(split_model_output) == 1:
        return False
    
    # Extract code from markdown block and clean language identifiers
    fixed_code = split_model_output[1].replace('python', '').replace('Python', '')
    return fixed_code


def extract_import_statements(code: str) -> str:
    """
    Extract all import statements from Python code.
    
    Args:
        code (str): Python code string
        
    Returns:
        str: All import statements joined with newlines
    """
    lines = code.split('\n')
    
    # Filter lines that contain import statements
    import_lines = [
        line for line in lines 
        if line.strip().startswith('import') or line.strip().startswith('from')
    ]
    
    return '\n'.join(import_lines)


def find_first_non_indented_line(text: str, start_index: int) -> int:
    """
    Find the first line that doesn't start with whitespace after a given index.
    
    Args:
        text (str): Multi-line text to search
        start_index (int): Starting line index
        
    Returns:
        int: Line index of first non-indented line, or -1 if not found
    """
    lines = text.splitlines()
    
    for i in range(start_index, len(lines)):
        # Check if the line exists and doesn't start with space or tab
        if lines[i] and not lines[i].startswith((' ', '\t')):
            return i
    
    return -1  # No non-indented line found


def find_first_function_definition(text: str, start_index: int) -> int:
    """
    Find the first line containing a function definition after a given index.
    
    Args:
        text (str): Multi-line text to search
        start_index (int): Starting line index
        
    Returns:
        int: Line index of first function definition, or -1 if not found
    """
    lines = text.splitlines()
    
    for i in range(start_index, len(lines)):
        if 'def ' in lines[i]:
            return i
    
    return -1


def extract_line_range(text: str, start_line: int, end_line: int) -> str:
    """
    Extract a range of lines from multi-line text.
    
    Args:
        text (str): Multi-line text
        start_line (int): Starting line index (inclusive)
        end_line (int): Ending line index (exclusive)
        
    Returns:
        str: Extracted lines joined with newlines
    """
    lines = text.splitlines()
    return '\n'.join(lines[start_line:end_line])


def extract_function_from_text(mixed_string: str, function_name: str, 
                              starting_index: int = 0) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract a function definition from mixed text content.
    
    Args:
        mixed_string (str): Text containing function definition
        function_name (str): Name of function to extract (currently unused but kept for API compatibility)
        starting_index (int): Starting index for search
        
    Returns:
        Tuple[Optional[str], Optional[int]]: (extracted function code, end index) or (None, None)
    """
    if starting_index is None:
        return None, None
    
    # Find the first function definition
    def_index = find_first_function_definition(mixed_string, starting_index)
    if def_index == -1:
        return None, None
    
    # Find where the function ends (first non-indented line)
    end_index = find_first_non_indented_line(mixed_string, def_index + 1)
    if end_index == -1:
        return None, None
    
    function_code = extract_line_range(mixed_string, def_index, end_index)
    return function_code, end_index


def extract_function_with_regex(mixed_string: str, function_name: str) -> Optional[str]:
    """
    Extract function using regex pattern matching for markdown code blocks.
    
    Args:
        mixed_string (str): Text containing function in markdown code block
        function_name (str): Name of the function to extract
        
    Returns:
        Optional[str]: Extracted function code or None if not found
    """
    # Pattern to match function in markdown code blocks
    pattern = rf"```\s*\n*def\s+{re.escape(function_name)}\s*\([^)]*\):(?:(?!\n```)[\s\S])*\n```"
    match = re.search(pattern, mixed_string, re.MULTILINE)
    
    if match:
        function_code = match.group(0)
        # Remove markdown code block markers
        function_code = re.sub(r'^```\s*\n*', '', function_code)
        function_code = re.sub(r'\n*```\s*$', '', function_code)
        return function_code.strip()
    
    return None


def evaluate_humaneval_generations(all_generations: Dict[str, Dict], 
                                  data_dict: Dict[str, Dict]) -> Dict[float, Tuple[float, float]]:
    """
    Evaluate code generations against HumanEval dataset across different coefficients.
    
    Args:
        all_generations: Dictionary mapping coefficients to generations
                        Format: {coeff: {task_id: [generation1, generation2, ...]}}
        data_dict: HumanEval dataset dictionary mapping task_id to problem data
        
    Returns:
        Dict[float, Tuple[float, float]]: Results mapping coefficient to (mean_success, std_error)
    """
    results = {}
    
    for coeff in all_generations:
        print(f"Evaluating coefficient: {coeff}")
        curr_generations = all_generations[coeff]
        success_percentage_list = []
        
        for task_id in curr_generations:
            print(f"  Processing task: {task_id}")
            curr_generation_batch = curr_generations[task_id]
            curr_problem = data_dict[task_id]
            
            success_list = []
            
            for i, generation in enumerate(curr_generation_batch):
                entry_point = curr_problem['entry_point']
                curr_answer = generation
                
                # Add function definition if missing
                add_prompt = f'def {entry_point}' not in curr_answer
                if add_prompt:
                    curr_answer = f"{curr_problem['prompt']}\n{curr_answer}"
                
                # Extract import statements (currently unused but kept for potential use)
                curr_imports = extract_import_statements(curr_answer)
                
                # Try to extract function code
                function_codes_to_try = []
                option1, index = extract_function_from_text(
                    curr_answer + "\nbuffer_text", 
                    curr_problem['entry_point']
                )
                function_codes_to_try.append(option1)
                
                # Test each extracted function
                is_pass = False
                for curr_function_code in function_codes_to_try:
                    if curr_function_code is None:
                        continue
                    
                    try:
                        is_pass = test_humaneval_function(curr_problem, curr_function_code)
                        if is_pass:
                            break
                    except Exception as e:
                        print(f"    Error testing function: {e}")
                        is_pass = False
                
                success_list.append(is_pass)
            
            # Calculate success percentage for this task
            success_percentage = np.sum(success_list) / len(success_list) if success_list else 0
            success_percentage_list.append(success_percentage)
        
        # Calculate overall statistics for this coefficient
        if success_percentage_list:
            mean_success = np.average(success_percentage_list)
            std_error = np.std(success_percentage_list) / len(success_percentage_list) ** 0.5
            results[float(coeff)] = (mean_success, std_error)
        else:
            results[float(coeff)] = (0.0, 0.0)
    
    return results


def plot_coefficient_results(coefficients: List[float], averages: List[float], 
                           std_errors: List[float], title: str = None) -> None:
    """
    Plot success rates vs coefficients with error bars.
    
    Args:
        coefficients: List of coefficient values
        averages: List of average success rates
        std_errors: List of standard errors
        title: Optional plot title
    """
    averages = np.array(averages)
    std_errors = np.array(std_errors)
    
    plt.figure(figsize=(12, 8))
    plt.plot(coefficients, averages, 'b-', linewidth=2, marker='o', markersize=6)
    plt.fill_between(coefficients, averages - std_errors, averages + std_errors, 
                    alpha=0.3, color='blue')
    
    plt.xlabel('REPE Coefficients', fontsize=12)
    plt.ylabel('Mean Success Rate', fontsize=12)
    plt.title(title or 'HumanEval Success Rate vs. REPE Coefficients', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add some formatting
    plt.ylim(0, 1)
    plt.xlim(min(coefficients), max(coefficients))
    
    plt.tight_layout()
    plt.show()


def plot_fine_tuned_model_comparison(results_dict: Dict[str, Tuple[float, float]]) -> None:
    """
    Plot comparison of fine-tuned model checkpoints.
    
    Args:
        results_dict: Dictionary mapping model paths to (mean, std_error) tuples
    """
    # Extract base model results
    base_model_key = "/path/to/models/Meta-Llama-3.1-8B"
    base_results = [results_dict.get(base_model_key, (0.0, 0.0))]
    
    # Remove base model from dict to process checkpoints
    if base_model_key in results_dict:
        results_dict.pop(base_model_key)
    
    # Sort checkpoints by step number
    checkpoints = list(results_dict.keys())
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]) if '-' in x else 0)
    
    print(f"Processing checkpoints: {checkpoints}")
    
    # Collect all results
    all_results = base_results + [results_dict[checkpoint] for checkpoint in checkpoints]
    
    averages = [result[0] for result in all_results]
    std_errors = [result[1] for result in all_results]
    
    # Create x-axis labels
    x_labels = ['Base Model'] + [f'Step {cp.split("-")[-1]}' for cp in checkpoints]
    x_positions = np.arange(len(all_results))
    
    plot_coefficient_results(x_positions, averages, std_errors, 
                           'Fine-tuned Model Checkpoint Comparison')


def load_generation_data(file_paths: List[str]) -> Dict[str, Dict]:
    """
    Load and merge code generation data from multiple JSON files.
    
    Args:
        file_paths: List of paths to JSON files containing generation data
        
    Returns:
        Dict: Merged generation data
    """
    all_generation_dict = {}
    
    for path in file_paths:
        try:
            with open(path, 'r') as file:
                curr_generation = json.load(file)
            
            for coeff_key in curr_generation:
                if coeff_key not in all_generation_dict:
                    all_generation_dict[coeff_key] = curr_generation[coeff_key]
                else:
                    # Merge data for existing coefficient
                    for task_key in curr_generation[coeff_key]:
                        if task_key not in all_generation_dict[coeff_key]:
                            all_generation_dict[coeff_key][task_key] = curr_generation[coeff_key][task_key]
                        else:
                            # Concatenate generation lists
                            all_generation_dict[coeff_key][task_key] += curr_generation[coeff_key][task_key]
            
            print(f"Successfully loaded: {path}")
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    return all_generation_dict


def main():
    """
    Main execution function for testing code generation against HumanEval dataset.
    """
    print("Loading HumanEval dataset...")
    
    # Load HumanEval dataset
    human_eval_data = load_dataset("openai/openai_humaneval")
    human_eval_dict = {q['task_id']: q for q in human_eval_data['test']}
    
    print(f"Loaded {len(human_eval_dict)} HumanEval problems")
    
    # Define paths to generation files (anonymized)
    generation_file_paths = [
        '/path/to/data/code_generations/code_generations_results_experiment1.json',
        '/path/to/data/code_generations/code_generations_results_experiment2.json',
        '/path/to/data/code_generations/code_generations_results_experiment3.json',
        '/path/to/data/code_generations/code_generations_results_experiment4.json'
    ]
    
    print("Loading generation data...")
    all_generation_dict = load_generation_data(generation_file_paths)
    
    print(f"Loaded generations for {len(all_generation_dict)} coefficients")
    
    # Evaluate generations
    print("Starting evaluation...")
    results = evaluate_humaneval_generations(all_generation_dict, human_eval_dict)
    
    # Sort results by coefficient value
    results = dict(sorted(results.items()))
    
    # Extract data for plotting
    sorted_coefficients = list(results.keys())
    averages = [result[0] for result in results.values()]
    std_errors = [result[1] for result in results.values()]
    
    print(f"\nResults summary:")
    for coeff, (avg, std_err) in results.items():
        print(f"  Coefficient {coeff}: {avg:.3f} ± {std_err:.3f}")
    
    # Plot results
    plot_coefficient_results(sorted_coefficients, averages, std_errors)
    
    print("Evaluation completed!")


if __name__ == '__main__':
    main()