import argparse

def parse_comma_separated(value: str):
    """Parse and validate a comma-separated string containing only 'str1', 'str2', 'str3'."""
    choices=['high_school_computer_science', 'medical_genetics', 'international_law', 'clinical_knowledge']
    values = value.split(',')
    
    # Validate each item
    for v in values:
        if v not in choices:
            raise argparse.ArgumentTypeError(f"Invalid choice: '{v}'. Choices are {', '.join(choices)}.")
    return value

def model_name_verify(value: str):
    """Parse and validate the provided model name."""
    choices=['meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B-Instruct', 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-13b-chat-hf']
    if value not in choices:
        raise argparse.ArgumentTypeError(f"Invalid choice: '{value}'. Choices are {', '.join(choices)}.")
    return value

def prompt_template_system_and_user(model_name):
    if "Llama-3" in model_name and "Instruct" in model_name:
        return "<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    elif "Llama-3" in model_name:
        return "{system_prompt}\n\n{user_message}"
    elif "Llama-2" in model_name and "chat" in model_name:
        return "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"
    return "{system_prompt}\n\n{user_message}"
    
def prompt_template_user(model_name):
    if "Llama-3" in model_name and "Instruct" in model_name:
        return "<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    elif "Llama-3" in model_name:
        return "{user_message}"
    elif "Llama-2" in model_name and "chat" in model_name:
        return "[INST]\n\n{user_message} [/INST]"
    return "{user_message}"
    
class GenerationArgsHelpfulness:
    def __init__(self):
        parser = argparse.ArgumentParser(description="parser for arguments from .py script call")
        # opt: 'meta-llama/Meta-Llama-3.1-8B-Instruct', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-13b-chat-hf'
        parser.add_argument('--model_name', default='meta-llama/Llama-2-13b-chat-hf', type=model_name_verify, help='Path for the model (huggingface or local)')
        parser.add_argument('--dataset_path', default='lukaemon/mmlu', type=str, help='Path for training_args.output_dir')
        parser.add_argument('--dataset_names', default='international_law,medical_genetics,clinical_knowledge,high_school_computer_science', type=parse_comma_separated, help='Path for training_args.output_dir')
        parser.add_argument('--start_coeff', default=-10.0, type=float, help='coeff to start the range of the norm injection of the representation vector')
        parser.add_argument('--end_coeff', default=10.2, type=float, help='coeff to end the range of the norm injection of the representation vector')
        parser.add_argument('--coeff_step', default=0.5, type=float, help='step for the range of the norm injection of the representation vector')
        parser.add_argument('--num_instructions', default=96, type=int, help='number of instructions to generate for each prompt')
        parser.add_argument('--num_samples', default=8, type=int, help='number of samples to generate for each instruction')
        parser.add_argument('--is_synth_reading_vectors', action='store_true', help='Whether to generate reading vectors synthetically (produced by the model) or load them from the dataset for REPE') 
        parser.add_argument('--output_dir', default="data/harmfulness_experiments_outputs/default_dir_helpfulness", type=str, help='Path for the output directory')
        
        args = parser.parse_args()
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
        self.template_system_and_user = prompt_template_system_and_user(self.model_name)
        self.template_user = prompt_template_user(self.model_name)
    
    def __str__(self):
        return (f"Model Name or Path: {self.model_name}\n"
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
                f"Template User: {self.template_user}\n")

class GenerationArgsSafety:
    def __init__(self):
        parser = argparse.ArgumentParser(description="parser for arguments from .py script call")
        # opt: 'meta-llama/Meta-Llama-3.1-8B-Instruct', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-13b-chat-hf'
        parser.add_argument('--model_name', default='meta-llama/Meta-Llama-3.1-8B-Instruct', type=model_name_verify, help='Path for the model (huggingface or local)')
        parser.add_argument('--dataset_path', default='justinphan3110/harmful_harmless_instructions', type=str, help='Path for training_args.output_dir')
        parser.add_argument('--dataset_names', default=None, type=str, help='Name of the dataset configuration for training_args.output_dir')
<<<<<<< Updated upstream
        parser.add_argument('--start_coeff', default=-4.0, type=float, help='coeff to start the range of the norm injection of the representation vector')
        parser.add_argument('--end_coeff', default=4.0, type=float, help='coeff to end the range of the norm injection of the representation vector')
=======
        parser.add_argument('--start_coeff', default=-1.0, type=float, help='coeff to start the range of the norm injection of the representation vector')
        parser.add_argument('--end_coeff', default=2.0, type=float, help='coeff to end the range of the norm injection of the representation vector')
>>>>>>> Stashed changes
        parser.add_argument('--coeff_step', default=0.5, type=float, help='step for the range of the norm injection of the representation vector')
        parser.add_argument('--num_instructions', default=16, type=int, help='number of instructions to generate for each prompt')
        parser.add_argument('--num_samples', default=1, type=int, help='number of samples to generate for each instruction')
        parser.add_argument('--is_synth_reading_vectors', default=False, type=bool, help='Whether to generate reading vectors synthetically (produced by the model) or load them from the dataset for REPE')
        parser.add_argument('--output_dir', default="data/harmfulness_experiments_outputs/default_dir_safety", type=str, help='Path for the output directory')
        
        args = parser.parse_args()
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
        self.template_system_and_user = prompt_template_system_and_user(self.model_name)
        self.template_user = prompt_template_user(self.model_name)
    
    def __str__(self):
        return (f"Model Name or Path: {self.model_name}\n"
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
                f"Template User: {self.template_user}\n")
