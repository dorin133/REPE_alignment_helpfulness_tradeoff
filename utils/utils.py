"""
Reading Vectors Generation Module

This module provides classes for generating and loading reading vectors used in 
Representation Engineering (REPE) experiments. It supports both harmfulness and 
fairness datasets, with options for synthetic vector generation or dataset-derived vectors.
"""

import os
import json
import torch
from tqdm import tqdm
from repe.rep_control_reading_vec import WrappedReadingVecModel
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from typing import List, Dict, Any, Tuple


class ReadingVectors_Harmfulness:
    """
    Class for loading and processing harmfulness reading vectors from existing datasets.
    
    This class handles the preparation of reading vectors for harmfulness experiments
    using the harmful/harmless instructions dataset for REPE interventions.
    """
    
    def __init__(self, args):
        """
        Initialize the harmfulness reading vectors handler.
        
        Args:
            args: Configuration arguments containing experiment parameters
        """
        self.args = args
        self.reading_vec_dataset = "justinphan3110/harmful_harmless_instructions"
        
    def load_reading_vec_dataset(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Load and return the reading vector dataset for harmfulness experiments.
        
        Returns:
            Tuple[List[str], List[int], List[str]]: (train_data, train_labels, test_data)
        """
        train_data, train_labels, test_data = self._reading_vec_dataset_chat_model()
        return train_data, train_labels, test_data
        
    def _reading_vec_dataset_chat_model(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Process the harmfulness dataset for chat models.
        
        Returns:
            Tuple[List[str], List[int], List[str]]: Processed training and test data
        """
        # Load the harmful/harmless instructions dataset
        dataset = load_dataset(self.reading_vec_dataset, cache_dir=None)
        
        # Split into train and test sets
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        
        # Process training data
        train_data = []
        train_labels = []
        
        for item in train_dataset:
            # Format as chat conversation
            formatted_text = f"Request: {item['instruction']} Answer: {item['output']}"
            train_data.append(formatted_text)
            # Label: 1 for harmless, 0 for harmful
            train_labels.append(1 if item['label'] == 'harmless' else 0)
        
        # Process test data
        test_data = []
        for item in test_dataset:
            test_data.append(item['instruction'])
        
        return train_data, train_labels, test_data


class Synthetic_ReadingVectors_Harmfulness:
    """
    Class for generating synthetic harmfulness reading vectors using a language model.
    
    This class creates synthetic reading vectors by generating responses to harmful
    prompts using a specified model, then uses these for REPE interventions.
    """
    
    def __init__(self, args, reading_vec_dataset_save_path=None, 
                 model_name_or_path_for_generation="meta-llama/Llama-2-13b-chat-hf"):
        """
        Initialize the synthetic harmfulness reading vectors generator.
        
        Args:
            args: Configuration arguments containing experiment parameters
            reading_vec_dataset_save_path: Path to save/load generated dataset
            model_name_or_path_for_generation: Model to use for synthetic generation
        """
        self.args = args
        self.reading_vec_dataset_save_path = reading_vec_dataset_save_path
        self.model_name_or_path_for_generation = model_name_or_path_for_generation
        
    def load_reading_vec_dataset(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Load or generate synthetic reading vector dataset.
        
        Returns:
            Tuple[List[str], List[int], List[str]]: (train_data, train_labels, test_data)
        """
        # Check if pre-generated dataset exists
        if (self.reading_vec_dataset_save_path and 
            os.path.exists(self.reading_vec_dataset_save_path)):
            
            print(f"Loading existing synthetic dataset from: {self.reading_vec_dataset_save_path}")
            with open(self.reading_vec_dataset_save_path, 'r') as f:
                data = json.load(f)
            return data['train_data'], data['train_labels'], data['test_data']
        else:
            print("Generating new synthetic reading vectors...")
            return self._reading_vec_dataset_raw_model()

    def _reading_vec_dataset_raw_model(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Generate synthetic reading vectors using a raw language model.
        
        Returns:
            Tuple[List[str], List[int], List[str]]: Generated training and test data
        """
        # Load base harmful/harmless dataset for prompts
        base_dataset = load_dataset("justinphan3110/harmful_harmless_instructions")
        
        # Load generation model
        print(f"Loading model for generation: {self.model_name_or_path_for_generation}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path_for_generation,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path_for_generation)
        
        train_data = []
        train_labels = []
        test_data = []
        
        print("Generating synthetic responses...")
        
        # Generate responses for training data
        for item in tqdm(base_dataset['train'], desc="Generating training data"):
            prompt = item['instruction']
            
            # Generate model response
            inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.to(model.device),
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from response
            response = response[len(prompt):].strip()
            
            # Format as Q&A pair
            formatted_text = f"QUESTION: {prompt}\n\nANSWER: {response}"
            train_data.append(formatted_text)
            train_labels.append(1 if item['label'] == 'harmless' else 0)
        
        # Extract test prompts
        for item in base_dataset['test']:
            test_data.append(item['instruction'])
        
        # Save generated dataset if path provided
        if self.reading_vec_dataset_save_path:
            os.makedirs(os.path.dirname(self.reading_vec_dataset_save_path), exist_ok=True)
            data = {
                'train_data': train_data,
                'train_labels': train_labels,
                'test_data': test_data
            }
            with open(self.reading_vec_dataset_save_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved synthetic dataset to: {self.reading_vec_dataset_save_path}")
        
        return train_data, train_labels, test_data
    

class ReadingVectors_Fairness:
    """
    Class for loading and processing fairness reading vectors from bias datasets.
    
    This class handles the preparation of reading vectors for fairness experiments
    using stereotypical and anti-stereotypical examples for REPE interventions.
    """
    
    def __init__(self, args):
        """
        Initialize the fairness reading vectors handler.
        
        Args:
            args: Configuration arguments containing experiment parameters
        """
        self.args = args
        self.reading_vec_dataset = "/path/to/data/stereoset_dataset"
        
    def load_reading_vec_dataset(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Load and return the reading vector dataset for fairness experiments.
        
        Returns:
            Tuple[List[str], List[int], List[str]]: (train_data, train_labels, test_data)
        """
        train_data, train_labels, test_data = self._reading_vec_dataset_chat_model()
        return train_data, train_labels, test_data

    def _reading_vec_dataset_chat_model(self, ntrain=150, bias='race') -> Tuple[List[str], List[int], List[str]]:
        """
        Process the fairness dataset for chat models with specific bias type.
        
        Args:
            ntrain: Number of training examples to use
            bias: Type of bias to focus on ('race', 'gender', etc.)
            
        Returns:
            Tuple[List[str], List[int], List[str]]: Processed fairness data
        """
        # Load StereoSet dataset
        try:
            dataset = load_from_disk(self.reading_vec_dataset)
        except:
            # Fallback to loading from HuggingFace if local dataset not found
            dataset = load_dataset("stereoset", "intrasentence")
            
        train_data = []
        train_labels = []
        test_data = []
        
        stereotypical_examples = []
        anti_stereotypical_examples = []
        
        print(f"Processing StereoSet data for bias type: {bias}")
        
        # Process the dataset
        for item in dataset['validation']:  # StereoSet uses validation split
            if item['bias_type'] == bias:
                context = item['context']
                
                for sentence in item['sentences']:
                    if sentence['gold_label'] == 'stereotype':
                        stereotypical_examples.append({
                            'text': context + " " + sentence['sentence'],
                            'label': 0  # Stereotypical = 0
                        })
                    elif sentence['gold_label'] == 'anti-stereotype':
                        anti_stereotypical_examples.append({
                            'text': context + " " + sentence['sentence'],
                            'label': 1  # Anti-stereotypical = 1
                        })
        
        # Balance the dataset
        min_examples = min(len(stereotypical_examples), len(anti_stereotypical_examples))
        balanced_examples = (stereotypical_examples[:min_examples] + 
                           anti_stereotypical_examples[:min_examples])
        
        # Split into train and test
        train_examples = balanced_examples[:ntrain]
        test_examples = balanced_examples[ntrain:ntrain+100]  # Use 100 for testing
        
        # Format training data
        for example in train_examples:
            # Format as conversational response
            formatted_text = f"Statement: {example['text']} Response: I understand this perspective."
            train_data.append(formatted_text)
            train_labels.append(example['label'])
        
        # Format test data
        for example in test_examples:
            test_data.append(example['text'])
        
        print(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples")
        return train_data, train_labels, test_data
    

class Synthetic_ReadingVectors_Fairness:
    """
    Class for generating synthetic fairness reading vectors using a language model.
    
    This class creates synthetic reading vectors by generating responses to biased
    prompts using a specified model, then uses these for fairness REPE interventions.
    """
    
    def __init__(self, args, reading_vec_dataset_save_path=None, 
                 model_name_or_path_for_generation="meta-llama/Llama-2-13b-chat-hf"):
        """
        Initialize the synthetic fairness reading vectors generator.
        
        Args:
            args: Configuration arguments containing experiment parameters
            reading_vec_dataset_save_path: Path to save/load generated dataset
            model_name_or_path_for_generation: Model to use for synthetic generation
        """
        self.args = args
        self.reading_vec_dataset_save_path = reading_vec_dataset_save_path
        self.model_name_or_path_for_generation = model_name_or_path_for_generation

    def load_reading_vec_dataset(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Load or generate synthetic fairness reading vector dataset.
        
        Returns:
            Tuple[List[str], List[int], List[str]]: (train_data, train_labels, test_data)
        """
        # Check if pre-generated dataset exists
        if (self.reading_vec_dataset_save_path and 
            os.path.exists(self.reading_vec_dataset_save_path)):
            
            print(f"Loading existing synthetic fairness dataset from: {self.reading_vec_dataset_save_path}")
            with open(self.reading_vec_dataset_save_path, 'r') as f:
                data = json.load(f)
            return data['train_data'], data['train_labels'], data['test_data']
        else:
            print("Generating new synthetic fairness reading vectors...")
            return self._reading_vec_dataset_raw_model()

    def _reading_vec_dataset_raw_model(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Generate synthetic fairness reading vectors using a raw language model.
        
        Returns:
            Tuple[List[str], List[int], List[str]]: Generated fairness training and test data
        """
        # Load base StereoSet dataset for prompts
        try:
            base_dataset = load_dataset("stereoset", "intrasentence")
        except:
            print("Warning: Could not load StereoSet dataset, using fallback prompts")
            return self._generate_fallback_fairness_data()
        
        # Load generation model
        print(f"Loading model for fairness generation: {self.model_name_or_path_for_generation}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path_for_generation,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path_for_generation)
        
        train_data = []
        train_labels = []
        test_data = []
        
        print("Generating synthetic fairness responses...")
        
        # Process validation split for fairness examples
        processed_count = 0
        for item in tqdm(base_dataset['validation'], desc="Processing fairness data"):
            if processed_count >= 300:  # Limit to avoid excessive generation
                break
                
            if item['bias_type'] in ['race', 'gender']:  # Focus on specific bias types
                context = item['context']
                
                for sentence in item['sentences']:
                    if sentence['gold_label'] in ['stereotype', 'anti-stereotype']:
                        prompt = context + " " + sentence['sentence']
                        
                        # Generate model response
                        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
                        with torch.no_grad():
                            outputs = model.generate(
                                inputs.to(model.device),
                                max_new_tokens=50,
                                do_sample=True,
                                temperature=0.7,
                                pad_token_id=tokenizer.eos_token_id
                            )
                        
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        response = response[len(prompt):].strip()
                        
                        # Format as Q&A pair
                        formatted_text = f"QUESTION: Complete this statement: {prompt}\n\nANSWER: {response}"
                        train_data.append(formatted_text)
                        # Label: 1 for anti-stereotype, 0 for stereotype
                        train_labels.append(1 if sentence['gold_label'] == 'anti-stereotype' else 0)
                        
                        processed_count += 1
                        
                        if processed_count >= 300:
                            break
        
        # Generate test data (use first 100 contexts as test prompts)
        for item in base_dataset['validation'][:100]:
            if item['bias_type'] in ['race', 'gender']:
                test_data.append(item['context'])
        
        # Save generated dataset if path provided
        if self.reading_vec_dataset_save_path:
            os.makedirs(os.path.dirname(self.reading_vec_dataset_save_path), exist_ok=True)
            data = {
                'train_data': train_data,
                'train_labels': train_labels,
                'test_data': test_data
            }
            with open(self.reading_vec_dataset_save_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved synthetic fairness dataset to: {self.reading_vec_dataset_save_path}")
        
        return train_data, train_labels, test_data
    
    def _generate_fallback_fairness_data(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Generate fallback fairness data when StereoSet is unavailable.
        
        Returns:
            Tuple[List[str], List[int], List[str]]: Fallback fairness data
        """
        print("Using fallback fairness prompts...")
        
        # Predefined examples for fairness testing
        stereotypical_prompts = [
            "People from certain backgrounds are naturally better at math",
            "Women are typically more emotional than men",
            "Certain ethnic groups are more likely to be criminals",
        ]
        
        anti_stereotypical_prompts = [
            "Mathematical ability depends on education and practice, not background",
            "Emotional expression varies by individual, not gender",
            "Criminal behavior is influenced by socioeconomic factors, not ethnicity",
        ]
        
        train_data = []
        train_labels = []
        test_data = []
        
        # Create training examples
        for prompt in stereotypical_prompts:
            formatted_text = f"QUESTION: {prompt}\n\nANSWER: This is a common perspective."
            train_data.append(formatted_text)
            train_labels.append(0)  # Stereotypical
        
        for prompt in anti_stereotypical_prompts:
            formatted_text = f"QUESTION: {prompt}\n\nANSWER: This is a fair perspective."
            train_data.append(formatted_text)
            train_labels.append(1)  # Anti-stereotypical
        
        # Create test data
        test_data = stereotypical_prompts + anti_stereotypical_prompts
        
        return train_data, train_labels, test_data