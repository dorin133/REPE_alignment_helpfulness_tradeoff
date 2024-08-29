import torch
import harmfulness_experiments.harmfulness_utils as harmfulness_utils
import os
from utils import sample_model, read_json_if_exists, clear_memory, load_model, set_seed

question_template_llama_3_1 = \
"""<|begin_of_text|>{question}"""

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, _, _, raw_test_data = harmfulness_utils.reading_vec_dataset_by_github()
harmful_test_data = raw_test_data[1::2][:100]

model_dir = '/cs/labs/shashua/binyamin/REPE_alignment_helpfulness_tradeoff/lora_finetuned_model_22_08_ver3/'
model_subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

output_path = 'fine-tuned_model_alignment_test.json'
results_dict = dict()
for model_subdir in model_subdirs:
    model_path = os.path.join(model_dir, model_subdir)
    print(f"Testing model in: {model_path}")
    model, tokenizer = load_model(model_path)

    results_dict[model_subdir] = harmfulness_utils.feed_dialog_behavior(model=model,
                                                                       model_name=model_subdir,
                                                                       tokenizer=tokenizer,
                                                                       dataset=harmful_test_data,
                                                                       coeff=0,
                                                                       num_samples=4,
                                                                       num_instructions=len(harmful_test_data),
                                                                       question_template=question_template_llama_3_1)

    # Clear memory
    del model
    del tokenizer
    clear_memory()
    print("Memory cleared")
    print()
