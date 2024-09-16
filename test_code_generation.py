from datasets import load_dataset
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from code_runner import test_humaneval_function
import pdb


class DummyOutput:
    def write(self, text):
        pass


def parser_code(model_response):
    split_model_output = model_response.split("```")
    if len(split_model_output) == 1:
        return False
    fixed_code = split_model_output[1].replace('python', '').replace('Python', '')
    return fixed_code


def get_import_lines(code: str) -> str:
    # Split the code into lines
    lines = code.split('\n')
    # Filter lines that contain import statements
    import_lines = [line for line in lines if line.strip().startswith('import') or line.strip().startswith('from')]
    # Join the import lines back into a single string
    return '\n'.join(import_lines)


def find_first_non_indent_line(text, start_index):
    lines = text.splitlines()
    for i in range(start_index, len(lines)):
        # Check if the line does not start with a space or a tab
        if lines[i] and not lines[i].startswith((' ', '\t')):
            return i
    return -1  # Return -1 if no such line is found


def find_first_def_line(text, start_index):
    lines = text.splitlines()
    for i in range(start_index, len(lines)):
        if 'def ' in lines[i]:
            return i
    return -1


def extract_line_range(text, start_line, end_line):
    lines = text.splitlines()
    return '\n'.join(lines[start_line:end_line])


def extract_function(mixed_string, function_name, starting_index=0):
    if starting_index is None:
        return None, None
    def_idx_fk = find_first_def_line(mixed_string, starting_index)
    if def_idx_fk == -1:
        return None, None
    end_idx_fk = find_first_non_indent_line(mixed_string, def_idx_fk+1)
    if end_idx_fk == -1:
        return None, None
    return extract_line_range(mixed_string, def_idx_fk, end_idx_fk), end_idx_fk


def extract_function_2(mixed_string, function_name):
    pattern = rf"```\s*\n*def\s+{re.escape(function_name)}\s*\([^)]*\):(?:(?!\n```)[\s\S])*\n```"
    match = re.search(pattern, mixed_string, re.MULTILINE)
    if match:
        function_code = match.group(0)
        function_code = re.sub(r'^```\s*\n*', '', function_code)
        function_code = re.sub(r'\n*```\s*$', '', function_code)
        return function_code.strip()
    else:
        return None


def test_human_eval_dataset(all_generations, data_dict):
    results = dict()
    for coeff in all_generations:
        curr_generations = all_generations[coeff]
        success_perc_list = []
        for key in curr_generations:
            print(key)
            curr_generation_batch = curr_generations[key]
            curr_problem = data_dict[key]

            full_success_list = []
            for i in range(len(curr_generation_batch)):
                entry_point = curr_problem['entry_point']
                curr_answer = curr_generation_batch[i]
                ADD_PROMPT = f'def {entry_point}' not in curr_answer
                if ADD_PROMPT:
                    curr_answer = f"{curr_problem['prompt']}\n{curr_answer}"
                curr_imports = get_import_lines(curr_answer)
                # try a few function extraction methods
                function_codes_to_try = []
                option1, index = extract_function(curr_answer + "\naaa", curr_problem['entry_point'])
                # option2, _ = extract_function(curr_answer + "\naaa", curr_problem['entry_point'], index)
                # option3 = extract_function_2(curr_answer, curr_problem['entry_point'])
                function_codes_to_try.append(option1)
                # function_codes_to_try.append(option2)
                # function_codes_to_try.append(option3)
                for curr_function_code in function_codes_to_try:
                    if curr_function_code is None:
                        is_pass = False
                        continue
                    curr_code_with_imports = f"{curr_imports}\n{curr_function_code}"
                    is_pass = test_humaneval_function(curr_problem, curr_function_code)
                    # pdb.set_trace()
                    if is_pass:
                        break
                full_success_list.append(is_pass)

            success_perc = np.sum(full_success_list) / len(curr_generation_batch)
            success_perc_list.append(success_perc)

        results[coeff] = np.average(success_perc_list), np.std(success_perc_list) / len(success_perc_list)**0.5

    return results


def plot_results(keys, averages, stds):
    averages = np.array(averages)
    stds = np.array(stds)
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(keys, averages)
    plt.fill_between(keys, averages - stds, averages + stds, alpha=0.2)
    # plt.errorbar(keys, averages, yerr=stds, fmt='o', capsize=5, capthick=2, ecolor='red',
    #                               markeredgecolor='black', markerfacecolor='blue')

    # Customize the plot
    plt.xlabel('coefficients')
    plt.ylabel('mean success rate')
    plt.title('Human eval success rate vs. REPE coefficients')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_fine_tuned_models_results(results_dict):
    results = [results_dict["/cs/labs/shashua/binyamin/models/Meta-Llama-3.1-8B"]]
    results_dict.pop("/cs/labs/shashua/binyamin/models/Meta-Llama-3.1-8B")
    checkpoints = list(results_dict.keys())
    checkpoints_numbers = sorted([int(checkpoint.split('-')[-1]) for checkpoint in checkpoints])
    print(checkpoints_numbers)
    for i in checkpoints_numbers:
        curr_results = results_dict[f"checkpoint-{i}"]
        results.append(curr_results)

    avgs = [res[0] for res in results]
    stds = [res[1] for res in results]
    plot_results(np.arange(len(results)), avgs, stds)


def main():
    human_eval_data = load_dataset("openai/openai_humaneval")
    human_eval_dict = {q['task_id']: q for q in human_eval_data['test']}
    all_gen_dict = {}
    gens_paths = ['fine-tuned_model_generations_15_09_2024_1_epoch.json']
    for path in gens_paths:
        curr_gen = open(path)
        curr_gen = json.load(curr_gen)
        for key in curr_gen:
            if key not in all_gen_dict:
                all_gen_dict[key] = curr_gen[key]
            else:
                for q_key in curr_gen[key]:
                    if q_key not in all_gen_dict[key]:
                        all_gen_dict[key][q_key] = curr_gen[key][q_key]
                    else:
                        all_gen_dict[key][q_key] += curr_gen[key][q_key]

    # this is the range we are interested in (it's all 0 out of it)
    # all_gen_dict = {key: val for key, val in all_gen_dict.items() if abs(float(key)) <= 3}
    results = test_human_eval_dataset(all_gen_dict, human_eval_dict)
    plot_fine_tuned_models_results(results)
    # print(results)
    # results = dict(sorted(results.items()))
    # sorted_keys = list(results.keys())
    # avgs = [key[0] for key in results.values()]
    # stds = [key[1] for key in results.values()]
    # plot_results(sorted_keys, avgs, stds)


if __name__ == '__main__':
    main()
