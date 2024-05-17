import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import Counter
from transformers import AutoModelForCausalLM
from scipy.optimize import minimize

# Auxilary functions:
def indices_of_k_most_probable_tokens(logits, k):
    # Apply softmax to get probabilities
    probabilities = F.softmax(logits, dim=-1)

    # Get the top k probabilities and corresponding indices
    top_k_values, top_k_indices = torch.topk(probabilities, k)

    # return probabilities , top_k_indices.tolist()
    return top_k_indices.tolist()

def k_most_common_indices(tensors_list, k):
    # Count occurrences of each integer
    counts = Counter(tensors_list)
    # Sort integers based on counts and select the top k
    most_common_idx_and_count = counts.most_common(k)
    most_common_idx = [x for (x,y) in most_common_idx_and_count]
    return most_common_idx

def keys_for_values(vocabulary, values):
    return [key for key, value in vocabulary.items() if value in values]

def values_for_keys(dictionary, keys):
    return [dictionary[key] for key in keys if key in dictionary]

############  initialize parameters  ############
model_name = "Llama-2-13b"
if model_name == "Llama-2-13b":
    # raw model
    save_data_dir = 'separation_by_dr_raw'
else:
    # chat model
    save_data_dir = 'separation_by_dr'
    
file_path = f'../../lab_data/{save_data_dir}/lamma2_vocab.json'
with open(file_path, 'r') as file:
  vocabulary = json.load(file)

amount_of_tokens = 40

x = [round(i, 1) for i in np.arange(-10, 10.5, 0.2)]
for i, coeff in enumerate(x):
  print(coeff)
  print(i)
  print('--------------')
##########################################################

gt_k_most_common_indices = {}
gt_most_probable_tokens_strings = {}
gt_most_probable_tokens_indices = {}
gt_norms = []

for idx in range(10):
  top_k_indices = []
  file_path = f'../../lab_data/{save_data_dir}/logits_iter_gt_inst_{idx}.json'
  with open(file_path, 'r') as file:
    gt_curr_logits = torch.tensor(json.load(file))
    gt_norms.append(torch.norm(gt_curr_logits).item())

  # ----- find the indices in the llama2 vocabulary of the most probable tokens of coeff=0 ----- #
  gt_k_most_common_indices[idx] = indices_of_k_most_probable_tokens(gt_curr_logits, amount_of_tokens)
  gt_most_probable_tokens_strings[idx] = keys_for_values(vocabulary, gt_k_most_common_indices[idx])
  gt_most_probable_tokens_strings[idx] = list(filter(lambda x: any(c.isalpha() for c in x) and not '<' in x and not '>' in x, gt_most_probable_tokens_strings[idx]))
  gt_most_probable_tokens_indices[idx] = values_for_keys(vocabulary, gt_most_probable_tokens_strings[idx])

  print(gt_k_most_common_indices[idx])
  print(gt_most_probable_tokens_indices[idx])
  print(f'idx gt = {idx}: {gt_most_probable_tokens_strings[idx]}')
  print("--------------")

token_indices_harmful = {}
most_probable_harmful_tokens_strings = {}
most_probable_harmful_tokens_indices = {}

for idx in range(10):
  all_top_k_indices = []
  # ----- find the indices in the llama2 vocabulary of the most probable tokens of harmful behavior responses ----- #
  for i in range(66, 70):
    file_path = f"../../lab_data/{save_data_dir}/logits_iter_{i}_inst_{idx}.json"
    with open(file_path, 'r') as file:
      tensor_logits = torch.tensor(json.load(file))
      all_top_k_indices.extend(indices_of_k_most_probable_tokens(tensor_logits, amount_of_tokens))
  token_indices_harmful[idx] = k_most_common_indices(all_top_k_indices, amount_of_tokens)
  most_probable_harmful_tokens_strings[idx] = keys_for_values(vocabulary, token_indices_harmful[idx])
  most_probable_harmful_tokens_strings[idx] = list(filter(lambda x: any(c.isalpha() for c in x) and not '<' in x and not '>' in x, most_probable_harmful_tokens_strings[idx]))
  most_probable_harmful_tokens_indices[idx] = values_for_keys(vocabulary, most_probable_harmful_tokens_strings[idx])
  print(f'idx harmful = {idx}: {most_probable_harmful_tokens_strings[idx]}')
  print(f'idx = {idx}: {most_probable_harmful_tokens_indices[idx]}')
  print("--------------")

# ----- find the logits values with coeff=0 of the tokens associated with harmfulness or benign behaviors
gt_logits_harmful_tokens = {}
gt_logits_top_gt_tokens = {}
for idx in range(10):
  file_path = f"../../lab_data/{save_data_dir}/logits_iter_gt_inst_{idx}.json"
  with open(file_path, 'r') as file:
    gt_curr_logits = torch.tensor(json.load(file))
  gt_logits_harmful_tokens[idx] = [gt_curr_logits[curr_idx] for curr_idx in most_probable_harmful_tokens_indices[idx]]
  gt_logits_top_gt_tokens[idx] = [gt_curr_logits[curr_idx] for curr_idx in gt_most_probable_tokens_indices[idx]]


# x = [i for i in np.arange(-10, 10.5, 0.5)]
x = [round(i, 1) for i in np.arange(-10, 10.2, 0.2)]
logits_all_inst = {'projection_over_harmful_tokens': {key: {} for key in range(10)}, 'projection_over_gt_tokens': {key: {} for key in range(10)}}
all_x_plots = {'projection_over_harmful_tokens': [np.zeros(len(x)) for _ in range(10)], 'projection_over_gt_tokens': [np.zeros(len(x)) for _ in range(10)]}
coeff_norms = {'projection_over_harmful_tokens': [np.zeros(len(x)) for _ in range(10)], 'projection_over_gt_tokens': [np.zeros(len(x)) for _ in range(10)]}
gt_logits = {'projection_over_harmful_tokens': gt_logits_harmful_tokens, 'projection_over_gt_tokens': gt_logits_top_gt_tokens}
token_indices = {'projection_over_harmful_tokens': most_probable_harmful_tokens_indices, 'projection_over_gt_tokens': gt_most_probable_tokens_indices}
y_plot = x

for plot_key in all_x_plots.keys():
  for idx in range(10):
    for i, coeff in enumerate(x):
      file_path = f"../../lab_data/{save_data_dir}/logits_iter_{i}_inst_{idx}.json"
      with open(file_path, 'r') as file:
        logits = json.load(file)
        curr_logits = torch.tensor(logits)
        coeff_norms[plot_key][idx][i] = torch.norm(curr_logits).item()
        # ----- find the logits values with the current coeff of the tokens associated with harmfulness or benign behaviors
        curr_logits_relevant_tokens = [curr_logits[curr_idx] for curr_idx in token_indices[plot_key][idx]]
        dr = torch.sum(torch.tensor(curr_logits_relevant_tokens)) - torch.sum(torch.tensor(gt_logits[plot_key][idx]))
        dr_normalized = dr/amount_of_tokens
        all_x_plots[plot_key][idx][i] = dr_normalized.item()

# ----- plot the average norm values of the logits of tokens associated with:
#       1. harmful behavior (blue)
#       2. no representation engineering behavior (red)
x = [round(i, 1) for i in np.arange(-10, 10.2, 0.2)]
num_plots = 10
fig, axs = plt.subplots(num_plots, 1, figsize=(28, 6 * num_plots))
for k in range(num_plots):
  axs[k].scatter(x, all_x_plots['projection_over_harmful_tokens'][k] , color='blue', label='harmful response tokens')
  axs[k].scatter(x, all_x_plots['projection_over_gt_tokens'][k] , color='red', label='aligned response tokens')
  # Marking the zero coordinate
  axs[k].axvline(x=0, linestyle='--', color='black')
  # Customizing the plot
  axs[k].set_xlabel('$r_e$', fontsize=12)
  axs[k].set_ylabel(r'$\langle$$\delta{r_e(q)}$ , $U^T(e_{i})$$\rangle$', fontsize=12)
  # axs[k].set_title('Projection on logits of coeff = 0', fontsize=16)
  axs[k].legend(fontsize = 14)

fig.savefig('scatter_plots_separation_2_instructions_delta_2d_format.png')
plt.show()


# ----- plot the average norm values of the logits of tokens 
#       associated with harmful behavior subtracted from tokens associated with no representation engineering behavior
#       also plot a fitted linear line to compute the lower bound of Assumption 2 in the article
x = [round(i, 1) for i in np.arange(0, 10.2, 0.2)]
num_plots = 10
fig, axs = plt.subplots(num_plots, 1, figsize=(28, 6 * num_plots))
for idx in range(num_plots):
  k = idx
  axs[idx].scatter(x, (all_x_plots['projection_over_harmful_tokens'][k][50:50+len(x)] - all_x_plots['projection_over_gt_tokens'][k][50:50+len(x)]) , color='purple')
  all_y_coor_of_x = all_x_plots['projection_over_harmful_tokens'][k][50:50+len(x)] - all_x_plots['projection_over_gt_tokens'][k][50:50+len(x)]
  x_linear1 = torch.tensor(x[:4])
  x_linear2 = torch.tensor([x[0],x[-1]])
  y_fit1 = torch.tensor(all_y_coor_of_x[:4])
  y_fit2 = torch.tensor([all_y_coor_of_x[0], all_y_coor_of_x[-1]])
  m1, b1 = np.polyfit(x_linear1, y_fit1, deg=1)
  m2, b2 = np.polyfit(x_linear2, y_fit2, deg=1)
  m = min(m1, m2)
  y_linear = torch.tensor(x) * m

  axs[idx].plot(x, y_linear , color='red', label=r'|$\delta{r_e(q)}$| $\cdot$ $\Delta$')
  # Customizing the plot
  axs[idx].set_xlabel('$r_e$', fontsize=12)
  axs[idx].set_ylabel(r'$\langle$$\delta{r_e(q)}$ , $U^T(e_{good}-e_{bad})$$\rangle$', fontsize=12)
  axs[idx].legend(fontsize = 14)

fig.savefig('plots_separation_lower_bound.png')
plt.show()
