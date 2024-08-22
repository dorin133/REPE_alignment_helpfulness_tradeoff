import csv
import tqdm
import torch
import json
import os
import pandas as pd
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from repe.rep_control_reading_vec import WrappedReadingVecModel
from harmfulness_experiments.harmfulness_utils import GenerationArgsHelpfulness
from harmfulness_experiments.harmfulness_utils import reading_vec_dataset_Q_and_A
from harmfulness_experiments.harmfulness_utils import reading_vec_dataset_by_github, feed_dialog_helpfulness, get_answer_probs
from datasets import load_dataset
import torch.nn.functional as F
import math
import argparse
import re

answers_path='/home/dshteyma/REPE_alignment_helpfulness_tradeoff/data/harmfulness_experiments_outputs/22-08_04-12/high_school_computer_science_Llama-2-13b-chat_answers.json'

with open(answers_path, 'r') as file:
    all_answers_dict = json.load(file)
    
for value in all_answers_dict.values():
    print(value)

