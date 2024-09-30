#!/bin/bash

# SBATCH -p g48
# SBATCH --job-name=REPE
# SBATCH --qos=exception
# SBATCH --nodes=1                 # Number of nodes
# SBATCH --ntasks=1         # Number of tasks (one for each script)
# SBATCH --cpus-per-task=30
# SBATCH --gres=gpu:2
# SBATCH --array=1-1                      # Array range
# SBATCH --output=/dev/null     # Discard standard output  # Because we write to the log.txt file
# #SBATCH --exclusive
# SBATCH --constraint=ampere

export HF_DATASETS_CACHE='/export/work/dshteyma/.cache/huggingface/datasets'
export HF_HOME='/export/work/dshteyma/.cache/huggingface'

declare -A DATASET_MMLU_NAMES=(
    ["1"]="high_school_computer_science"
    ["2"]="clinical_knowledge"
    ["3"]="medical_genetics"
    ["4"]="international_law"
)

declare -A MODEL_NAMES=(
    ["1"]="meta-llama/Meta-Llama-3.1-8B"
    ["2"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
    ["3"]="meta-llama/Llama-2-13b-hf"
    ["4"]="meta-llama/Llama-2-13b-chat-hf"
)

source /home/dshteyma/miniconda3/envs/phi3_env/bin/activate

mkdir_is_exists() {
    if [ -d "$1" ]; then
        echo "Directory '$1' already exists."
    else
        mkdir -p "$1"
        echo "Directory '$1' created."
    fi
}

# OUTPUT_DIR="/home/dshteyma/REPE_alignment_helpfulness_tradeoff/data/fairness_experiments_outputs/24-09_09-45"

current_time=$(date +"%d-%m_%H-%M-%S")
OUTPUT_DIR="/home/dshteyma/REPE_alignment_helpfulness_tradeoff/data/fairness_experiments_outputs/${current_time}"
mkdir_is_exists $OUTPUT_DIR

MODEL_NAME="--model_name meta-llama/Llama-2-13b-chat-hf"
DATASET_PATH="--dataset_path lukaemon/mmlu"
DATASET_NAMES="--dataset_names international_law,medical_genetics,high_school_computer_science"
START_COEFF="--start_coeff -10.0"
END_COEFF="--end_coeff 10.2"
STEP_COEFF="--coeff_step 0.25"
# START_COEFF="--start_coeff -5.0"
# END_COEFF="--end_coeff 5.1"
# STEP_COEFF="--coeff_step 0.2"
# IS_SYNTH_READING_VECS="--is_synth_reading_vectors"
NUM_INSTRUCTIONS="--num_instructions 96"
NUM_SAMPLES="--num_samples 1"
OUTPUT_DIR_ARG="--output_dir $OUTPUT_DIR"

CMD="python experiments/fairness_experiments/fairness_helpfulness_new.py \
                                                    $MODEL_NAME \
                                                    $DATASET_PATH \
                                                    $DATASET_NAMES \
                                                    $START_COEFF \
                                                    $END_COEFF \
                                                    $STEP_COEFF \
                                                    $NUM_INSTRUCTIONS \
                                                    $NUM_SAMPLES \
                                                    $IS_SYNTH_READING_VECS \
                                                    $OUTPUT_DIR_ARG"

echo $CMD > $OUTPUT_DIR/metadata.txt
$CMD 2>&1 | tee $OUTPUT_DIR/log.txt


