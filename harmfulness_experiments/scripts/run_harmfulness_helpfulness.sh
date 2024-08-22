#!/bin/bash

# # SBATCH -p g24
# # SBATCH --job-name=REPE
# # SBATCH --qos=exception
# # SBATCH --nodes=1                 # Number of nodes
# # SBATCH --ntasks=1         # Number of tasks (one for each script)
# # SBATCH --cpus-per-task=60
# # SBATCH --gres=gpu:4
# # SBATCH --array=1-1                      # Array range
# # SBATCH --output=/dev/null     # Discard standard output  # Because we write to the log.txt file
# # #SBATCH --exclusive
# # SBATCH --constraint=ampere

export HF_DATASETS_CACHE='/export/work/dshteyma/.cache/huggingface/datasets'
export HF_HOME='/export/work/dshteyma/.cache/huggingface'

declare -A DATASET_MMLU_NAMES=(
    ["1"]="high_school_computer_science"
    ["2"]="clinical_knowledge"
    ["3"]="medical_genetics"
    ["4"]="international_law"
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

current_time=$(date +"%d-%m_%H-%M")

OUTPUT_DIR="/home/dshteyma/REPE_alignment_helpfulness_tradeoff/data/harmfulness_experiments_outputs/${current_time}"
mkdir_is_exists $OUTPUT_DIR

MODEL_NAME="--model_name Llama-2-13b-chat"
START_COEFF="--start_coeff -5.0"
END_COEFF="--end_coeff 5.2"
NUM_INSTRUCTIONS="--num_instructions 64"
NUM_SAMPLES="--num_samples 1"
OUTPUT_DIR_ARG="--output_dir $OUTPUT_DIR"
# choices = ['high_school_computer_science', 'medical_genetics', 'international_law', 'clinical_knowledge']
DATASET_MMLU_NAMES_ARG="--dataset_mmlu_names high_school_computer_science,medical_genetics,international_law,clinical_knowledge"

CMD="python -m harmfulness_experiments.harmfulness_helpfulness \
                                                    $MODEL_NAME \
                                                    $DATASET_MMLU_NAMES_ARG \
                                                    $OUTPUT_DIR_ARG \
                                                    $START_COEFF \
                                                    $END_COEFF \
                                                    $NUM_INSTRUCTIONS \
                                                    $NUM_SAMPLES"

echo $CMD > $OUTPUT_DIR/metadata.txt
$CMD 2>&1 | tee $OUTPUT_DIR/log.txt

# $CMD
# $CMD 2>&1 | tee $OUTPUT_DIR/log.txt
