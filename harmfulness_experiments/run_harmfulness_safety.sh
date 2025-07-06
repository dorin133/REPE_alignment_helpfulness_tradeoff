#!/bin/bash

#SBATCH -p g48
#SBATCH --job-name=REPE_harmfulness_safety
#SBATCH --qos=exception
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:2
#SBATCH --array=1-1
#SBATCH --output=/dev/null
#SBATCH --constraint=ampere

# =============================================================================
# REPE Harmfulness vs Safety Experiment Runner
# =============================================================================
# This script runs experiments to analyze the trade-off between harmfulness
# and safety in language models using Representation Engineering (REPE).
# It evaluates models on StereoSet dataset with various coefficient values.
# =============================================================================

# Environment setup
export HF_DATASETS_CACHE='/path/to/huggingface/datasets/cache'
export HF_HOME='/path/to/huggingface/cache'

# Activate conda environment
source /path/to/miniconda3/envs/phi3_env/bin/activate

# =============================================================================
# CONFIGURATION
# =============================================================================

# Available MMLU dataset categories for reference
declare -A DATASET_MMLU_NAMES=(
    ["1"]="high_school_computer_science"
    ["2"]="clinical_knowledge"
    ["3"]="medical_genetics"
    ["4"]="international_law"
)

# Available model configurations
declare -A MODEL_NAMES=(
    ["1"]="meta-llama/Meta-Llama-3.1-8B"
    ["2"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
    ["3"]="meta-llama/Llama-2-13b-hf"
    ["4"]="meta-llama/Llama-2-13b-chat-hf"
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Create directory if it doesn't exist
mkdir_if_not_exists() {
    if [ -d "$1" ]; then
        echo "Directory '$1' already exists."
    else
        mkdir -p "$1"
        echo "Directory '$1' created."
    fi
}

# =============================================================================
# EXPERIMENT SETUP
# =============================================================================

# Generate timestamp for unique output directory
current_time=$(date +"%d-%m_%H-%M-%S")

# Set up output directory
OUTPUT_DIR="/path/to/experiment/outputs/harmfulness_experiments/safety_${current_time}"
mkdir_if_not_exists "$OUTPUT_DIR"

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

# Model configuration
MODEL_NAME="--model_name meta-llama/Meta-Llama-3.1-8B"

# Dataset configuration (StereoSet for safety evaluation)
DATASET_PATH="--dataset_path /path/to/stereoset_dataset"

# Coefficient range for REPE intervention
START_COEFF="--start_coeff -5.0"
END_COEFF="--end_coeff 5.1"
STEP_COEFF="--coeff_step 0.2"

# Alternative coefficient settings (commented out)
# START_COEFF="--start_coeff -10.0"
# END_COEFF="--end_coeff 10.2"
# STEP_COEFF="--coeff_step 0.5"

# Experiment parameters
NUM_INSTRUCTIONS="--num_instructions 96"
NUM_SAMPLES="--num_samples 1"

# Optional: Enable synthetic reading vectors (uncomment if needed)
# IS_SYNTH_READING_VECS="--is_synth_reading_vectors"

# Output configuration
OUTPUT_DIR_ARG="--output_dir $OUTPUT_DIR"

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

# Build command string
CMD="python experiments/harmfulness_experiments/harmfulness_safety_new.py \
    $MODEL_NAME \
    $DATASET_PATH \
    $START_COEFF \
    $END_COEFF \
    $STEP_COEFF \
    $NUM_INSTRUCTIONS \
    $NUM_SAMPLES \
    $OUTPUT_DIR_ARG"

# Add synthetic reading vectors flag if defined
if [ ! -z "$IS_SYNTH_READING_VECS" ]; then
    CMD="$CMD $IS_SYNTH_READING_VECS"
fi

# Save command and metadata
echo "Experiment started at: $(date)" > "$OUTPUT_DIR/metadata.txt"
echo "Command executed:" >> "$OUTPUT_DIR/metadata.txt"
echo "$CMD" >> "$OUTPUT_DIR/metadata.txt"

# Execute experiment with logging
echo "Starting harmfulness vs safety experiment..."
echo "Output directory: $OUTPUT_DIR"
echo "Command: $CMD"

$CMD 2>&1 | tee "$OUTPUT_DIR/log.txt"

echo "Experiment completed. Results saved to: $OUTPUT_DIR"