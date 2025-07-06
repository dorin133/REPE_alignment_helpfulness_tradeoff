#!/bin/bash

# =============================================================================
# Fairness Helpfulness Evaluation Job Script
# =============================================================================
# This script runs fairness helpfulness evaluation experiments using SLURM job scheduler.
# It evaluates the impact of representation engineering on model helpfulness across
# different MMLU datasets and model configurations, analyzing the trade-off between
# fairness interventions and model performance.
# =============================================================================

# SLURM job configuration
#SBATCH -p g48                           # Partition/queue name
#SBATCH --job-name=REPE_Fairness_Help    # Job name for identification
#SBATCH --qos=exception                  # Quality of service
#SBATCH --nodes=1                        # Number of compute nodes
#SBATCH --ntasks=1                       # Number of parallel tasks
#SBATCH --cpus-per-task=30              # CPU cores per task
#SBATCH --gres=gpu:2                     # GPU resources (2 GPUs)
#SBATCH --array=1-1                      # Job array range (single job)
#SBATCH --output=/dev/null               # Discard standard output (using custom log)
#SBATCH --constraint=ampere              # GPU architecture constraint

# =============================================================================
# Environment Configuration
# =============================================================================

# Set Hugging Face cache directories to avoid conflicts and save space
export HF_DATASETS_CACHE='/path/to/cache/huggingface/datasets'
export HF_HOME='/path/to/cache/huggingface'

# =============================================================================
# Configuration Arrays
# =============================================================================

# Available MMLU dataset options for fairness evaluation
declare -A DATASET_MMLU_NAMES=(
    ["1"]="high_school_computer_science"
    ["2"]="clinical_knowledge"
    ["3"]="medical_genetics"
    ["4"]="international_law"
)

# Available model configurations for testing
declare -A MODEL_NAMES=(
    ["1"]="meta-llama/Meta-Llama-3.1-8B"
    ["2"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
    ["3"]="meta-llama/Llama-2-13b-hf"
    ["4"]="meta-llama/Llama-2-13b-chat-hf"
)

# =============================================================================
# Utility Functions
# =============================================================================

# Function: setup_conda_environment
# Description: Activate the required conda environment for the experiment
setup_conda_environment() {
    local env_path="/path/to/conda/envs/experiment_env/bin/activate"
    
    if [ -f "$env_path" ]; then
        source "$env_path"
        echo "Conda environment activated successfully."
    else
        echo "Error: Conda environment not found at $env_path"
        exit 1
    fi
}

# Function: mkdir_if_not_exists
# Description: Create directory if it doesn't exist, with informative output
# Parameters:
#   $1 - Directory path to create
mkdir_if_not_exists() {
    local dir_path="$1"
    
    if [ -d "$dir_path" ]; then
        echo "Directory '$dir_path' already exists."
    else
        mkdir -p "$dir_path"
        echo "Directory '$dir_path' created successfully."
    fi
}

# Function: generate_timestamp
# Description: Generate timestamp for unique output directory naming
# Returns: Formatted timestamp string (DD-MM_HH-MM-SS)
generate_timestamp() {
    date +"%d-%m_%H-%M-%S"
}

# Function: setup_output_directory
# Description: Create timestamped output directory for experiment results
# Returns: Path to created output directory
setup_output_directory() {
    local base_dir="/path/to/experiment/outputs/fairness_experiments"
    local timestamp=$(generate_timestamp)
    local output_dir="${base_dir}/helpfulness_${timestamp}"
    
    mkdir_if_not_exists "$output_dir"
    echo "$output_dir"
}

# Function: build_experiment_command
# Description: Construct the Python command with all necessary parameters
# Parameters:
#   $1 - Output directory path
# Returns: Complete command string
build_experiment_command() {
    local output_dir="$1"
    
    # Model configuration
    local model_name="--model_name meta-llama/Llama-2-13b-chat-hf"
    
    # Dataset configuration
    local dataset_path="--dataset_path lukaemon/mmlu"
    local dataset_names="--dataset_names international_law,medical_genetics,high_school_computer_science"
    
    # Coefficient range configuration (extended range for thorough analysis)
    local start_coeff="--start_coeff -10.0"
    local end_coeff="--end_coeff 10.2"
    local step_coeff="--coeff_step 0.25"
    
    # Alternative coefficient range (smaller, faster testing)
    # local start_coeff="--start_coeff -5.0"
    # local end_coeff="--end_coeff 5.1"
    # local step_coeff="--coeff_step 0.2"
    
    # Experiment parameters
    local num_instructions="--num_instructions 96"
    local num_samples="--num_samples 1"
    local output_dir_arg="--output_dir $output_dir"
    
    # Optional flags (uncomment as needed)
    # local is_synth_reading_vecs="--is_synth_reading_vectors"
    
    # Construct full command
    local cmd="python experiments/fairness_experiments/fairness_helpfulness_new.py \
                    $model_name \
                    $dataset_path \
                    $dataset_names \
                    $start_coeff \
                    $end_coeff \
                    $step_coeff \
                    $num_instructions \
                    $num_samples \
                    $output_dir_arg"
    
    echo "$cmd"
}

# Function: save_experiment_metadata
# Description: Save experiment configuration and command to metadata file
# Parameters:
#   $1 - Command string
#   $2 - Output directory path
save_experiment_metadata() {
    local cmd="$1"
    local output_dir="$2"
    local metadata_file="$output_dir/metadata.txt"
    
    {
        echo "# Fairness Helpfulness Experiment Metadata"
        echo "# Generated on: $(date)"
        echo "# SLURM Job ID: $SLURM_JOB_ID"
        echo "# Node: $SLURMD_NODENAME"
        echo ""
        echo "# Experiment Command:"
        echo "$cmd"
        echo ""
        echo "# Environment Variables:"
        echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
        echo "HF_HOME=$HF_HOME"
        echo ""
        echo "# Dataset Configuration:"
        echo "Available MMLU datasets:"
        for key in "${!DATASET_MMLU_NAMES[@]}"; do
            echo "  [$key]: ${DATASET_MMLU_NAMES[$key]}"
        done
        echo ""
        echo "Available models:"
        for key in "${!MODEL_NAMES[@]}"; do
            echo "  [$key]: ${MODEL_NAMES[$key]}"
        done
    } > "$metadata_file"
    
    echo "Experiment metadata saved to: $metadata_file"
}

# Function: run_experiment
# Description: Execute the fairness helpfulness experiment with logging
# Parameters:
#   $1 - Command string
#   $2 - Output directory path
run_experiment() {
    local cmd="$1"
    local output_dir="$2"
    local log_file="$output_dir/log.txt"
    
    echo "Starting fairness helpfulness experiment..."
    echo "Output directory: $output_dir"
    echo "Log file: $log_file"
    echo ""
    echo "Command: $cmd"
    echo ""
    
    # Execute command with output redirection
    eval "$cmd" 2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "Experiment completed successfully!"
        echo "Results saved to: $output_dir"
    else
        echo "Experiment failed with exit code: $exit_code"
        echo "Check log file for details: $log_file"
        exit $exit_code
    fi
}

# Function: validate_environment
# Description: Check if required files and directories exist before running
validate_environment() {
    local python_script="experiments/fairness_experiments/fairness_helpfulness_new.py"
    
    if [ ! -f "$python_script" ]; then
        echo "Error: Python script not found: $python_script"
        echo "Please ensure you're running from the correct directory."
        exit 1
    fi
    
    echo "Environment validation passed."
}

# Function: print_experiment_info
# Description: Display experiment configuration information
print_experiment_info() {
    echo "Experiment Configuration:"
    echo "  Available MMLU Datasets:"
    for key in "${!DATASET_MMLU_NAMES[@]}"; do
        echo "    [$key]: ${DATASET_MMLU_NAMES[$key]}"
    done
    echo ""
    echo "  Available Models:"
    for key in "${!MODEL_NAMES[@]}"; do
        echo "    [$key]: ${MODEL_NAMES[$key]}"
    done
    echo ""
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo "=========================================="
    echo "Fairness Helpfulness Evaluation Experiment"
    echo "=========================================="
    echo "Starting at: $(date)"
    echo ""
    
    # Validate environment
    validate_environment
    
    # Setup environment
    setup_conda_environment
    
    # Display experiment information
    print_experiment_info
    
    # Setup output directory
    local output_dir=$(setup_output_directory)
    
    # Build experiment command
    local cmd=$(build_experiment_command "$output_dir")
    
    # Save metadata
    save_experiment_metadata "$cmd" "$output_dir"
    
    # Run experiment
    run_experiment "$cmd" "$output_dir"
    
    echo ""
    echo "=========================================="
    echo "Experiment completed at: $(date)"
    echo "Results saved to: $output_dir"
    echo "=========================================="
}

# Execute main function with all command line arguments
main "$@"