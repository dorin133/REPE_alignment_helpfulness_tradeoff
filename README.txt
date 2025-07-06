# REPE Alignment and Helpfulness Tradeoff Experiments

This project investigates the tradeoff between model alignment (safety/harmfulness) and helpfulness in language models using Representation Engineering (REPE) techniques. The experiments measure how safety interventions affect model utility across different dimensions including harmfulness, fairness, and code generation capabilities.

## Project Structure

### `utils/`
Core utilities and shared functionality for REPE experiments:
- **GenArgs.py**: Command-line argument parsing and configuration management
- **generate_code_with_REPE.py**: Code generation using REPE steering techniques
- **generate_reading_vectors.py**: Reading vector generation for representation engineering
- **utils.py**: Common utilities for model loading, sampling, and data processing
- **WrapModel.py**: Model wrapper classes providing consistent interfaces

### `harmfulness_experiments/`
Safety and harmfulness evaluation experiments:
- **harmfulness_helpfulness.py**: Measures tradeoff between harmfulness reduction and helpfulness
- **harmfulness_safety.py**: Safety-focused evaluation and testing
- **harmfulness_utils.py**: Utilities for harmfulness data processing and analysis
- **separation_by_dr.py**: Dimensional reduction analysis for separating harmful vs helpful behaviors
Includes automated shell scripts for batch experiment execution

### `fairness_experiments/`
Fairness evaluation across demographic groups and scenarios:
- **fairness_helpfulness.py**: Evaluates fairness in helpful model responses
- **fairness_safety.py**: Measures fairness in safety-related model behaviors
- **humaneval_experiments/**: Code generation fairness using HumanEval benchmark
Includes automated shell scripts for batch experiment execution

## Key Features
- REPE-based model steering and intervention techniques
- Multi-dimensional evaluation (safety, helpfulness, fairness)
- Integration with MMLU and HumanEval benchmarks
- Statistical analysis and visualization tools
- Support for various language models (Llama family, etc.)
- Batch processing capabilities for large-