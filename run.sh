#!/bin/bash
# Script to run the entire LLM Ethics Experiment pipeline

set -e  # Exit on error

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

# Default values
RESUME=false
PARALLEL=1
MAX_SAMPLES=""
OUTPUT_DIR="results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --resume)
      RESUME=true
      shift
      ;;
    --parallel)
      PARALLEL="$2"
      shift 2
      ;;
    --max-samples|-n)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --output-dir|-o)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--resume] [--parallel N] [--max-samples N] [--output-dir DIR]"
      exit 1
      ;;
  esac
done

# Create necessary directories
mkdir -p data "$OUTPUT_DIR"

# Function to print section headers
print_header() {
    echo
    echo "================================================================================"
    echo "  $1"
    echo "================================================================================"
    echo
}

# Function to check if Ollama is running
check_ollama() {
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Error: Ollama service is not running."
        echo "Please start Ollama using 'ollama serve' and try again."
        exit 1
    fi
}

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Error: Poetry is not installed."
    echo "Please install Poetry from https://python-poetry.org/docs/#installation"
    exit 1
fi

# Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed."
    echo "Please install Ollama from https://ollama.com/download"
    exit 1
fi

# Check if Ollama is running
check_ollama

# Check if required models are pulled
print_header "Checking required models"
MODELS=("llama3.2:8b" "deepseek-r1:8b")
MODELS_TO_PULL=()

for model in "${MODELS[@]}"; do
    if ! ollama list | grep -q "$model"; then
        MODELS_TO_PULL+=("$model")
    else
        echo "✅ Model $model is already pulled"
    fi
done

# Pull missing models
if [ ${#MODELS_TO_PULL[@]} -gt 0 ]; then
    echo "The following models need to be pulled:"
    for model in "${MODELS_TO_PULL[@]}"; do
        echo "  - $model"
    done
    
    read -p "Do you want to pull these models now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for model in "${MODELS_TO_PULL[@]}"; do
            echo "Pulling $model..."
            ollama pull "$model"
        done
    else
        echo "Cannot proceed without required models. Exiting."
        exit 1
    fi
fi

# Install dependencies
print_header "Installing dependencies"
poetry install

# Check for checkpoint
CHECKPOINT_FILE="$OUTPUT_DIR/checkpoint.json"
RESUME_FLAG=""
if $RESUME && [ -f "$CHECKPOINT_FILE" ]; then
    print_header "Checkpoint detected"
    echo "A checkpoint file was found at $CHECKPOINT_FILE"
    echo "The experiment will resume from where it left off."
    RESUME_FLAG="--resume"
else
    if $RESUME; then
        echo "No checkpoint file found at $CHECKPOINT_FILE"
        echo "Starting a new experiment..."
    fi
fi

# Download datasets
print_header "Downloading datasets"
CMD="poetry run python scripts/download_datasets.py"
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi
echo "Executing: $CMD"
eval "$CMD"

# Run experiment
print_header "Running experiment"
echo "This may take a long time depending on the number of scenarios and models."
echo "Options:"
echo "  - Resume from checkpoint: $RESUME"
echo "  - Parallel workers: $PARALLEL"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  - Max samples per dataset: $MAX_SAMPLES"
fi
echo "  - Output directory: $OUTPUT_DIR"

read -p "Proceed with experiment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    CMD="poetry run python scripts/run_experiment.py --output-dir $OUTPUT_DIR --parallel $PARALLEL $RESUME_FLAG"
    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max-samples $MAX_SAMPLES"
    fi
    echo "Executing: $CMD"
    eval "$CMD"
else
    echo "Experiment skipped. Exiting."
    exit 0
fi

# Analyze results
print_header "Analyzing results"
poetry run python scripts/analyze_results.py --results-dir "$OUTPUT_DIR"

# All done
print_header "Experiment completed!"
echo "Results and analysis are available in the '$OUTPUT_DIR' directory."
echo "Summary report: $OUTPUT_DIR/analysis/summary.md"
echo "Visualizations: $OUTPUT_DIR/plots/"

# Provide information about how to resume if interrupted
print_header "Resuming Information"
echo "If you need to resume this experiment later, run:"
echo "./run.sh --resume --output-dir $OUTPUT_DIR"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  --max-samples $MAX_SAMPLES"
fi
if [ "$PARALLEL" -ne 1 ]; then
    echo "  --parallel $PARALLEL"
fi 