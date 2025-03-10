#!/bin/bash
# Script to run the entire LLM Ethics Experiment pipeline

set -e  # Exit on error

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

# Default values
PARALLEL=1
OUTPUT_DIR="results"
MAX_SAMPLES=""
MODELS=""
FRAMEWORKS=""
DATASETS=""
RESUME=false
DEBUG=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --frameworks)
            FRAMEWORKS="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
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
MODELS_LIST=("llama3.2:latest" "deepseek-r1:8b")
MODELS_TO_PULL=()

for model in "${MODELS_LIST[@]}"; do
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

# Fix Python package structure issues
print_header "Setting up environment"
echo "Ensuring project can be imported as a Python package..."

# Ensure we have an __init__.py in scripts directory for fix_path module
if [ ! -f "scripts/__init__.py" ]; then
    echo "Creating scripts/__init__.py"
    echo '"""Scripts package for the LLM Ethics Experiment."""' > scripts/__init__.py
fi

# Install dependencies and project in development mode
print_header "Installing dependencies"
echo "Installing project in development mode..."
poetry install -v --no-root

# Build the command with only provided arguments
CMD="python scripts/run_experiment.py"

if [ ! -z "$PARALLEL" ]; then
    CMD="$CMD --parallel $PARALLEL"
fi

if [ ! -z "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

if [ ! -z "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

if [ ! -z "$MODELS" ]; then
    CMD="$CMD --models $MODELS"
fi

if [ ! -z "$FRAMEWORKS" ]; then
    CMD="$CMD --frameworks $FRAMEWORKS"
fi

if [ ! -z "$DATASETS" ]; then
    CMD="$CMD --datasets $DATASETS"
fi

if [ "$RESUME" = true ]; then
    CMD="$CMD --resume"
fi

if [ "$DEBUG" = true ]; then
    CMD="$CMD --debug"
fi

# Print the command that will be executed
echo "This may take a long time depending on the number of scenarios and models."
echo "Options:"
echo "  - Resume from checkpoint: $RESUME"
echo "  - Parallel workers: $PARALLEL"
echo "  - Output directory: $OUTPUT_DIR"
if [ ! -z "$MAX_SAMPLES" ]; then
    echo "  - Max samples per dataset: $MAX_SAMPLES"
fi

# Ask for confirmation
read -p "Proceed with experiment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Executing: $CMD"
    eval "$CMD"
    
    # If experiment completed successfully, run analysis
    if [ $? -eq 0 ]; then
        print_header "Analyzing results"
        python scripts/analyze_results.py --results-dir "$OUTPUT_DIR"
        
        # All done
        print_header "Experiment completed!"
        echo "Results and analysis are available in the '$OUTPUT_DIR' directory."
        echo "Summary report: $OUTPUT_DIR/analysis/summary.md"
        echo "Visualizations: $OUTPUT_DIR/plots/"
    else
        echo "Experiment failed. Check logs for details."
        exit 1
    fi
else
    echo "Experiment cancelled."
    exit 1
fi

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