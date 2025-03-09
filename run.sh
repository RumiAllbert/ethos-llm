#!/bin/bash
# Script to run the entire LLM Ethics Experiment pipeline

set -e  # Exit on error

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p data results

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

# Download datasets
print_header "Downloading datasets"
poetry run python scripts/download_datasets.py

# Run experiment
print_header "Running experiment"
echo "This may take a long time depending on the number of scenarios and models."
read -p "Proceed with experiment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    poetry run python scripts/run_experiment.py
else
    echo "Experiment skipped. Exiting."
    exit 0
fi

# Analyze results
print_header "Analyzing results"
poetry run python scripts/analyze_results.py --results-dir results

# All done
print_header "Experiment completed!"
echo "Results and analysis are available in the 'results' directory."
echo "Summary report: results/analysis/summary.md"
echo "Visualizations: results/plots/" 