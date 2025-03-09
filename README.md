# LLM Ethics Experiment

This project evaluates how small, locally deployable LLMs respond to moral scenarios when prompted to adopt different ethical reasoning frameworks (utilitarian, deontological, and virtue-ethics).

## Overview

The experiment tests two models:
- Llama 3.2 (8B)
- DeepSeek-Reasoner-1 (8B)

Using two datasets:
- EthicsSuite
- Moral Stories

The goal is to measure the models' alignment stability and ability to adapt to different ethical paradigms.

## Setup

### Prerequisites

- Python 3.10+
- Poetry (dependency management)
- Ollama (for running local LLMs)

### Installation

1. Install Ollama by following the instructions at [https://ollama.com/](https://ollama.com/)

2. Pull the required models:
```bash
ollama pull llama3.2:8b
ollama pull deepseek-r1:8b
```

3. Install project dependencies:
```bash
poetry install
```

### Running the Experiments

1. Download the datasets:
```bash
poetry run python scripts/download_datasets.py
```

2. Run the experiments:
```bash
poetry run python scripts/run_experiment.py
```

3. Analyze the results:
```bash
poetry run python scripts/analyze_results.py
```

## Project Structure

```
llm-ethic/
├── src/                # Source code
│   ├── data/           # Data handling code
│   ├── models/         # Model interaction code
│   ├── utils/          # Utility functions
│   ├── config/         # Configuration files
│   └── analysis/       # Analysis code
├── scripts/            # Scripts for running experiments
├── notebooks/          # Jupyter notebooks for exploration
├── tests/              # Test code
├── pyproject.toml      # Project dependencies
└── README.md           # This file
```

## License
