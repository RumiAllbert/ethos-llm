#!/usr/bin/env python
"""Main script for running the LLM Ethics Experiment.

This script loads the configuration, datasets, and models, and runs the
experiment with the specified ethical frameworks.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

# Add the project root to the Python path
import fix_path  # noqa
import ollama
import pandas as pd
import yaml
from tqdm import tqdm

from src.analysis.dilemma_analyzer import DilemmaAnalyzer
from src.data.dataset import ScenarioItem
from src.data.dataset_handlers import DailyDilemmaDataset
from src.models.ollama_client import OllamaClient
from src.utils.config import load_config
from src.utils.dashboard import (
    should_skip_combination,
    update_progress,
)
from src.utils.logging import get_experiment_logger, setup_logging

# Get logger for this module
logger = get_experiment_logger("experiment")


def deep_convert_config(obj: Any) -> Any:
    """Recursively convert OmegaConf objects to standard Python types.

    Args:
        obj: Object to convert, can be of any type

    Returns:
        Same object converted to standard Python types
    """
    # First try using OmegaConf's to_container if available
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        # If it's an OmegaConf container, convert it first
        if isinstance(obj, (DictConfig, ListConfig)):
            return deep_convert_config(OmegaConf.to_container(obj, resolve=True))

    except ImportError:
        # OmegaConf not available, proceed with regular conversion
        pass

    # Handle dictionaries by recursively converting their values
    if isinstance(obj, dict):
        return {k: deep_convert_config(v) for k, v in obj.items()}

    # Handle lists by recursively converting their items
    elif isinstance(obj, list):
        return [deep_convert_config(item) for item in obj]

    # Handle OmegaConf object attributes if this is some other object with __dict__
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        try:
            # Try to convert to a dictionary if it has a __dict__
            return deep_convert_config(obj.__dict__)
        except (AttributeError, TypeError):
            # If conversion fails, return as is
            return obj

    # Return other types as is (int, str, bool, None, etc.)
    return obj


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Run the LLM Ethics Experiment")
    parser.add_argument("--config", default="src/config/config.yaml", help="Path to config file")
    parser.add_argument("--output-dir", "-o", default="results", help="Directory to save results")
    parser.add_argument(
        "--models", type=str, nargs="+", help="List of models to use (overrides config)"
    )
    parser.add_argument(
        "--frameworks",
        type=str,
        nargs="+",
        help="List of ethical frameworks to test (overrides config)",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", help="List of datasets to use (overrides config)"
    )
    parser.add_argument(
        "--max-samples",
        "-n",
        type=int,
        help="Maximum number of samples to use from each dataset (overrides config)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument(
        "--parallel", type=int, default=1, help="Number of parallel workers (default: 1)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def process_scenario_parallel(
    scenario: ScenarioItem,
    model: str,
    frameworks: list[str],
    client: OllamaClient,
    save_dir: Path,
    completed_combinations: set[str],
    dashboard: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Process a scenario with a model and frameworks (for parallel execution).

    Args:
        scenario: The scenario to process
        model: The model to use
        frameworks: List of frameworks to test
        client: OllamaClient instance
        save_dir: Directory to save results
        completed_combinations: Set of already completed combinations
        dashboard: Optional dashboard for tracking progress

    Returns:
        Result dictionary or None if skipped/error
    """
    # Check if this combination has already been processed
    if should_skip_combination(scenario.id, model, completed_combinations):
        logger.info(f"Skipping already processed scenario {scenario.id} with model {model}")
        return None

    try:
        # Process the scenario
        result = client.process_scenario(scenario, model, frameworks)

        # Save raw response data
        response_path = save_dir / "raw_responses" / f"{scenario.id}_{model}.json"
        with open(response_path, "w") as f:
            json.dump(result, f, indent=2)

        # Update progress if dashboard is provided
        if dashboard is not None:
            update_progress(dashboard, scenario.id, model, save_dir)

        return result
    except Exception as e:
        logger.error(f"Error processing scenario {scenario.id} with model {model}: {e}")
        return None


def load_config(config_path: str) -> dict[str, Any]:
    """Load experiment configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def query_model(model_name: str, messages: list[dict[str, str]], config: dict[str, Any]) -> str:
    """Query the model with retry logic and timeout."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model_name,
                messages=messages,
                options={"timeout": config["experiment"]["ollama_timeout"]},
            )
            return response["message"]["content"]
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to query {model_name} after {max_retries} attempts: {e}")
                return f"Error: {str(e)}"
            time.sleep(2**attempt)  # Exponential backoff
    return "Error: Maximum retries exceeded"


def run_experiment(config: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    """
    Run the enhanced LLM ethics experiment.

    Args:
        config: Experiment configuration
        args: Command line arguments

    Returns:
        DataFrame containing experiment results
    """
    results = []

    # Initialize dataset handler
    dataset_handler = DailyDilemmaDataset(config)
    scenarios, metadata_list = dataset_handler.process_dataset()

    # Initialize analyzer
    analyzer = DilemmaAnalyzer(save_dir=Path(args.output_dir) / "plots")

    # Process each scenario
    for scenario, metadata in tqdm(
        zip(scenarios, metadata_list, strict=False), total=len(scenarios)
    ):
        for model in config["models"]:
            model_name = model["name"]

            # Get baseline response
            baseline_messages = [
                {
                    "role": "user",
                    "content": f"Consider the following ethical scenario and provide your reasoning and recommendation:\n\n{scenario}",
                }
            ]
            baseline_response = query_model(model_name, baseline_messages, config)

            # Process each framework
            for framework in config["frameworks"]:
                # Get framework response
                framework_messages = baseline_messages + [
                    {"role": "assistant", "content": baseline_response},
                    {"role": "user", "content": framework["prompt"]},
                ]
                framework_response = query_model(model_name, framework_messages, config)

                # Calculate metrics
                value_alignment = analyzer.analyze_value_alignment(
                    framework_response, metadata["values"]
                )

                # Store results
                result = {
                    "model": model_name,
                    "framework": framework["name"],
                    "scenario": scenario,
                    "baseline_response": baseline_response,
                    "framework_response": framework_response,
                    "value_alignment": sum(value_alignment.values()) / len(value_alignment)
                    if value_alignment
                    else 0,
                    "censored": any(
                        phrase in framework_response.lower()
                        for phrase in config["censorship_phrases"]
                    ),
                    "stance_changed": True,  # Will be updated in post-processing
                    **metadata,  # Include all metadata
                }
                results.append(result)

                # Add delay between calls
                time.sleep(config["experiment"]["delay_between_calls"])

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate stance changes using similarity
    for idx, row in results_df.iterrows():
        baseline_embedding = analyzer.get_text_embedding(str(row["baseline_response"]))
        framework_embedding = analyzer.get_text_embedding(str(row["framework_response"]))
        similarity = float(cosine_similarity([baseline_embedding], [framework_embedding])[0][0])
        results_df.at[idx, "stance_changed"] = (
            similarity < config["experiment"]["similarity_threshold"]
        )

    return results_df


def main():
    """Main function to run the experiment."""
    args = parse_args()

    # Setup logging
    setup_logging()

    # Load configuration
    config = load_config(args.config)

    # Update config with command line arguments
    if args.max_samples:
        for dataset in config["datasets"].values():
            dataset["max_samples"] = args.max_samples

    # Create results directory
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    logging.info("Starting experiment with enhanced analysis")
    results_df = run_experiment(config, args)

    # Save raw results
    results_df.to_csv(results_dir / "enhanced_results.csv", index=False)

    # Initialize analyzer and calculate metrics
    analyzer = DilemmaAnalyzer(save_dir=results_dir / "plots")
    metrics = analyzer.calculate_enhanced_metrics(results_df)

    # Generate visualizations
    analyzer.create_enhanced_visualizations(results_df)

    # Generate and save report
    report = analyzer.generate_analysis_report(results_df, metrics)
    with open(results_dir / "enhanced_analysis_report.md", "w") as f:
        f.write(report)

    logging.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
