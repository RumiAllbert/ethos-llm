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

from src.analysis.dilemma_analyzer import DilemmaAnalyzer
from src.data.dataset_handlers import DailyDilemmaDataset
from src.models.ollama_client import OllamaClient
from src.utils.dashboard import create_progress_dashboard, update_progress
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


def load_config(config_path: str) -> dict[str, Any]:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def query_model(model_name: str, messages: list[dict[str, str]], config: dict[str, Any]) -> str:
    """Query the model with retry logic and timeout.

    Args:
        model_name: Name of the model to query
        messages: List of message dictionaries
        config: Configuration dictionary

    Returns:
        Model response text
    """
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
    # Create necessary directories
    results_dir = Path(args.output_dir)
    raw_dir = results_dir / "raw_responses"
    plots_dir = results_dir / "plots"
    analysis_dir = results_dir / "analysis"

    for directory in [results_dir, raw_dir, plots_dir, analysis_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Apply max_samples override if provided
    if args.max_samples is not None:
        for dataset_config in config["datasets"].values():
            dataset_config["max_samples"] = args.max_samples
            logger.info(f"Overriding max_samples to {args.max_samples}")

    # Initialize dataset handler
    try:
        dataset_handler = DailyDilemmaDataset(config)
        scenarios, metadata_list = dataset_handler.process_dataset()
        logger.info(f"Loaded {len(scenarios)} scenarios from dataset")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Initialize OllamaClient
    try:
        client = OllamaClient(config)
        logger.info("Initialized OllamaClient successfully")
    except Exception as e:
        logger.error(f"Error initializing OllamaClient: {e}")
        raise

    # Initialize dashboard for progress tracking
    model_names = [model["name"] for model in config["models"]]
    framework_names = [framework["name"] for framework in config["frameworks"]]
    dashboard = create_progress_dashboard(
        total_scenarios=len(scenarios),
        models=model_names,
        frameworks=framework_names,
        save_dir=results_dir,
    )
    logger.info("Created progress dashboard")

    # Prepare results storage
    results = []

    # Process each scenario
    for i, (scenario, metadata) in enumerate(zip(scenarios, metadata_list, strict=False)):
        logger.info(f"Processing scenario {i + 1}/{len(scenarios)}")

        for model_config in config["models"]:
            model_name = model_config["name"]
            logger.info(f"Using model: {model_name}")

            # Get baseline response
            try:
                baseline_response, is_censored = client.query_baseline(model_name, scenario)
                logger.info(f"Got baseline response (censored: {is_censored})")

                # Save raw baseline response
                baseline_file = raw_dir / f"baseline_{i}_{model_name.replace(':', '_')}.txt"
                with open(baseline_file, "w") as f:
                    f.write(baseline_response)
            except Exception as e:
                logger.error(f"Error getting baseline response: {e}")
                baseline_response = f"Error: {str(e)}"
                is_censored = False

            # Process each framework
            for framework in config["frameworks"]:
                framework_name = framework["name"]
                logger.info(f"Applying framework: {framework_name}")

                try:
                    # Skip control framework for baseline comparison
                    if framework_name == "control":
                        framework_response = baseline_response
                        framework_censored = is_censored
                    else:
                        # Get framework response
                        framework_response, framework_censored = client.query_framework(
                            model_name, scenario, baseline_response, framework_name
                        )

                    # Save raw framework response
                    framework_file = (
                        raw_dir / f"{framework_name}_{i}_{model_name.replace(':', '_')}.txt"
                    )
                    with open(framework_file, "w") as f:
                        f.write(framework_response)

                    # Store result
                    result = {
                        "scenario_id": i,
                        "model": model_name,
                        "framework": framework_name,
                        "scenario": scenario,
                        "baseline_response": baseline_response,
                        "framework_response": framework_response,
                        "baseline_censored": is_censored,
                        "framework_censored": framework_censored,
                    }

                    # Add metadata
                    for key, value in metadata.items():
                        result[f"metadata_{key}"] = value

                    results.append(result)

                    # Update dashboard
                    update_progress(dashboard, str(i), model_name, results_dir)

                except Exception as e:
                    logger.error(f"Error processing framework {framework_name}: {e}")

            # Add delay between model calls to avoid rate limiting
            time.sleep(config["experiment"].get("delay_between_calls", 1))

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(results_dir / "results.csv", index=False)
    logger.info(f"Results saved to {results_dir / 'results.csv'}")

    # Also save to the analysis directory for compatibility with analyze_results.py
    results_df.to_csv(analysis_dir / "responses.csv", index=False)
    logger.info(f"Results also saved to {analysis_dir / 'responses.csv'}")

    return results_df


def main():
    """Main function to run the experiment."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level)

    # Load configuration
    config = load_config(args.config)

    # Create results directory
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration to results directory
    try:
        # Save full configuration
        with open(results_dir / "config.json", "w") as f:
            json.dump(deep_convert_config(config), f, indent=2)

        # Save simplified configuration (strings only)
        simplified_config = {}
        for key, value in deep_convert_config(config).items():
            simplified_config[key] = str(value)

        with open(results_dir / "config_simplified.json", "w") as f:
            json.dump(simplified_config, f, indent=2)

        logger.info(f"Configuration saved to {results_dir}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")

    # Run experiment
    logger.info("Starting experiment with enhanced analysis")
    try:
        results_df = run_experiment(config, args)

        # Initialize analyzer and calculate metrics
        analyzer = DilemmaAnalyzer(save_dir=results_dir / "plots")

        # Generate visualizations if we have results
        if not results_df.empty:
            logger.info("Generating visualizations")
            try:
                analyzer.create_enhanced_visualizations(results_df)
                logger.info("Visualizations created successfully")
            except Exception as e:
                logger.error(f"Error creating visualizations: {e}")

        logger.info("Experiment completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
