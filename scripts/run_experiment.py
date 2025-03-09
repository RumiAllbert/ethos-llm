#!/usr/bin/env python
"""Main script for running the LLM Ethics Experiment.

This script loads the configuration, datasets, and models, and runs the
experiment with the specified ethical frameworks.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.analysis.metrics import ResponseAnalyzer
from src.data.dataset import DatasetLoader
from src.models.ollama_client import OllamaClient
from src.utils.config import load_config, setup_experiment_dirs
from src.utils.logging import get_experiment_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Run the LLM Ethics Experiment")
    parser.add_argument("--config", "-c", type=str, help="Path to the configuration file")
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
    parser.add_argument(
        "--output-dir", "-o", type=str, help="Directory to save results (overrides config)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def run_experiment(
    config: dict[str, Any], args: argparse.Namespace | None = None
) -> dict[str, Any]:
    """Run the full experiment pipeline.

    Args:
        config: Configuration dictionary
        args: Optional command line arguments (overrides config settings)

    Returns:
        Dictionary with experiment results
    """
    logger = get_experiment_logger("experiment")
    logger.info("Starting LLM Ethics Experiment")

    # Apply command line overrides
    if args:
        if args.models:
            config["models"] = [model for model in config["models"] if model["name"] in args.models]
        if args.frameworks:
            config["frameworks"] = [
                fw for fw in config["frameworks"] if fw["name"] in args.frameworks
            ]
        if args.datasets and all(ds in config["datasets"] for ds in args.datasets):
            config["datasets"] = {k: v for k, v in config["datasets"].items() if k in args.datasets}
        if args.max_samples:
            for ds in config["datasets"].values():
                ds["max_samples"] = args.max_samples
        if args.output_dir:
            config["experiment"]["save_dir"] = args.output_dir

    # Setup experiment directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["experiment"]["run_id"] = timestamp
    save_dir = Path(config["experiment"]["save_dir"])
    setup_experiment_dirs(config)

    # Save the configuration
    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved configuration to {config_path}")

    # Load datasets
    logger.info("Loading datasets")
    dataset_loader = DatasetLoader(config)
    datasets = dataset_loader.load_all_datasets()

    # Log dataset statistics
    total_scenarios = sum(len(scenarios) for scenarios in datasets.values())
    logger.info(f"Loaded {total_scenarios} total scenarios across {len(datasets)} datasets")
    for name, scenarios in datasets.items():
        logger.info(f"  {name}: {len(scenarios)} scenarios")

    # Initialize Ollama client
    logger.info("Initializing Ollama client")
    client = OllamaClient(config)

    # Initialize response analyzer
    logger.info("Initializing response analyzer")
    analyzer = ResponseAnalyzer(config)

    # Get list of models and frameworks to test
    models = [model["name"] for model in config["models"]]
    frameworks = [fw["name"] for fw in config["frameworks"]]

    logger.info(f"Testing models: {', '.join(models)}")
    logger.info(f"Testing frameworks: {', '.join(frameworks)}")

    # Process scenarios
    all_scenarios = dataset_loader.get_all_scenarios()
    results = []

    logger.info(
        f"Processing {len(all_scenarios)} scenarios with {len(models)} models and {len(frameworks)} frameworks"
    )
    scenario_count = 0

    for scenario in all_scenarios:
        scenario_count += 1
        logger.info(f"Processing scenario {scenario_count}/{len(all_scenarios)}: {scenario.id}")

        # Process each model
        for model in models:
            logger.info(f"  Using model: {model}")

            # Process scenario with all frameworks
            try:
                result = client.process_scenario(scenario, model, frameworks)
                results.append(result)

                # Save raw response data
                response_path = save_dir / "raw_responses" / f"{scenario.id}_{model}.json"
                with open(response_path, "w") as f:
                    json.dump(result, f, indent=2)

            except Exception as e:
                logger.error(f"Error processing scenario {scenario.id} with model {model}: {e}")

    logger.info(f"Completed processing {len(results)} scenario-model combinations")

    # Analyze results
    logger.info("Analyzing results")
    analysis = analyzer.analyze_results(results)

    # Save analysis
    analysis_dir = save_dir / "analysis"

    # Save full analysis to JSON (convert DataFrame to records)
    analysis_copy = analysis.copy()
    analysis_copy["dataframe"] = analysis["dataframe"].to_dict(orient="records")

    with open(analysis_dir / "analysis.json", "w") as f:
        json.dump(analysis_copy, f, indent=2)

    # Save DataFrame to CSV
    analysis["dataframe"].to_csv(analysis_dir / "responses.csv", index=False)

    # Generate and save summary
    summary = analyzer.generate_summary(analysis)
    with open(analysis_dir / "summary.md", "w") as f:
        f.write(summary)

    logger.info(f"Saved analysis to {analysis_dir}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(summary)
    print("=" * 80)

    logger.info("Experiment completed successfully!")

    return {"config": config, "results": results, "analysis": analysis}


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level)

    # Load configuration
    config = load_config(args.config)

    # Run experiment
    run_experiment(config, args)
