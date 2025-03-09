#!/usr/bin/env python
"""Main script for running the LLM Ethics Experiment.

This script loads the configuration, datasets, and models, and runs the
experiment with the specified ethical frameworks.
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from src.analysis.metrics import ResponseAnalyzer
from src.data.dataset import DatasetLoader, ScenarioItem
from src.models.ollama_client import OllamaClient
from src.utils.config import load_config, setup_experiment_dirs
from src.utils.dashboard import (
    create_progress_dashboard,
    load_checkpoint,
    should_skip_combination,
    update_progress,
)
from src.utils.logging import get_experiment_logger, setup_logging

# Get logger for this module
logger = get_experiment_logger("experiment")


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
    if "run_id" not in config["experiment"]:
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

    # Check for checkpoint if resuming
    completed_combinations = set()
    if args and args.resume:
        completed_combinations, has_checkpoint = load_checkpoint(save_dir)
        if has_checkpoint:
            logger.info(
                f"Resuming experiment with {len(completed_combinations)} already processed combinations"
            )
        else:
            logger.info("No checkpoint found. Starting fresh experiment.")

    # Create progress dashboard
    dashboard = create_progress_dashboard(
        total_scenarios=len(all_scenarios), models=models, frameworks=frameworks, save_dir=save_dir
    )

    # Update dashboard with completed combinations from checkpoint
    if completed_combinations:
        dashboard["completed_combinations"] = list(completed_combinations)
        dashboard["completed"] = len(completed_combinations)

        # Update models progress
        for combo in completed_combinations:
            if "_" in combo:
                _, model = combo.split("_", 1)
                if model in dashboard["models_progress"]:
                    dashboard["models_progress"][model] += 1

    # Start processing
    results = []
    logger.info(
        f"Processing {len(all_scenarios)} scenarios with {len(models)} models and {len(frameworks)} frameworks"
    )

    # Determine parallel execution mode
    num_workers = 1
    if args and args.parallel and args.parallel > 1:
        num_workers = min(args.parallel, 8)  # Cap at 8 to avoid overwhelming the system
        logger.info(f"Using {num_workers} parallel workers for processing")

    if num_workers > 1:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = []
            for scenario in all_scenarios:
                for model in models:
                    # Skip already completed combinations
                    if should_skip_combination(scenario.id, model, completed_combinations):
                        logger.debug(
                            f"Skipping already processed scenario {scenario.id} with model {model}"
                        )
                        continue

                    futures.append(
                        executor.submit(
                            process_scenario_parallel,
                            scenario,
                            model,
                            frameworks,
                            client,
                            save_dir,
                            completed_combinations,
                            dashboard,
                        )
                    )

            # Process results as they complete
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel execution: {e}")
    else:
        # Sequential execution
        scenario_count = 0
        for scenario in all_scenarios:
            scenario_count += 1
            logger.info(f"Processing scenario {scenario_count}/{len(all_scenarios)}: {scenario.id}")

            # Process each model
            for model in models:
                # Skip if already processed
                if should_skip_combination(scenario.id, model, completed_combinations):
                    logger.info(
                        f"Skipping already processed scenario {scenario.id} with model {model}"
                    )
                    continue

                logger.info(f"  Using model: {model}")

                # Process scenario with all frameworks
                try:
                    result = client.process_scenario(scenario, model, frameworks)
                    results.append(result)

                    # Save raw response data
                    response_path = save_dir / "raw_responses" / f"{scenario.id}_{model}.json"
                    with open(response_path, "w") as f:
                        json.dump(result, f, indent=2)

                    # Update progress dashboard
                    update_progress(dashboard, scenario.id, model, save_dir)

                except Exception as e:
                    logger.error(f"Error processing scenario {scenario.id} with model {model}: {e}")

    logger.info(f"Completed processing {len(results)} scenario-model combinations")

    # Handle case where we only processed part of the data due to resuming
    # We need to load any already processed results that weren't re-processed
    if completed_combinations:
        logger.info("Loading previously processed results from files")
        for combination in completed_combinations:
            if "_" in combination:
                scenario_id, model = combination.split("_", 1)

                # Skip if we already have this result in memory
                if any(
                    r.get("scenario_id") == scenario_id and r.get("model") == model for r in results
                ):
                    continue

                # Load from file
                response_path = save_dir / "raw_responses" / f"{scenario_id}_{model}.json"
                if response_path.exists():
                    try:
                        with open(response_path) as f:
                            result = json.load(f)
                        results.append(result)
                        logger.debug(f"Loaded previous result for {scenario_id} with {model}")
                    except Exception as e:
                        logger.error(
                            f"Error loading previous result for {scenario_id} with {model}: {e}"
                        )

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
