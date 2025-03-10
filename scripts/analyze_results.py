#!/usr/bin/env python
"""Script for analyzing and visualizing results from the LLM Ethics Experiment.

This script loads the results from a previous experiment run and generates
visualizations and additional analyses.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

# Add the project root to the Python path
import fix_path  # noqa
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.logging import get_experiment_logger, setup_logging

# Get logger for this module
logger = get_experiment_logger("analyze")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Analyze results from the LLM Ethics Experiment")
    parser.add_argument(
        "--results-dir",
        "-r",
        type=str,
        required=True,
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Directory to save analysis outputs (defaults to results-dir)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def load_results(results_dir: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load results from a previous experiment run.

    Args:
        results_dir: Directory containing experiment results

    Returns:
        Tuple of (config dict, results DataFrame)
    """
    logger = get_experiment_logger("analysis")
    results_path = Path(results_dir)

    # Load configuration
    config_path = results_path / "config.json"
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    # Load results DataFrame
    csv_path = results_path / "analysis" / "responses.csv"
    if not csv_path.exists():
        logger.error(f"Results CSV file not found: {csv_path}")
        raise FileNotFoundError(f"Results CSV file not found: {csv_path}")

    logger.info(f"Loading results from {csv_path}")
    df = pd.read_csv(csv_path)

    logger.info(f"Loaded {len(df)} response entries")

    return config, df


def generate_visualizations(df: pd.DataFrame, config: dict[str, Any], output_dir: str) -> None:
    """Generate visualizations from experiment results.

    Args:
        df: DataFrame with experiment results
        config: Experiment configuration
        output_dir: Directory to save visualizations
    """
    logger = get_experiment_logger("visualization")
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Generating visualizations in {plots_dir}")

    # Set plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 8),
            "figure.autolayout": True,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
        }
    )

    # Filter out baseline responses for framework-specific plots
    framework_df = df[df["framework"] != "baseline"]

    # Various visualization functions would be called here
    # For now, we'll just create a few basic ones

    # 1. Create fluctuation rate bar chart
    logger.info("Generating fluctuation rate plot")
    plt.figure(figsize=(10, 6))
    fluctuation_by_model = framework_df.groupby("model")["stance_changed"].mean()
    ax = fluctuation_by_model.plot(kind="bar", color="skyblue")
    plt.title("Fluctuation Rate by Model")
    plt.xlabel("Model")
    plt.ylabel("Fluctuation Rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add value labels on top of bars
    for i, v in enumerate(fluctuation_by_model):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

    plt.savefig(plots_dir / "fluctuation_by_model.png", dpi=300)
    plt.close()

    # 2. Create censorship rate plot
    logger.info("Generating censorship rate plot")
    plt.figure(figsize=(10, 6))
    censorship_by_model = df.groupby("model")["censored"].mean()
    ax = censorship_by_model.plot(kind="bar", color="salmon")
    plt.title("Censorship Rate by Model")
    plt.xlabel("Model")
    plt.ylabel("Censorship Rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add value labels on top of bars
    for i, v in enumerate(censorship_by_model):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

    plt.savefig(plots_dir / "censorship_by_model.png", dpi=300)
    plt.close()

    logger.info(f"All visualizations saved to {plots_dir}")


def main(args: argparse.Namespace) -> None:
    """Main function for analyzing experiment results.

    Args:
        args: Command line arguments
    """
    logger = get_experiment_logger("analyze")
    logger.info("Starting results analysis")

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir

    # Load results
    config, df = load_results(results_dir)

    # Generate visualizations
    generate_visualizations(df, config, output_dir)

    logger.info("Analysis completed successfully!")

    # Print summary information
    print("\nAnalysis Summary:")
    print(f"- Total responses analyzed: {len(df)}")
    print(f"- Models: {', '.join(df['model'].unique())}")
    print(f"- Frameworks: {', '.join(df['framework'].unique())}")
    print(f"- Generated visualizations in: {Path(output_dir) / 'plots'}")

    # Overall metrics
    framework_df = df[df["framework"] != "baseline"]
    overall_fluctuation = framework_df["stance_changed"].mean()
    overall_censorship = df["censored"].mean()
    baseline_censorship = df[df["framework"] == "baseline"]["censored"].mean()
    framework_censorship = framework_df["censored"].mean()

    print("\nOverall Metrics:")
    print(f"- Fluctuation Rate: {overall_fluctuation:.2f}")
    print(f"- Overall Censorship Rate: {overall_censorship:.2f}")
    print(f"- Baseline Censorship Rate: {baseline_censorship:.2f}")
    print(f"- Framework Censorship Rate: {framework_censorship:.2f}")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level)

    # Run analysis
    main(args)
