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

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.analysis.visualization import (
    create_metrics_dashboard,
    plot_censorship_analysis,
    plot_comparative_fluctuation,
    plot_response_trajectories,
    plot_statistical_significance,
)
from src.utils.logging import get_experiment_logger, setup_logging


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

    # 1. Plot fluctuation rates by model and framework
    logger.info("Generating fluctuation rate plots")

    # Overall fluctuation rate by model
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

    # Fluctuation rate by model and framework
    plt.figure(figsize=(12, 8))
    fluctuation_pivot = framework_df.pivot_table(
        index="model", columns="framework", values="stance_changed", aggfunc="mean"
    )
    ax = fluctuation_pivot.plot(kind="bar", colormap="viridis")
    plt.title("Fluctuation Rate by Model and Framework")
    plt.xlabel("Model")
    plt.ylabel("Fluctuation Rate")
    plt.ylim(0, 1)
    plt.legend(title="Framework")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / "fluctuation_by_model_framework.png", dpi=300)
    plt.close()

    # 2. Plot censorship rates by model and framework
    logger.info("Generating censorship rate plots")

    # Overall censorship rate by model
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

    # Censorship rate by model and framework
    plt.figure(figsize=(12, 8))
    censorship_pivot = df.pivot_table(
        index="model", columns="framework", values="censored", aggfunc="mean"
    )
    ax = censorship_pivot.plot(kind="bar", colormap="Reds")
    plt.title("Censorship Rate by Model and Framework")
    plt.xlabel("Model")
    plt.ylabel("Censorship Rate")
    plt.ylim(0, 1)
    plt.legend(title="Framework")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / "censorship_by_model_framework.png", dpi=300)
    plt.close()

    # 3. Plot similarity distribution
    logger.info("Generating similarity distribution plots")

    plt.figure(figsize=(12, 8))
    for model in df["model"].unique():
        model_df = framework_df[framework_df["model"] == model]
        sns.kdeplot(data=model_df, x="similarity_to_baseline", hue="framework", label=model)

    plt.title("Distribution of Similarity Scores by Framework")
    plt.xlabel("Similarity to Baseline Response")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(plots_dir / "similarity_distribution.png", dpi=300)
    plt.close()

    # 4. Plot framework influence (heatmap of similarity scores)
    logger.info("Generating framework influence heatmap")

    plt.figure(figsize=(10, 8))
    similarity_pivot = framework_df.pivot_table(
        index="model", columns="framework", values="similarity_to_baseline", aggfunc="mean"
    )
    sns.heatmap(similarity_pivot, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f")
    plt.title("Framework Influence (Avg. Similarity to Baseline)")
    plt.tight_layout()
    plt.savefig(plots_dir / "framework_influence_heatmap.png", dpi=300)
    plt.close()

    # 5. Comparison of frameworks effect across models
    logger.info("Generating framework comparison plots")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Fluctuation by framework (aggregated across models)
    framework_fluctuation = framework_df.groupby("framework")["stance_changed"].mean()
    ax = framework_fluctuation.plot(kind="bar", color="skyblue", ax=axes[0])
    axes[0].set_title("Fluctuation Rate by Framework")
    axes[0].set_xlabel("Framework")
    axes[0].set_ylabel("Fluctuation Rate")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=45)

    # Add value labels
    for i, v in enumerate(framework_fluctuation):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center")

    # Censorship by framework (aggregated across models)
    framework_censorship = framework_df.groupby("framework")["censored"].mean()
    ax = framework_censorship.plot(kind="bar", color="salmon", ax=axes[1])
    axes[1].set_title("Censorship Rate by Framework")
    axes[1].set_xlabel("Framework")
    axes[1].set_ylabel("Censorship Rate")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=45)

    # Add value labels
    for i, v in enumerate(framework_censorship):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(plots_dir / "framework_comparison.png", dpi=300)
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

    # Generate existing visualizations
    generate_visualizations(df, config, output_dir)

    # Generate enhanced visualizations
    plots_dir = Path(output_dir) / "plots"

    # Create comprehensive dashboard
    create_metrics_dashboard(df, str(plots_dir))

    # Create statistical significance visualizations
    plot_statistical_significance(df, str(plots_dir))

    # Create response trajectory visualization
    plot_response_trajectories(df, str(plots_dir))

    # Create comparative fluctuation analysis
    plot_comparative_fluctuation(df, str(plots_dir))

    # Create censorship analysis visualization
    plot_censorship_analysis(df, str(plots_dir))

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
