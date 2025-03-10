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

    # Load configuration - try main config file first, fall back to simplified
    config_path = results_path / "config.json"
    config_simplified_path = results_path / "config_simplified.json"

    config = None

    # Try loading the main config file
    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path) as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Error decoding {config_path}: {e}")
            logger.info("Will try to load from simplified config instead")

    # If main config failed or doesn't exist, try simplified config
    if config is None and config_simplified_path.exists():
        logger.info(f"Loading configuration from {config_simplified_path}")
        try:
            with open(config_simplified_path) as f:
                config_str = json.load(f)

            # Convert string representations back to objects
            config = {}
            for key, value in config_str.items():
                if isinstance(value, str) and (value.startswith("[") or value.startswith("{")):
                    try:
                        # Replace single quotes with double quotes for proper JSON parsing
                        json_value = value.replace("'", '"')
                        config[key] = json.loads(json_value)
                    except:
                        # If parsing fails, keep the original string
                        config[key] = value
                else:
                    config[key] = value
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding {config_simplified_path}: {e}")
            raise

    if config is None:
        logger.error(f"No usable configuration found in {results_dir}")
        raise FileNotFoundError(f"Configuration files not found or corrupt in: {results_dir}")

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

    # 1. Create fluctuation rate bar chart (original plot)
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

    # 2. Create censorship rate plot (original plot)
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

    # NEW VISUALIZATIONS

    # 3. Framework influence heatmap
    logger.info("Generating framework influence heatmap")
    plt.figure(figsize=(12, 8))
    # Create pivot table with models as rows and frameworks as columns
    pivot = pd.pivot_table(
        framework_df, values="stance_changed", index="model", columns="framework", aggfunc="mean"
    )
    # Plot heatmap
    ax = sns.heatmap(pivot, annot=True, cmap="viridis", vmin=0, vmax=1, fmt=".2f")
    plt.title("Framework Influence on Model Responses")
    plt.tight_layout()
    plt.savefig(plots_dir / "framework_influence_heatmap.png", dpi=300)
    plt.close()

    # 4. Response length analysis
    logger.info("Generating response length comparison")
    plt.figure(figsize=(14, 8))

    # Check if full_response column exists (which contains complete responses)
    if "full_response" in df.columns:
        logger.info("Computing response lengths using full_response column")

        # Inspect a sample of the full_response to debug
        sample_response = df["full_response"].iloc[0]
        logger.info(f"Sample full response first 100 chars: {str(sample_response)[:100]}")
        logger.info(f"Sample full response type: {type(sample_response)}")

        # Add response length columns - using the full response text
        df["response_length"] = df["full_response"].astype(str).str.len()
        logger.info(
            f"Calculated response lengths. Min: {df['response_length'].min()}, Max: {df['response_length'].max()}, Mean: {df['response_length'].mean():.2f}"
        )

        # Check if the deepseek model is present (which has thinking sections)
        deepseek_present = any("deepseek" in model.lower() for model in df["model"].unique())
        if deepseek_present:
            # Extract thinking sections and their lengths
            def extract_thinking(text):
                import re

                thinking_pattern = r"<think>(.*?)</think>"
                thinking_matches = re.findall(thinking_pattern, str(text), re.DOTALL)
                if thinking_matches:
                    return thinking_matches[0].strip()
                return ""

            def extract_final_answer(text):
                import re

                # Get everything after the last </think> tag
                parts = re.split(r"</think>", str(text), flags=re.DOTALL)
                if len(parts) > 1:
                    return parts[-1].strip()
                return str(text).strip()

            logger.info("Extracting thinking sections for deepseek model")
            # Only apply to deepseek models
            thinking_lengths = []
            final_lengths = []

            for idx, row in df[df["model"].str.contains("deepseek")].iterrows():
                text = str(row["full_response"])  # Use full_response instead of response_text
                thinking_text = extract_thinking(text)
                final_answer = extract_final_answer(text)

                df.at[idx, "thinking_text"] = thinking_text
                df.at[idx, "final_answer"] = final_answer

                thinking_lengths.append(len(thinking_text))
                final_lengths.append(len(final_answer))

            if thinking_lengths:
                logger.info(
                    f"Thinking section stats - Min: {min(thinking_lengths)}, Max: {max(thinking_lengths)}, Mean: {sum(thinking_lengths) / len(thinking_lengths):.2f}"
                )
            if final_lengths:
                logger.info(
                    f"Final answer stats - Min: {min(final_lengths)}, Max: {max(final_lengths)}, Mean: {sum(final_lengths) / len(final_lengths):.2f}"
                )

            # Convert thinking and final answer sections to Series before using str accessor
            if df.get("thinking_text") is not None:
                df["thinking_length"] = pd.Series(df["thinking_text"]).str.len()
            if df.get("final_answer") is not None:
                df["final_answer_length"] = pd.Series(df["final_answer"]).str.len()

        # Create boxplot of response lengths by framework and model
        if (
            pd.to_numeric(df["response_length"], errors="coerce").var() > 0
        ):  # Safely check for variance
            sns.boxplot(x="framework", y="response_length", hue="model", data=df)
            plt.title("Response Length by Framework and Model")
            plt.ylabel("Character Count")
            plt.xlabel("Framework")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / "response_length_comparison.png", dpi=300)
            plt.close()

            # 5. Response length distribution by model
            logger.info("Generating response length distribution by model")
            plt.figure(figsize=(12, 6))
            for model in df["model"].unique():
                model_data = pd.to_numeric(
                    df[df["model"] == model]["response_length"], errors="coerce"
                )
                if model_data.var() > 0:  # Only plot if there's variance
                    sns.kdeplot(model_data, label=model)
            plt.title("Response Length Distribution by Model (Including Thinking Sections)")
            plt.xlabel("Character Count")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "response_length_distribution.png", dpi=300)
            plt.close()

            # Create a fair comparison that only includes final answers for both models
            logger.info("Generating fair comparison of final answer lengths")

            # Create a new column for final answer length for all models
            # For deepseek, use the extracted final answer
            # For other models, use the full response (since they don't have thinking sections)
            df["final_answer_only"] = df["response_length"]  # Default to full response length

            # If there's a deepseek model and we extracted final answers
            if deepseek_present and "final_answer_length" in df.columns:
                # Update deepseek rows with just their final answer length
                deepseek_mask = df["model"].str.contains("deepseek")
                df.loc[deepseek_mask, "final_answer_only"] = df.loc[
                    deepseek_mask, "final_answer_length"
                ]

                # Plot fair comparison - distribution
                plt.figure(figsize=(12, 6))
                for model in df["model"].unique():
                    model_data = pd.to_numeric(
                        df[df["model"] == model]["final_answer_only"], errors="coerce"
                    )
                    if model_data.var() > 0:
                        sns.kdeplot(model_data, label=model)
                plt.title("Final Answer Length Distribution by Model (Fair Comparison)")
                plt.xlabel("Character Count")
                plt.ylabel("Density")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / "final_answer_distribution.png", dpi=300)
                plt.close()

                # Plot fair comparison - boxplot
                plt.figure(figsize=(14, 8))
                sns.boxplot(x="framework", y="final_answer_only", hue="model", data=df)
                plt.title("Final Answer Length by Framework and Model (Fair Comparison)")
                plt.ylabel("Character Count")
                plt.xlabel("Framework")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(plots_dir / "final_answer_comparison.png", dpi=300)
                plt.close()

                # Plot fair comparison - average by framework and model
                plt.figure(figsize=(12, 8))
                avg_by_model_framework = (
                    df.groupby(["model", "framework"])["final_answer_only"].mean().unstack()
                )
                ax = avg_by_model_framework.plot(kind="bar", figsize=(12, 8))
                plt.title("Average Final Answer Length by Model and Framework")
                plt.ylabel("Character Count")
                plt.xlabel("Model")
                plt.xticks(rotation=45)
                plt.legend(title="Framework")
                plt.grid(axis="y", linestyle="--", alpha=0.7)

                # Add value labels
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.0f", fontweight="bold")

                plt.savefig(plots_dir / "avg_final_answer_by_model_framework.png", dpi=300)
                plt.close()

                # Plot deepseek summary with breakdown
                plt.figure(figsize=(10, 6))
                deepseek_subset = df[df["model"].str.contains("deepseek")]
                deepseek_df_avg = deepseek_subset.groupby("framework")[
                    ["thinking_length", "final_answer_length"]
                ].mean()
                deepseek_df_avg.plot(kind="bar", stacked=True)
                plt.title("DeepSeek Response Breakdown by Framework")
                plt.xlabel("Framework")
                plt.ylabel("Average Character Count")
                plt.legend(["Thinking Process", "Final Answer"])
                plt.xticks(rotation=45)
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.savefig(plots_dir / "deepseek_response_breakdown.png", dpi=300)
                plt.close()

            # Framework thinking effort visualization
            if (
                deepseek_present
                and "thinking_length" in df.columns
                and "final_answer_length" in df.columns
            ):
                logger.info("Generating thinking-to-answer ratio visualization")

                # Get deepseek subset
                deepseek_df = df[df["model"].str.contains("deepseek")].copy()

                # Calculate thinking effort ratio (thinking / final answer)
                deepseek_df["thinking_effort_ratio"] = (
                    deepseek_df["thinking_length"] / deepseek_df["final_answer_length"]
                )

                plt.figure(figsize=(12, 6))
                effort_ratio = (
                    deepseek_df.groupby("framework")["thinking_effort_ratio"]
                    .mean()
                    .sort_values(ascending=False)
                )
                ax = effort_ratio.plot(
                    kind="bar", color=sns.color_palette("viridis", len(effort_ratio))
                )
                plt.title("DeepSeek Thinking Effort Ratio by Framework")
                plt.ylabel("Thinking Length / Final Answer Length")
                plt.xlabel("Framework")
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.tight_layout()

                # Add value labels
                for i, v in enumerate(effort_ratio):
                    ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontweight="bold")

                plt.savefig(plots_dir / "deepseek_thinking_effort_ratio.png", dpi=300)
                plt.close()
        else:
            logger.warning(
                "No variance in response lengths. Check if responses are being truncated."
            )

        # 5b. Response length by framework (bar chart)
        logger.info("Generating average response length by framework")
        plt.figure(figsize=(12, 6))
        framework_lengths = (
            df.groupby("framework")["response_length"].mean().sort_values(ascending=False)
        )
        ax = framework_lengths.plot(
            kind="bar", color=sns.color_palette("viridis", len(framework_lengths))
        )
        plt.title("Average Response Length by Framework")
        plt.ylabel("Character Count")
        plt.xlabel("Framework")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Add value labels
        for i, v in enumerate(framework_lengths):
            ax.text(
                i, v + framework_lengths.max() * 0.02, f"{int(v)}", ha="center", fontweight="bold"
            )

        plt.savefig(plots_dir / "avg_response_length_by_framework.png", dpi=300)
        plt.close()

        # If we have deepseek model with thinking sections
        if deepseek_present and "thinking_length" in df.columns:
            logger.info("Generating deepseek thinking analysis visualizations")

            # Filter for deepseek models only
            deepseek_df = df[df["model"].str.contains("deepseek")].copy()

            # 5c. Thinking vs final answer length
            plt.figure(figsize=(12, 6))
            thinking_vs_final = deepseek_df.groupby("framework")[
                ["thinking_length", "final_answer_length"]
            ].mean()
            thinking_vs_final.plot(kind="bar", figsize=(12, 6))
            plt.title("Deepseek Model: Thinking vs Final Answer Length by Framework")
            plt.ylabel("Character Count")
            plt.xlabel("Framework")
            plt.xticks(rotation=45)
            plt.legend(["Thinking Section", "Final Answer"])
            plt.tight_layout()
            plt.savefig(plots_dir / "deepseek_thinking_vs_final.png", dpi=300)
            plt.close()

            # 5d. Thinking to total ratio by framework
            plt.figure(figsize=(12, 6))
            deepseek_df["thinking_ratio"] = (
                deepseek_df["thinking_length"] / deepseek_df["response_length"]
            )
            thinking_ratio = (
                deepseek_df.groupby("framework")["thinking_ratio"]
                .mean()
                .sort_values(ascending=False)
            )
            ax = thinking_ratio.plot(
                kind="bar", color=sns.color_palette("rocket", len(thinking_ratio))
            )
            plt.title("Deepseek Model: Thinking Section Ratio by Framework")
            plt.ylabel("Thinking Section / Total Response Ratio")
            plt.xlabel("Framework")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Add value labels
            for i, v in enumerate(thinking_ratio):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

            plt.savefig(plots_dir / "deepseek_thinking_ratio.png", dpi=300)
            plt.close()

    elif "response_text" in df.columns:
        logger.info("Computing response lengths from response_text column (note: may be truncated)")
        # Add response length columns
        df["response_length"] = df["response_text"].astype(str).str.len()
        logger.info(
            f"Calculated response lengths. Min: {df['response_length'].min()}, Max: {df['response_length'].max()}, Mean: {df['response_length'].mean():.2f}"
        )
        logger.warning(
            "Using response_text column which may be truncated. Consider using full_response if available."
        )

        # Continue with the rest of the visualizations as before...
        sns.boxplot(x="framework", y="response_length", hue="model", data=df)
        plt.title("Response Length by Framework and Model (potentially truncated)")
        plt.ylabel("Character Count")
        plt.xlabel("Framework")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / "response_length_comparison.png", dpi=300)
        plt.close()

    else:
        logger.warning(
            "Neither 'full_response' nor 'response_text' column found. Skipping response length visualizations."
        )

    # 6. Scenario difficulty analysis
    logger.info("Generating scenario difficulty analysis")
    plt.figure(figsize=(12, 8))
    # Group by scenario_id and calculate stance_changed mean
    if "scenario_id" in framework_df.columns:
        scenario_difficulty = (
            framework_df.groupby("scenario_id")["stance_changed"]
            .mean()
            .sort_values(ascending=False)
        )
        # Plot top 10 scenarios with highest fluctuation rate
        top_scenarios = scenario_difficulty.head(10)
        ax = top_scenarios.plot(kind="bar", color="teal")
        plt.title("Top 10 Scenarios by Fluctuation Rate")
        plt.ylabel("Stance Change Rate")
        plt.xlabel("Scenario ID")
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Add value labels
        for i, v in enumerate(top_scenarios):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.savefig(plots_dir / "scenario_difficulty.png", dpi=300)
        plt.close()

    # 7. Framework effectiveness by dataset
    logger.info("Generating framework effectiveness by dataset")
    if "dataset" in framework_df.columns:
        plt.figure(figsize=(14, 8))
        # Create pivot table with datasets as rows and frameworks as columns
        dataset_pivot = pd.pivot_table(
            framework_df,
            values="stance_changed",
            index="dataset",
            columns="framework",
            aggfunc="mean",
        )
        # Plot heatmap
        sns.heatmap(dataset_pivot, annot=True, cmap="coolwarm", vmin=0, vmax=1, fmt=".2f")
        plt.title("Framework Effectiveness by Dataset")
        plt.tight_layout()
        plt.savefig(plots_dir / "framework_by_dataset.png", dpi=300)
        plt.close()

    # 8. Framework comparison across models - ENHANCED to make it more prominent
    logger.info("Generating framework comparison bar chart")
    plt.figure(figsize=(14, 8))
    framework_effectiveness = (
        framework_df.groupby("framework")["stance_changed"].mean().sort_values(ascending=False)
    )

    # Create bar chart with enhanced colors
    bars = plt.bar(
        framework_effectiveness.index,
        framework_effectiveness.values,
        color=sns.color_palette("viridis", len(framework_effectiveness)),
    )

    plt.title("Effectiveness of Different Ethical Frameworks", fontsize=18, fontweight="bold")
    plt.xlabel("Framework", fontsize=14)
    plt.ylabel("Stance Change Rate", fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(plots_dir / "framework_comparison.png", dpi=300)

    # Also create a horizontal version for better readability of framework names
    plt.figure(figsize=(14, 8))
    framework_effectiveness.plot(
        kind="barh", color=sns.color_palette("viridis", len(framework_effectiveness))
    )
    plt.title("Effectiveness of Different Ethical Frameworks", fontsize=18, fontweight="bold")
    plt.xlabel("Stance Change Rate", fontsize=14)
    plt.ylabel("Framework", fontsize=14)
    plt.xlim(0, 1)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / "framework_comparison_horizontal.png", dpi=300)
    plt.close()

    # 9. Baseline vs Framework Response Comparison
    logger.info("Generating baseline vs framework comparison")
    # Calculate the average stance change rate when going from baseline to each framework
    if all(col in df.columns for col in ["scenario_id", "model", "framework"]):
        baseline_df = df[df["framework"] == "baseline"].copy()
        baseline_df["baseline_censored"] = baseline_df["censored"]

        # Create a dataframe for comparison
        compare_df = framework_df.merge(
            baseline_df[["scenario_id", "model", "baseline_censored"]],
            on=["scenario_id", "model"],
            how="left",
        )

        # See if censorship changes between baseline and framework
        if "censored" in compare_df.columns and "baseline_censored" in compare_df.columns:
            compare_df["censorship_changed"] = (
                compare_df["censored"] != compare_df["baseline_censored"]
            )

            # Plot censorship change by framework
            plt.figure(figsize=(12, 6))
            censorship_change = compare_df.groupby("framework")["censorship_changed"].mean()
            ax = censorship_change.plot(kind="bar", color="crimson")
            plt.title("Censorship Change Rate by Framework")
            plt.xlabel("Framework")
            plt.ylabel("Rate of Censorship Change")
            plt.ylim(0, 1)
            plt.tight_layout()

            # Add value labels
            for i, v in enumerate(censorship_change):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

            plt.savefig(plots_dir / "censorship_change_by_framework.png", dpi=300)
            plt.close()

    logger.info("All visualizations saved to results/plots")


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
