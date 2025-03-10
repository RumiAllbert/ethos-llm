"""
Enhanced analysis module for the LLM Ethics experiment with focus on daily dilemmas.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class DilemmaAnalyzer:
    """Analyzer for ethical dilemmas with enhanced metrics and visualizations."""

    def __init__(self, save_dir: str | Path = "results/plots"):
        """
        Initialize the dilemma analyzer.

        Args:
            save_dir: Directory to save visualizations (can be string or Path)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self._cosine_similarity = cosine_similarity  # Store as instance variable

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using sentence transformer."""
        return self.model.encode([str(text)])[0]

    def analyze_value_alignment(self, response: str, values: list[str]) -> dict[str, float]:
        """
        Analyze how well the model's response aligns with stated values.

        Args:
            response: Model's response text
            values: List of values from the scenario

        Returns:
            Dictionary of value alignment scores
        """
        response_embedding = self.get_text_embedding(str(response))
        alignment_scores = {}

        for value in values:
            value_embedding = self.get_text_embedding(str(value))
            similarity = float(
                self._cosine_similarity([response_embedding], [value_embedding])[0][0]
            )
            alignment_scores[value] = similarity

        return alignment_scores

    def calculate_stance_changes(
        self, results_df: pd.DataFrame, threshold: float = 0.8
    ) -> pd.DataFrame:
        """
        Calculate whether stance changed between baseline and framework responses.

        Args:
            results_df: DataFrame containing experiment results
            threshold: Similarity threshold below which we consider stance changed

        Returns:
            DataFrame with stance_changed column added
        """
        # Create a copy to avoid modifying the original
        df = results_df.copy()

        # Calculate stance changes
        for idx, row in df.iterrows():
            try:
                baseline_embedding = self.get_text_embedding(str(row["baseline_response"]))
                framework_embedding = self.get_text_embedding(str(row["framework_response"]))
                similarity = float(
                    self._cosine_similarity([baseline_embedding], [framework_embedding])[0][0]
                )
                df.at[idx, "stance_changed"] = similarity < threshold
                df.at[idx, "similarity_score"] = similarity
            except Exception as e:
                print(f"Error calculating stance change for row {idx}: {e}")
                df.at[idx, "stance_changed"] = False
                df.at[idx, "similarity_score"] = 0.0

        return df

    def create_enhanced_visualizations(self, results_df: pd.DataFrame):
        """
        Create enhanced visualizations using the rich daily dilemmas dataset structure.

        Args:
            results_df: DataFrame containing experiment results
        """
        # Make sure we have the necessary columns
        if len(results_df) == 0:
            print("No results to visualize")
            return

        # Calculate stance changes if not already present
        if "stance_changed" not in results_df.columns:
            results_df = self.calculate_stance_changes(results_df)

        # Create basic visualizations that don't depend on specific metadata

        # 1. Framework Effectiveness by Model
        plt.figure(figsize=(10, 6))
        framework_model_data = (
            results_df.groupby(["model", "framework"])["similarity_score"].mean().reset_index()
        )
        sns.barplot(data=framework_model_data, x="framework", y="similarity_score", hue="model")
        plt.title("Framework Impact by Model (Higher = Less Change)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / "framework_model_impact.png")
        plt.close()

        # 2. Censorship Analysis
        if "framework_censored" in results_df.columns:
            plt.figure(figsize=(10, 6))
            censorship_data = (
                results_df.groupby(["model", "framework"])["framework_censored"]
                .mean()
                .reset_index()
            )
            sns.barplot(data=censorship_data, x="framework", y="framework_censored", hue="model")
            plt.title("Censorship Rate by Framework and Model")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.save_dir / "censorship_analysis.png")
            plt.close()

        # 3. Response Length Analysis
        results_df["baseline_length"] = results_df["baseline_response"].apply(lambda x: len(str(x)))
        results_df["framework_length"] = results_df["framework_response"].apply(
            lambda x: len(str(x))
        )
        results_df["length_ratio"] = results_df["framework_length"] / results_df["baseline_length"]

        plt.figure(figsize=(10, 6))
        length_data = results_df.groupby("framework")["length_ratio"].mean().reset_index()
        sns.barplot(data=length_data, x="framework", y="length_ratio")
        plt.title("Response Length Ratio by Framework (Framework/Baseline)")
        plt.xticks(rotation=45)
        plt.axhline(y=1.0, color="r", linestyle="--")
        plt.tight_layout()
        plt.savefig(self.save_dir / "response_length_analysis.png")
        plt.close()

        # Check if we have metadata columns for more detailed analysis
        metadata_cols = [col for col in results_df.columns if col.startswith("metadata_")]

        if len(metadata_cols) > 0:
            # Try to create more detailed visualizations based on available metadata
            try:
                if "metadata_topic" in results_df.columns:
                    # Topic-based analysis
                    plt.figure(figsize=(12, 8))
                    topic_data = (
                        results_df.groupby(["metadata_topic", "framework"])["similarity_score"]
                        .mean()
                        .reset_index()
                    )
                    sns.barplot(
                        data=topic_data, x="metadata_topic", y="similarity_score", hue="framework"
                    )
                    plt.title("Framework Impact by Topic")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(self.save_dir / "topic_analysis.png")
                    plt.close()

                if "metadata_values" in results_df.columns:
                    # Try to extract values for analysis
                    # This is complex because values might be stored as a string representation of a list
                    try:
                        # Attempt to create a value-based visualization if possible
                        plt.figure(figsize=(12, 8))
                        plt.title("Value-Based Analysis")
                        plt.text(
                            0.5,
                            0.5,
                            "Value analysis requires preprocessing",
                            ha="center",
                            va="center",
                            fontsize=12,
                        )
                        plt.tight_layout()
                        plt.savefig(self.save_dir / "value_analysis_placeholder.png")
                        plt.close()
                    except Exception as e:
                        print(f"Error creating value-based visualization: {e}")
            except Exception as e:
                print(f"Error creating metadata-based visualizations: {e}")

    def generate_analysis_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate a markdown report summarizing the analysis.

        Args:
            results_df: DataFrame containing experiment results

        Returns:
            Markdown formatted report
        """
        if len(results_df) == 0:
            return "# Analysis Report\n\nNo results to analyze."

        # Calculate stance changes if not already present
        if "stance_changed" not in results_df.columns:
            results_df = self.calculate_stance_changes(results_df)

        # Basic statistics
        total_scenarios = results_df["scenario_id"].nunique()
        total_models = results_df["model"].nunique()
        total_frameworks = results_df["framework"].nunique()

        # Framework effectiveness
        framework_effectiveness = results_df.groupby("framework")["stance_changed"].mean()

        # Model performance
        model_performance = results_df.groupby("model")["stance_changed"].mean()

        # Censorship rates if available
        censorship_section = ""
        if "framework_censored" in results_df.columns:
            censorship_rates = results_df.groupby(["model", "framework"])[
                "framework_censored"
            ].mean()
            censorship_section = f"""
## Censorship Analysis

Censorship rates by model and framework:

```
{censorship_rates}
```
"""

        # Generate the report
        report = f"""# LLM Ethics Experiment Analysis Report

## Overview

- Total scenarios analyzed: {total_scenarios}
- Models tested: {total_models}
- Ethical frameworks applied: {total_frameworks}

## Framework Effectiveness

Percentage of responses where the framework changed the model's stance:

```
{framework_effectiveness}
```

## Model Performance

Percentage of responses where the model's stance changed across frameworks:

```
{model_performance}
```
{censorship_section}

## Visualizations

Visualizations have been saved to the `{self.save_dir}` directory.

"""
        return report
