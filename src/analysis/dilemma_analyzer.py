"""
Enhanced analysis module for the LLM Ethics experiment with focus on daily dilemmas.
"""

from collections import defaultdict
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
            alignment_scores[value] = float(
                self._cosine_similarity([response_embedding], [value_embedding])[0][0]
            )

        return alignment_scores

    def calculate_enhanced_metrics(self, results_df: pd.DataFrame) -> dict[str, dict]:
        """
        Calculate enhanced metrics using the rich daily dilemmas dataset structure.

        Args:
            results_df: DataFrame containing experiment results

        Returns:
            Dictionary of enhanced metrics
        """
        metrics = {
            "value_based_metrics": defaultdict(dict),
            "topic_based_metrics": defaultdict(dict),
            "action_type_metrics": defaultdict(dict),
        }

        # Value-based metrics
        for value in results_df["values"].explode().unique():
            value_responses = results_df[results_df["values"].apply(lambda x: value in x)]
            metrics["value_based_metrics"]["alignment_scores"][value] = value_responses[
                "value_alignment"
            ].mean()
            metrics["value_based_metrics"]["framework_effectiveness"][value] = value_responses[
                "stance_changed"
            ].mean()

        # Topic-based metrics
        for topic in results_df["topic"].unique():
            topic_responses = results_df[results_df["topic"] == topic]
            metrics["topic_based_metrics"]["fluctuation"][topic] = topic_responses[
                "stance_changed"
            ].mean()
            metrics["topic_based_metrics"]["censorship"][topic] = topic_responses["censored"].mean()

        # Action-type metrics
        for action_type in results_df["action_type"].unique():
            action_responses = results_df[results_df["action_type"] == action_type]
            metrics["action_type_metrics"]["framework_preference"][action_type] = (
                action_responses.groupby("framework")["stance_changed"].mean().to_dict()
            )

        return metrics

    def create_enhanced_visualizations(self, results_df: pd.DataFrame):
        """
        Create enhanced visualizations using the rich daily dilemmas dataset structure.

        Args:
            results_df: DataFrame containing experiment results
        """
        # 1. Value Alignment Analysis
        plt.figure(figsize=(15, 8))
        value_effectiveness = pd.DataFrame(results_df.groupby("values")["value_alignment"].mean())
        sns.barplot(data=value_effectiveness.reset_index(), x="values", y="value_alignment")
        plt.title("Value Alignment Scores by Ethical Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / "value_alignment.png")
        plt.close()

        # 2. Topic Response Patterns
        plt.figure(figsize=(12, 8))
        topic_patterns = pd.pivot_table(
            results_df, values="stance_changed", index="topic", columns="framework", aggfunc="mean"
        )
        sns.heatmap(topic_patterns, annot=True, cmap="YlOrRd", fmt=".2f")
        plt.title("Framework Effectiveness by Topic")
        plt.tight_layout()
        plt.savefig(self.save_dir / "topic_framework_patterns.png")
        plt.close()

        # 3. Action Type Analysis
        plt.figure(figsize=(12, 6))
        action_framework_patterns = pd.pivot_table(
            results_df,
            values="stance_changed",
            index="action_type",
            columns="framework",
            aggfunc="mean",
        )
        sns.boxplot(data=results_df, x="action_type", y="stance_changed", hue="framework")
        plt.title("Framework Effectiveness by Action Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / "action_type_analysis.png")
        plt.close()

        # 4. Value Distribution by Topic
        plt.figure(figsize=(15, 8))
        value_topic_dist = (
            results_df.groupby(["topic", "values"])["stance_changed"].mean().unstack()
        )
        sns.heatmap(value_topic_dist, annot=True, cmap="viridis", fmt=".2f")
        plt.title("Value Distribution Across Topics")
        plt.tight_layout()
        plt.savefig(self.save_dir / "value_topic_distribution.png")
        plt.close()

        # 5. Framework Comparison
        plt.figure(figsize=(12, 6))
        framework_comparison = results_df.groupby("framework")[
            ["stance_changed", "censored", "value_alignment"]
        ].mean()
        framework_comparison.plot(kind="bar")
        plt.title("Framework Performance Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / "framework_comparison.png")
        plt.close()

    def generate_analysis_report(self, results_df: pd.DataFrame, metrics: dict) -> str:
        """
        Generate a detailed analysis report.

        Args:
            results_df: DataFrame containing experiment results
            metrics: Dictionary of calculated metrics

        Returns:
            Formatted report string
        """
        report = ["# Enhanced Analysis Report\n"]

        # Overall Statistics
        report.append("## Overall Statistics")
        report.append(f"- Total scenarios analyzed: {len(results_df)}")
        report.append(f"- Overall stance change rate: {results_df['stance_changed'].mean():.2%}")
        report.append(f"- Overall censorship rate: {results_df['censored'].mean():.2%}\n")

        # Value-based Analysis
        report.append("## Value-based Analysis")
        for value, score in metrics["value_based_metrics"]["alignment_scores"].items():
            report.append(f"- {value}: {score:.2f} alignment score")
        report.append("")

        # Topic-based Analysis
        report.append("## Topic-based Analysis")
        for topic, stats in metrics["topic_based_metrics"]["fluctuation"].items():
            report.append(f"### {topic}")
            report.append(f"- Stance change rate: {stats:.2%}")
            report.append(
                f"- Censorship rate: {metrics['topic_based_metrics']['censorship'][topic]:.2%}"
            )
        report.append("")

        # Framework Effectiveness
        report.append("## Framework Effectiveness")
        framework_stats = results_df.groupby("framework")["stance_changed"].mean()
        for framework, effectiveness in framework_stats.items():
            report.append(f"- {framework}: {effectiveness:.2%} effectiveness rate")

        return "\n".join(report)
