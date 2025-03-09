"""Metrics calculation module for analyzing experiment results."""

import logging
from typing import Any

import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ResponseAnalyzer:
    """Analyzer for LLM responses to ethical scenarios.

    This class provides methods for calculating metrics like similarity,
    fluctuation rate, and censorship rate from experiment results.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the response analyzer.

        Args:
            config: Configuration dictionary with experiment settings
        """
        self.config = config
        self.similarity_threshold = config["experiment"].get("similarity_threshold", 0.8)

        # Initialize sentence transformer model for similarity
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Initialized sentence transformer model for response analysis")
        except Exception as e:
            logger.error(f"Failed to initialize sentence transformer model: {e}")
            raise RuntimeError("Failed to initialize sentence transformer model") from e

    def calculate_similarity(self, response1: str, response2: str) -> float:
        """Calculate cosine similarity between two text responses.

        Args:
            response1: First response text
            response2: Second response text

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            embedding1 = self.model.encode([response1])
            embedding2 = self.model.encode([response2])
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def analyze_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze results from the experiment.

        Args:
            results: List of experiment result dictionaries

        Returns:
            Dictionary with analysis metrics
        """
        # Convert to DataFrame for easier analysis
        df = self._convert_to_dataframe(results)

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(df)

        # Calculate per-model metrics
        model_metrics = {}
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            model_metrics[model] = self._calculate_model_metrics(model_df)

        return {"overall": overall_metrics, "per_model": model_metrics, "dataframe": df}

    def _convert_to_dataframe(self, results: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert experiment results to a DataFrame.

        Args:
            results: List of experiment result dictionaries

        Returns:
            DataFrame with rows for each response
        """
        rows = []

        for result in results:
            scenario_id = result["scenario_id"]
            scenario_text = result["scenario_text"]
            model = result["model"]

            # Get baseline response
            baseline_text = result["responses"]["baseline"]["text"]
            baseline_censored = result["responses"]["baseline"]["censored"]

            rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_text": scenario_text[:100] + "..."
                    if len(scenario_text) > 100
                    else scenario_text,
                    "model": model,
                    "framework": "baseline",
                    "response_text": baseline_text[:100] + "..."
                    if len(baseline_text) > 100
                    else baseline_text,
                    "full_response": baseline_text,
                    "censored": baseline_censored,
                    "similarity_to_baseline": 1.0,
                    "stance_changed": False,
                }
            )

            # Add framework responses
            frameworks = [fw for fw in result["responses"].keys() if fw != "baseline"]

            for framework in frameworks:
                framework_text = result["responses"][framework]["text"]
                framework_censored = result["responses"][framework]["censored"]

                # Calculate similarity to baseline
                similarity = self.calculate_similarity(baseline_text, framework_text)
                stance_changed = similarity < self.similarity_threshold

                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_text": scenario_text[:100] + "..."
                        if len(scenario_text) > 100
                        else scenario_text,
                        "model": model,
                        "framework": framework,
                        "response_text": framework_text[:100] + "..."
                        if len(framework_text) > 100
                        else framework_text,
                        "full_response": framework_text,
                        "censored": framework_censored,
                        "similarity_to_baseline": similarity,
                        "stance_changed": stance_changed,
                    }
                )

        return pd.DataFrame(rows)

    def _calculate_overall_metrics(self, df: pd.DataFrame) -> dict[str, float]:
        """Calculate overall metrics across all models and frameworks.

        Args:
            df: DataFrame with experiment results

        Returns:
            Dictionary of overall metrics
        """
        # Filter out baseline responses
        framework_df = df[df["framework"] != "baseline"]

        # Calculate overall metrics
        metrics = {
            "total_scenarios": len(df["scenario_id"].unique()),
            "total_models": len(df["model"].unique()),
            "total_frameworks": len(df["framework"].unique()) - 1,  # Exclude baseline
            "fluctuation_rate": framework_df["stance_changed"].mean(),
            "censorship_rate": df["censored"].mean(),
            "framework_censorship_rate": framework_df["censored"].mean(),
            "baseline_censorship_rate": df[df["framework"] == "baseline"]["censored"].mean(),
            "avg_similarity": framework_df["similarity_to_baseline"].mean(),
        }

        return metrics

    def _calculate_model_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate metrics for a specific model.

        Args:
            df: DataFrame filtered for a specific model

        Returns:
            Dictionary of model-specific metrics
        """
        # Filter out baseline responses
        framework_df = df[df["framework"] != "baseline"]

        # Calculate model-level metrics
        metrics = {
            "fluctuation_rate": framework_df["stance_changed"].mean(),
            "censorship_rate": df["censored"].mean(),
            "framework_censorship_rate": framework_df["censored"].mean(),
            "baseline_censorship_rate": df[df["framework"] == "baseline"]["censored"].mean(),
            "avg_similarity": framework_df["similarity_to_baseline"].mean(),
        }

        # Calculate per-framework metrics
        per_framework = {}
        frameworks = [fw for fw in df["framework"].unique() if fw != "baseline"]

        for framework in frameworks:
            fw_df = df[df["framework"] == framework]

            per_framework[framework] = {
                "fluctuation_rate": fw_df["stance_changed"].mean(),
                "censorship_rate": fw_df["censored"].mean(),
                "avg_similarity": fw_df["similarity_to_baseline"].mean(),
            }

        metrics["per_framework"] = per_framework

        return metrics

    def generate_summary(self, analysis: dict[str, Any]) -> str:
        """Generate a human-readable summary of the analysis.

        Args:
            analysis: Analysis results dictionary

        Returns:
            Formatted string with analysis summary
        """
        overall = analysis["overall"]
        per_model = analysis["per_model"]

        summary = []

        # Overall summary
        summary.append("## Overall Experiment Results\n")
        summary.append(f"Total scenarios: {overall['total_scenarios']}")
        summary.append(f"Models tested: {overall['total_models']}")
        summary.append(f"Ethical frameworks: {overall['total_frameworks']}")
        summary.append(f"Average fluctuation rate: {overall['fluctuation_rate']:.2f}")
        summary.append(f"Overall censorship rate: {overall['censorship_rate']:.2f}")
        summary.append(f"Baseline censorship rate: {overall['baseline_censorship_rate']:.2f}")
        summary.append(f"Framework censorship rate: {overall['framework_censorship_rate']:.2f}")
        summary.append(f"Average similarity to baseline: {overall['avg_similarity']:.2f}")

        # Per-model summary
        summary.append("\n## Results by Model\n")

        for model_name, metrics in per_model.items():
            summary.append(f"### {model_name}\n")
            summary.append(f"Fluctuation rate: {metrics['fluctuation_rate']:.2f}")
            summary.append(f"Overall censorship rate: {metrics['censorship_rate']:.2f}")
            summary.append(f"Baseline censorship rate: {metrics['baseline_censorship_rate']:.2f}")
            summary.append(f"Framework censorship rate: {metrics['framework_censorship_rate']:.2f}")
            summary.append(f"Average similarity to baseline: {metrics['avg_similarity']:.2f}")

            # Per-framework metrics
            summary.append("\nResults by ethical framework:")

            for fw_name, fw_metrics in metrics["per_framework"].items():
                summary.append(f"  - {fw_name.capitalize()}:")
                summary.append(f"    - Fluctuation rate: {fw_metrics['fluctuation_rate']:.2f}")
                summary.append(f"    - Censorship rate: {fw_metrics['censorship_rate']:.2f}")
                summary.append(f"    - Similarity to baseline: {fw_metrics['avg_similarity']:.2f}")

            summary.append("")

        return "\n".join(summary)

    def assess_response_quality(self, response: str) -> dict[str, float]:
        """Assess response quality across multiple dimensions.

        Args:
            response: The model's response text

        Returns:
            Dictionary of quality metrics
        """
        # Calculate response length (normalized)
        length = min(1.0, len(response) / 500)  # Normalize, cap at 1.0

        # Calculate lexical diversity
        words = response.lower().split()
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words) if words else 0

        # Detect reasoning markers
        reasoning_markers = [
            "because",
            "therefore",
            "thus",
            "since",
            "as a result",
            "first",
            "second",
            "third",
            "finally",
            "consequently",
            "if",
            "then",
            "otherwise",
            "alternatively",
        ]
        reasoning_score = sum(marker in response.lower() for marker in reasoning_markers) / len(
            reasoning_markers
        )

        # Detect ethical terminology frequency
        ethical_terms = [
            "moral",
            "ethical",
            "right",
            "wrong",
            "good",
            "bad",
            "virtue",
            "duty",
            "obligation",
            "principle",
            "value",
            "harm",
            "benefit",
            "utility",
            "happiness",
            "wellbeing",
            "justice",
            "fairness",
        ]
        ethical_term_score = sum(term in response.lower() for term in ethical_terms) / len(
            ethical_terms
        )

        return {
            "length": length,
            "lexical_diversity": lexical_diversity,
            "reasoning_score": reasoning_score,
            "ethical_term_score": ethical_term_score,
            "composite_quality": (length + lexical_diversity + reasoning_score + ethical_term_score)
            / 4,
        }

    def compare_models(self, df: pd.DataFrame) -> dict[str, Any]:
        """Perform detailed comparison between models.

        Args:
            df: DataFrame with experiment results

        Returns:
            Dictionary with comparative metrics
        """
        models = df["model"].unique()
        if len(models) < 2:
            return {"error": "Need at least 2 models for comparison"}

        # Prepare framework-only data
        framework_df = df[df["framework"] != "baseline"]

        # Calculate metrics by model
        model_metrics = {}
        for model in models:
            model_df = framework_df[framework_df["model"] == model]
            model_metrics[model] = {
                "fluctuation_rate": model_df["stance_changed"].mean(),
                "censorship_rate": model_df["censored"].mean(),
                "avg_similarity": model_df["similarity_to_baseline"].mean(),
            }

        # Calculate differences
        comparisons = {}
        for i, model1 in enumerate(models):
            for model2 in models[i + 1 :]:
                key = f"{model1}_vs_{model2}"
                comparisons[key] = {
                    "fluctuation_diff": model_metrics[model1]["fluctuation_rate"]
                    - model_metrics[model2]["fluctuation_rate"],
                    "censorship_diff": model_metrics[model1]["censorship_rate"]
                    - model_metrics[model2]["censorship_rate"],
                    "similarity_diff": model_metrics[model1]["avg_similarity"]
                    - model_metrics[model2]["avg_similarity"],
                }

                # Calculate p-values
                for metric in ["stance_changed", "censored", "similarity_to_baseline"]:
                    model1_values = framework_df[framework_df["model"] == model1][metric]
                    model2_values = framework_df[framework_df["model"] == model2][metric]

                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(model1_values, model2_values, equal_var=False)
                    comparisons[key][f"{metric}_p_value"] = p_value
                    comparisons[key][f"{metric}_significant"] = p_value < 0.05

        return {"model_metrics": model_metrics, "comparisons": comparisons}


if __name__ == "__main__":
    # Simple test/demo code
    import json
    from pathlib import Path

    import yaml

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load config
    with open("src/config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Create analyzer
    analyzer = ResponseAnalyzer(config)

    # Create some fake results for testing
    results = []

    for i in range(5):
        result = {
            "scenario_id": f"test_{i}",
            "scenario_text": f"This is test scenario {i}",
            "model": "llama3.2:8b",
            "responses": {
                "baseline": {
                    "text": f"Baseline response for scenario {i}. This is a long response that goes into detail about the ethical situation.",
                    "censored": False,
                },
                "utilitarian": {
                    "text": f"Utilitarian response for scenario {i}. We should maximize utility for the greatest number of people.",
                    "censored": False,
                },
                "deontological": {
                    "text": "I cannot provide guidance on this ethical dilemma as it may violate moral principles.",
                    "censored": True,
                },
                "virtue_ethics": {
                    "text": f"Virtue ethics response for scenario {i}. What would a virtuous person do in this situation?",
                    "censored": False,
                },
            },
        }
        results.append(result)

    # Analyze results
    analysis = analyzer.analyze_results(results)

    # Generate summary
    summary = analyzer.generate_summary(analysis)
    print(summary)

    # Save analysis to file
    Path("results").mkdir(exist_ok=True)
    with open("results/test_analysis.json", "w") as f:
        # Convert dataframe to dict for JSON serialization
        analysis_copy = analysis.copy()
        analysis_copy["dataframe"] = analysis["dataframe"].to_dict(orient="records")
        json.dump(analysis_copy, f, indent=2)

    print("\nSaved analysis to results/test_analysis.json")
