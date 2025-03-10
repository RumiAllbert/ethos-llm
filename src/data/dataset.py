"""Dataset loading and processing module."""

import logging
import random
import re
from typing import Any

import pandas as pd
from datasets import load_dataset
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ScenarioItem(BaseModel):
    """Data model for a single scenario item.

    Attributes:
        id: Unique identifier for the scenario
        text: The scenario text/description
        source: The source dataset of the scenario
        metadata: Optional additional metadata
    """

    id: str
    text: str
    source: str
    metadata: dict[str, Any] = {}


class DatasetLoader:
    """Class for loading and processing ethical reasoning datasets.

    This class handles loading ethical reasoning datasets from the Hugging Face
    datasets library and processing them into a standardized format.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the dataset loader.

        Args:
            config: Configuration dictionary with dataset settings
        """
        self.config = config
        self.datasets: dict[str, list[ScenarioItem]] = {}

    def load_ethics_suite(self, max_samples: int | None = None) -> list[ScenarioItem]:
        """Load the EthicsSuite dataset.

        Args:
            max_samples: Optional maximum number of samples to load

        Returns:
            List of scenario items
        """
        dataset_config = self.config["datasets"]["ethics_suite"]
        source = dataset_config["source"]
        subset = dataset_config.get("subset")
        field = dataset_config.get("scenario_field", "text")

        logger.info(f"Loading EthicsSuite dataset from {source}")

        # Load dataset from Hugging Face
        try:
            if subset:
                dataset = load_dataset(source, subset)
            else:
                dataset = load_dataset(source)

            # Get the test split if available, otherwise use train
            split = "test" if "test" in dataset else "train"
            data = dataset[split]

            # Convert to standardized format
            scenarios = []
            for i, item in enumerate(data):
                if max_samples and i >= max_samples:
                    break

                scenario = ScenarioItem(
                    id=f"ethics_{i}",
                    text=item[field],
                    source="ethics_suite",
                    metadata={k: v for k, v in item.items() if k != field},
                )
                scenarios.append(scenario)

            logger.info(f"Loaded {len(scenarios)} scenarios from EthicsSuite")
            return scenarios

        except Exception as e:
            logger.error(f"Error loading EthicsSuite dataset: {e}")
            return []

    def load_moral_stories(self, max_samples: int | None = None) -> list[ScenarioItem]:
        """Load the Moral Stories dataset.

        Args:
            max_samples: Optional maximum number of samples to load

        Returns:
            List of scenario items
        """
        dataset_config = self.config["datasets"]["moral_stories"]
        source = dataset_config["source"]
        field = dataset_config.get("scenario_field", "situation")
        config_name = dataset_config.get(
            "config", "full"
        )  # Get the config name or default to "full"

        logger.info(f"Loading Moral Stories dataset from {source} (config: {config_name})")

        # Load dataset from Hugging Face
        try:
            # Pass the config name to load_dataset
            dataset = load_dataset(source, config_name, trust_remote_code=True)

            # Get the test split if available, otherwise use train
            split = "test" if "test" in dataset else "train"
            data = dataset[split]

            # Convert to standardized format
            scenarios = []
            for i, item in enumerate(data):
                if max_samples and i >= max_samples:
                    break

                scenario = ScenarioItem(
                    id=f"moral_{i}",
                    text=item[field],
                    source="moral_stories",
                    metadata={k: v for k, v in item.items() if k != field},
                )
                scenarios.append(scenario)

            logger.info(f"Loaded {len(scenarios)} scenarios from Moral Stories")
            return scenarios

        except Exception as e:
            logger.error(f"Error loading Moral Stories dataset: {e}")
            return []

    def load_all_datasets(self) -> dict[str, list[ScenarioItem]]:
        """Load all configured datasets.

        Returns:
            Dictionary mapping dataset names to lists of scenario items
        """
        max_samples_ethics = self.config["datasets"]["ethics_suite"].get("max_samples")
        max_samples_moral = self.config["datasets"]["moral_stories"].get("max_samples")

        self.datasets = {
            "ethics_suite": self.load_ethics_suite(max_samples=max_samples_ethics),
            "moral_stories": self.load_moral_stories(max_samples=max_samples_moral),
        }

        total_scenarios = sum(len(scenarios) for scenarios in self.datasets.values())
        logger.info(f"Loaded {total_scenarios} total scenarios across all datasets")

        return self.datasets

    def get_all_scenarios(self) -> list[ScenarioItem]:
        """Get all loaded scenarios combined into a single list.

        Returns:
            Combined list of all scenario items
        """
        if not self.datasets:
            self.load_all_datasets()

        all_scenarios = []
        for scenarios in self.datasets.values():
            all_scenarios.extend(scenarios)

        return all_scenarios

    def export_to_csv(self, filepath: str) -> None:
        """Export all loaded scenarios to a CSV file.

        Args:
            filepath: Path to save the CSV file
        """
        scenarios = self.get_all_scenarios()

        # Convert to DataFrame
        data = []
        for scenario in scenarios:
            item = {"id": scenario.id, "text": scenario.text, "source": scenario.source}
            # Add metadata as flattened columns
            for k, v in scenario.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    item[k] = v

            data.append(item)

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(df)} scenarios to {filepath}")

    def stratified_sample(self, max_samples: int | None = None) -> list[ScenarioItem]:
        """Get a stratified sample of scenarios to ensure diversity.

        Args:
            max_samples: Maximum number of samples to return

        Returns:
            List of scenario items with balanced representation
        """
        all_scenarios = self.get_all_scenarios()

        # Group by source
        scenarios_by_source = {}
        for scenario in all_scenarios:
            if scenario.source not in scenarios_by_source:
                scenarios_by_source[scenario.source] = []
            scenarios_by_source[scenario.source].append(scenario)

        # Calculate samples per source
        if max_samples:
            samples_per_source = max(1, max_samples // len(scenarios_by_source))
            result = []

            # Take balanced samples from each source
            for source, scenarios in scenarios_by_source.items():
                # Random sample without replacement
                source_samples = random.sample(scenarios, min(samples_per_source, len(scenarios)))
                result.extend(source_samples)

            # If we need more samples to reach max_samples
            if len(result) < max_samples:
                # Get remaining scenarios not already sampled
                remaining = [s for s in all_scenarios if s not in result]
                # Add random samples until we reach max_samples or run out
                additional = random.sample(
                    remaining, min(max_samples - len(result), len(remaining))
                )
                result.extend(additional)

            return result

        return all_scenarios

    def analyze_scenario_complexity(self, scenario: ScenarioItem) -> dict[str, Any]:
        """Analyze the complexity of an ethical scenario.

        Args:
            scenario: The scenario to analyze

        Returns:
            Dictionary with complexity metrics
        """
        text = scenario.text

        # Word count
        word_count = len(text.split())

        # Sentence count
        sentence_count = len(re.split(r"[.!?]+", text))

        # Average words per sentence
        avg_words_per_sentence = word_count / max(1, sentence_count)

        # Count ethical dilemma indicators
        dilemma_terms = [
            "should",
            "must",
            "ought",
            "right",
            "wrong",
            "good",
            "bad",
            "ethical",
            "moral",
            "dilemma",
            "choice",
            "decide",
            "obligation",
        ]
        dilemma_score = sum(term in text.lower() for term in dilemma_terms)

        # Determine complexity category
        if word_count < 50:
            complexity = "simple"
        elif word_count < 100:
            complexity = "moderate"
        else:
            complexity = "complex"

        # Check for presence of multiple stakeholders
        stakeholder_markers = ["person", "people", "they", "he", "she", "individual", "group"]
        stakeholders_present = sum(marker in text.lower() for marker in stakeholder_markers)

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": avg_words_per_sentence,
            "dilemma_score": dilemma_score,
            "complexity": complexity,
            "stakeholders_present": stakeholders_present
            > 2,  # Boolean indicating multiple stakeholders
        }

    def tag_scenarios(self, scenarios: list[ScenarioItem]) -> list[ScenarioItem]:
        """Tag scenarios with demographic and ethical theme information.

        Args:
            scenarios: List of scenarios to tag

        Returns:
            Tagged scenarios
        """
        # Demographic identifiers
        demographics = {
            "gender": ["man", "woman", "boy", "girl", "male", "female", "he", "she"],
            "age": ["child", "young", "old", "elderly", "senior", "teen", "adult"],
            "family": ["parent", "mother", "father", "son", "daughter", "family", "sibling"],
        }

        # Ethical themes
        themes = {
            "harm": ["harm", "hurt", "damage", "injury", "pain"],
            "fairness": ["fair", "unfair", "equal", "unequal", "justice", "injustice"],
            "loyalty": ["loyal", "disloyal", "betray", "faithful", "allegiance"],
            "authority": ["authority", "obey", "respect", "tradition", "honor"],
            "purity": ["pure", "impure", "disgusting", "sanctity", "sacred", "clean", "dirty"],
        }

        for scenario in scenarios:
            # Initialize tags
            demographic_tags = {}
            theme_tags = {}

            text = scenario.text.lower()

            # Check for demographic mentions
            for demo_category, demo_terms in demographics.items():
                demo_mentions = sum(1 for term in demo_terms if term in text)
                if demo_mentions > 0:
                    demographic_tags[demo_category] = True

            # Check for ethical themes
            for theme_name, theme_terms in themes.items():
                theme_mentions = sum(1 for term in theme_terms if term in text)
                if theme_mentions > 0:
                    theme_tags[theme_name] = theme_mentions

            # Add tags to metadata
            scenario.metadata["demographics"] = demographic_tags
            scenario.metadata["ethical_themes"] = theme_tags

            # Determine primary ethical theme (if any)
            if theme_tags:
                primary_theme = max(theme_tags.items(), key=lambda x: x[1])[0]
                scenario.metadata["primary_theme"] = primary_theme

        return scenarios


if __name__ == "__main__":
    # Simple test/demo code
    import yaml

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load config
    with open("src/config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Test dataset loading
    loader = DatasetLoader(config)
    datasets = loader.load_all_datasets()

    # Print some statistics
    for name, scenarios in datasets.items():
        print(f"{name}: {len(scenarios)} scenarios")

    # Print a sample scenario
    if datasets["ethics_suite"]:
        print("\nSample Ethics scenario:")
        print(datasets["ethics_suite"][0].text)

    if datasets["moral_stories"]:
        print("\nSample Moral Stories scenario:")
        print(datasets["moral_stories"][0].text)
