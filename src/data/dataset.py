"""Dataset loading and processing module."""

import logging
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

        logger.info(f"Loading Moral Stories dataset from {source}")

        # Load dataset from Hugging Face
        try:
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
