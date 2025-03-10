"""
Dataset handlers for different data sources in the LLM Ethics experiment.
"""

import ast
from typing import Any

import pandas as pd
from datasets import load_dataset


class DailyDilemmaDataset:
    """Handler for the daily_dilemmas dataset with rich metadata and analysis capabilities."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the daily dilemmas dataset handler.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config["datasets"]["daily_dilemmas"]
        self.template = self.config["scenario_construction"]["template"]
        self.max_samples = self.config["max_samples"]

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the dataset from HuggingFace datasets.

        Returns:
            DataFrame containing the dataset
        """
        dataset = load_dataset(self.config["source"])
        df = pd.DataFrame(dataset["test"])  # Using test split as default

        if self.max_samples:
            df = df.sample(n=min(self.max_samples, len(df)), random_state=42)

        return df

    def construct_scenario(self, row: pd.Series) -> tuple[str, dict[str, Any]]:
        """
        Construct a rich scenario from the dataset fields.

        Args:
            row: DataFrame row containing dilemma data

        Returns:
            Tuple of (formatted scenario string, metadata dictionary)
        """
        # Safely evaluate the values_aggregated string as a list
        try:
            values = ast.literal_eval(row["values_aggregated"])
        except:
            values = []

        # Format the scenario using the template
        scenario = self.template.format(
            basic_situation=row["basic_situation"],
            dilemma_situation=row["dilemma_situation"],
            action=row["action"],
            negative_consequence=row["negative_consequence"],
            values_aggregated=", ".join(values),
        )

        # Collect metadata for analysis
        metadata = {
            "topic": row["topic"],
            "topic_group": row["topic_group"],
            "values": values,
            "action_type": row["action_type"],
        }

        return scenario, metadata

    def process_dataset(self) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Process the entire dataset into scenarios and metadata.

        Returns:
            Tuple of (list of scenarios, list of metadata dictionaries)
        """
        df = self.load_dataset()
        scenarios = []
        metadata_list = []

        for _, row in df.iterrows():
            scenario, metadata = self.construct_scenario(row)
            scenarios.append(scenario)
            metadata_list.append(metadata)

        return scenarios, metadata_list
