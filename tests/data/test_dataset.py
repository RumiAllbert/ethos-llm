"""Tests for the dataset loader module."""

import os
import tempfile
from typing import Any
from unittest import mock

import pandas as pd
import pytest

from src.data.dataset import DatasetLoader, ScenarioItem


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Fixture providing a mock configuration."""
    return {
        "datasets": {
            "ethics_suite": {
                "source": "hendrycks/ethics",
                "subset": "commonsense",
                "max_samples": 5,
                "scenario_field": "text",
            },
            "moral_stories": {
                "source": "demelin/moral_stories",
                "max_samples": 5,
                "scenario_field": "situation",
            },
        }
    }


@pytest.fixture
def mock_ethics_dataset():
    """Fixture providing a mock ethics dataset."""
    return {
        "test": [
            {"text": "Scenario 1", "label": "good"},
            {"text": "Scenario 2", "label": "bad"},
            {"text": "Scenario 3", "label": "good"},
            {"text": "Scenario 4", "label": "bad"},
            {"text": "Scenario 5", "label": "good"},
        ]
    }


@pytest.fixture
def mock_moral_dataset():
    """Fixture providing a mock moral stories dataset."""
    return {
        "train": [
            {"situation": "Moral situation 1", "intention": "good", "action": "helped"},
            {"situation": "Moral situation 2", "intention": "bad", "action": "harmed"},
            {"situation": "Moral situation 3", "intention": "good", "action": "helped"},
            {"situation": "Moral situation 4", "intention": "bad", "action": "harmed"},
            {"situation": "Moral situation 5", "intention": "good", "action": "helped"},
        ]
    }


def test_scenario_item():
    """Test ScenarioItem data model."""
    scenario = ScenarioItem(id="test_1", text="Test scenario", source="test")

    assert scenario.id == "test_1"
    assert scenario.text == "Test scenario"
    assert scenario.source == "test"
    assert scenario.metadata == {}

    # Test with metadata
    scenario = ScenarioItem(
        id="test_2",
        text="Test scenario with metadata",
        source="test",
        metadata={"label": "good", "score": 0.95},
    )

    assert scenario.id == "test_2"
    assert scenario.text == "Test scenario with metadata"
    assert scenario.source == "test"
    assert scenario.metadata == {"label": "good", "score": 0.95}


@mock.patch("src.data.dataset.load_dataset")
def test_load_ethics_suite(mock_load_dataset, mock_config, mock_ethics_dataset):
    """Test loading the EthicsSuite dataset."""
    mock_load_dataset.return_value = mock_ethics_dataset

    loader = DatasetLoader(mock_config)
    scenarios = loader.load_ethics_suite(max_samples=5)

    # Check that load_dataset was called correctly
    mock_load_dataset.assert_called_with("hendrycks/ethics", "commonsense")

    # Check returned scenarios
    assert len(scenarios) == 5
    assert all(isinstance(s, ScenarioItem) for s in scenarios)
    assert all(s.source == "ethics_suite" for s in scenarios)
    assert scenarios[0].text == "Scenario 1"
    assert scenarios[0].metadata["label"] == "good"


@mock.patch("src.data.dataset.load_dataset")
def test_load_moral_stories(mock_load_dataset, mock_config, mock_moral_dataset):
    """Test loading the Moral Stories dataset."""
    mock_load_dataset.return_value = mock_moral_dataset

    loader = DatasetLoader(mock_config)
    scenarios = loader.load_moral_stories(max_samples=5)

    # Check that load_dataset was called correctly
    mock_load_dataset.assert_called_with("demelin/moral_stories")

    # Check returned scenarios
    assert len(scenarios) == 5
    assert all(isinstance(s, ScenarioItem) for s in scenarios)
    assert all(s.source == "moral_stories" for s in scenarios)
    assert scenarios[0].text == "Moral situation 1"
    assert scenarios[0].metadata["intention"] == "good"
    assert scenarios[0].metadata["action"] == "helped"


@mock.patch("src.data.dataset.load_dataset")
def test_load_all_datasets(mock_load_dataset, mock_config, mock_ethics_dataset, mock_moral_dataset):
    """Test loading all datasets."""

    # Set up mock to return different datasets based on arguments
    def side_effect(*args, **kwargs):
        if args[0] == "hendrycks/ethics" and args[1] == "commonsense":
            return mock_ethics_dataset
        elif args[0] == "demelin/moral_stories":
            return mock_moral_dataset
        return {}

    mock_load_dataset.side_effect = side_effect

    loader = DatasetLoader(mock_config)
    datasets = loader.load_all_datasets()

    # Check returned datasets
    assert "ethics_suite" in datasets
    assert "moral_stories" in datasets
    assert len(datasets["ethics_suite"]) == 5
    assert len(datasets["moral_stories"]) == 5


@mock.patch("src.data.dataset.load_dataset")
def test_get_all_scenarios(mock_load_dataset, mock_config, mock_ethics_dataset, mock_moral_dataset):
    """Test getting all scenarios."""

    # Set up mock to return different datasets based on arguments
    def side_effect(*args, **kwargs):
        if args[0] == "hendrycks/ethics" and args[1] == "commonsense":
            return mock_ethics_dataset
        elif args[0] == "demelin/moral_stories":
            return mock_moral_dataset
        return {}

    mock_load_dataset.side_effect = side_effect

    loader = DatasetLoader(mock_config)
    scenarios = loader.get_all_scenarios()

    # Check returned scenarios
    assert len(scenarios) == 10  # 5 from each dataset
    assert len([s for s in scenarios if s.source == "ethics_suite"]) == 5
    assert len([s for s in scenarios if s.source == "moral_stories"]) == 5


@mock.patch("src.data.dataset.load_dataset")
def test_export_to_csv(mock_load_dataset, mock_config, mock_ethics_dataset, mock_moral_dataset):
    """Test exporting scenarios to CSV."""

    # Set up mock to return different datasets based on arguments
    def side_effect(*args, **kwargs):
        if args[0] == "hendrycks/ethics" and args[1] == "commonsense":
            return mock_ethics_dataset
        elif args[0] == "demelin/moral_stories":
            return mock_moral_dataset
        return {}

    mock_load_dataset.side_effect = side_effect

    loader = DatasetLoader(mock_config)

    # Create a temporary file for the CSV
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Export to CSV
        loader.export_to_csv(tmp_path)

        # Check the CSV file
        df = pd.read_csv(tmp_path)
        assert len(df) == 10  # 5 from each dataset
        assert "id" in df.columns
        assert "text" in df.columns
        assert "source" in df.columns
        assert "label" in df.columns  # Metadata from ethics
        assert "intention" in df.columns  # Metadata from moral stories
        assert "action" in df.columns  # Metadata from moral stories
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
