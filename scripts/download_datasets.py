#!/usr/bin/env python
"""Script for downloading datasets for the LLM Ethics Experiment.

This script downloads the required datasets from the Hugging Face datasets
repository and caches them locally.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

# Add the project root to the Python path
import fix_path  # noqa
from datasets import load_dataset

from src.data.dataset import DatasetLoader
from src.utils.config import load_config
from src.utils.logging import get_experiment_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Download datasets for the LLM Ethics Experiment")
    parser.add_argument("--config", "-c", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="List of datasets to download (defaults to all in config)",
    )
    parser.add_argument("--cache-dir", type=str, help="Directory to cache datasets")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def download_datasets(
    config: dict[str, Any],
    datasets_to_download: list[str] | None = None,
    cache_dir: str | None = None,
) -> None:
    """Download datasets and cache them locally.

    Args:
        config: Configuration dictionary
        datasets_to_download: Optional list of specific datasets to download
        cache_dir: Optional directory to cache datasets
    """
    logger = get_experiment_logger("download")
    logger.info("Starting dataset download")

    dataset_configs = config["datasets"]

    if datasets_to_download:
        # Filter to requested datasets
        dataset_configs = {k: v for k, v in dataset_configs.items() if k in datasets_to_download}

    if not dataset_configs:
        logger.error("No datasets configured or selected for download")
        return

    logger.info(
        f"Will download {len(dataset_configs)} datasets: {', '.join(dataset_configs.keys())}"
    )

    # Create cache directory if specified
    if cache_dir:
        os.environ["HF_DATASETS_CACHE"] = cache_dir
        logger.info(f"Using custom cache directory: {cache_dir}")
        Path(cache_dir).mkdir(exist_ok=True, parents=True)

    # Download each dataset
    for name, ds_config in dataset_configs.items():
        logger.info(f"Downloading dataset: {name}")

        try:
            source = ds_config["source"]
            subset = ds_config.get("subset")
            config_name = ds_config.get("config")  # Get dataset-specific config name

            if subset and config_name:
                # Dataset with both subset and config name
                logger.info(f"Loading {source} (subset: {subset}, config: {config_name})")
                dataset = load_dataset(source, subset, name=config_name, trust_remote_code=True)
            elif subset:
                # Dataset with just subset
                logger.info(f"Loading {source} (subset: {subset})")
                dataset = load_dataset(source, subset, trust_remote_code=True)
            elif config_name:
                # Dataset with just config name
                logger.info(f"Loading {source} (config: {config_name})")
                dataset = load_dataset(source, config_name, trust_remote_code=True)
            else:
                # Dataset with no subset or config
                logger.info(f"Loading {source}")
                dataset = load_dataset(source, trust_remote_code=True)

            # Log available splits
            splits = list(dataset.keys())
            logger.info(f"Dataset loaded successfully with splits: {', '.join(splits)}")

            # Log number of examples in each split
            for split in splits:
                logger.info(f"  Split '{split}' contains {len(dataset[split])} examples")

        except Exception as e:
            logger.error(f"Error downloading dataset {name}: {e}")

    logger.info("Dataset download completed")

    # Initialize the dataset loader to verify the datasets can be properly loaded
    logger.info("Verifying datasets can be loaded correctly")
    try:
        dataset_loader = DatasetLoader(config)
        loaded_datasets = dataset_loader.load_all_datasets()

        # Log success information
        total_scenarios = sum(len(scenarios) for scenarios in loaded_datasets.values())
        logger.info(f"Successfully verified loading of {total_scenarios} total scenarios")

        # Create a sample CSV file
        csv_path = Path("data/sample_scenarios.csv")
        csv_path.parent.mkdir(exist_ok=True)
        dataset_loader.export_to_csv(str(csv_path))
        logger.info(f"Exported sample scenarios to {csv_path}")

    except Exception as e:
        logger.error(f"Error verifying datasets: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level)

    # Load configuration
    config = load_config(args.config)

    # Download datasets
    download_datasets(config, datasets_to_download=args.datasets, cache_dir=args.cache_dir)
