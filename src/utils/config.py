"""Configuration utilities for loading and validating experiment settings."""

import logging
from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize
from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)


class EthicalFramework(BaseModel):
    """Model for an ethical framework configuration.

    Attributes:
        name: The name of the ethical framework
        prompt: The prompt text to use for this framework
    """

    name: str
    prompt: str


class ModelConfig(BaseModel):
    """Model for LLM model configuration.

    Attributes:
        name: The Ollama model name
        display_name: Human-readable display name
    """

    name: str
    display_name: str


class DatasetConfig(BaseModel):
    """Model for dataset configuration.

    Attributes:
        source: The dataset source (Hugging Face dataset ID)
        subset: Optional dataset subset name
        max_samples: Maximum number of samples to use
        scenario_field: Field name containing the scenario text
    """

    source: str
    subset: str | None = None
    max_samples: int | None = None
    scenario_field: str = "text"


class ExperimentConfig(BaseModel):
    """Model for experiment configuration settings.

    Attributes:
        similarity_threshold: Threshold for determining if stance changed
        ollama_timeout: Timeout in seconds for Ollama API calls
        delay_between_calls: Delay in seconds between API calls
        save_dir: Directory to save results in
        random_seed: Random seed for reproducibility
    """

    similarity_threshold: float = 0.8
    ollama_timeout: int = 30
    delay_between_calls: float = 1.0
    save_dir: str = "results"
    random_seed: int = 42

    @validator("similarity_threshold")
    def validate_threshold(cls, v: float) -> float:
        """Validate that the similarity threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        return v

    @validator("ollama_timeout")
    def validate_timeout(cls, v: int) -> int:
        """Validate that the timeout is positive."""
        if v <= 0:
            raise ValueError("ollama_timeout must be positive")
        return v

    @validator("delay_between_calls")
    def validate_delay(cls, v: float) -> float:
        """Validate that the delay is non-negative."""
        if v < 0:
            raise ValueError("delay_between_calls must be non-negative")
        return v


class Config(BaseModel):
    """Top-level configuration model for the experiment.

    Attributes:
        models: List of models to use in the experiment
        frameworks: List of ethical frameworks to test
        datasets: Dictionary of dataset configurations
        experiment: Experiment settings
        censorship_phrases: List of phrases indicating censorship
    """

    models: list[ModelConfig]
    frameworks: list[EthicalFramework]
    datasets: dict[str, DatasetConfig]
    experiment: ExperimentConfig
    censorship_phrases: list[str]


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file or Hydra.

    Args:
        config_path: Optional path to a YAML configuration file

    Returns:
        Configuration dictionary
    """
    if config_path:
        # Load from specified YAML file
        config_path = Path(config_path)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading configuration from {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Load using Hydra
        try:
            logger.info("Loading configuration using Hydra")
            with initialize(version_base=None, config_path="../config"):
                config = compose(config_name="config")
                # Convert OmegaConf to dict
                config = {k: dict(v) if isinstance(v, dict) else v for k, v in dict(config).items()}
        except Exception as e:
            logger.error(f"Error loading configuration with Hydra: {e}")
            # Fall back to default config file
            logger.info("Falling back to default config file")
            with open(Path(__file__).parent.parent / "config" / "config.yaml") as f:
                config = yaml.safe_load(f)

    # Validate configuration
    try:
        Config(**config)
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ValueError(f"Invalid configuration: {e}") from e

    return config


def setup_experiment_dirs(config: dict[str, Any]) -> None:
    """Set up directories for saving experiment results.

    Args:
        config: Configuration dictionary
    """
    save_dir = Path(config["experiment"]["save_dir"])
    save_dir.mkdir(exist_ok=True, parents=True)

    # Create subdirectories
    (save_dir / "logs").mkdir(exist_ok=True)
    (save_dir / "raw_responses").mkdir(exist_ok=True)
    (save_dir / "analysis").mkdir(exist_ok=True)
    (save_dir / "plots").mkdir(exist_ok=True)

    logger.info(f"Created experiment directories in {save_dir}")


if __name__ == "__main__":
    # Simple test/demo code
    logging.basicConfig(level=logging.INFO)

    # Test loading config from default location
    config = load_config()

    # Print some configuration values
    print("\nConfiguration Summary:")
    print(f"Models: {', '.join(model['name'] for model in config['models'])}")
    print(f"Frameworks: {', '.join(fw['name'] for fw in config['frameworks'])}")
    print(f"Datasets: {', '.join(config['datasets'].keys())}")

    # Setup experiment directories
    setup_experiment_dirs(config)

    print("\nExperiment directories created successfully!")
