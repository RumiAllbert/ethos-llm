"""Logging utilities for the LLM Ethics Experiment."""

import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(
    log_level: int = logging.INFO,
    log_file: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Set up logging for the experiment.

    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file
        config: Optional configuration dictionary with experiment settings
    """
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler (if log_file specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    # Use config to determine log file (if config provided but no explicit log_file)
    elif config and not log_file:
        save_dir = Path(config["experiment"]["save_dir"])
        log_path = save_dir / "logs" / "experiment.log"
        log_path.parent.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Log basic information
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level {logging.getLevelName(log_level)}")
    if log_file or (config and not log_file):
        logger.info(f"Logging to file: {log_path}")


def get_experiment_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module of the experiment.

    Args:
        name: Name of the module or component

    Returns:
        Logger instance
    """
    return logging.getLogger(f"llm_ethic.{name}")


if __name__ == "__main__":
    # Simple test/demo code
    import yaml

    # Load config
    with open("src/config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Test setting up logging
    setup_logging(config=config)

    # Get a logger and test it
    logger = get_experiment_logger("test")
    logger.info("This is a test message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    print("Logging test completed. Check the log file in the results directory.")
