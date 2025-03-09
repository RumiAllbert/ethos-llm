"""Dashboard utilities for tracking experiment progress and checkpointing."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_progress_dashboard(
    total_scenarios: int, models: list[str], frameworks: list[str], save_dir: Path
) -> dict[str, Any]:
    """Create a progress tracking dashboard.

    Args:
        total_scenarios: Total number of scenarios to process
        models: List of models being tested
        frameworks: List of frameworks being tested
        save_dir: Directory where results are saved

    Returns:
        Dashboard state dictionary
    """
    progress = {
        "total_scenarios": total_scenarios,
        "total_models": len(models),
        "total_frameworks": len(frameworks),
        "total_combinations": total_scenarios * len(models) * len(frameworks),
        "completed": 0,
        "start_time": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "elapsed_time": "00:00:00",
        "estimated_time_remaining": "Unknown",
        "models_progress": {model: 0 for model in models},
        "scenarios_completed": [],
        # Track completed combinations for checkpointing
        "completed_combinations": [],
    }

    # Initialize dashboard file
    dashboard_path = save_dir / "dashboard.json"
    with open(dashboard_path, "w") as f:
        json.dump(progress, f, indent=2)

    # Create checkpoint file
    checkpoint_path = save_dir / "checkpoint.json"
    with open(checkpoint_path, "w") as f:
        json.dump(
            {"completed_combinations": [], "last_updated": datetime.now().isoformat()}, f, indent=2
        )

    return progress


def update_progress(
    dashboard: dict[str, Any], scenario_id: str, model: str, save_dir: Path
) -> dict[str, Any]:
    """Update the progress dashboard.

    Args:
        dashboard: Current dashboard state
        scenario_id: ID of completed scenario
        model: Model that completed the scenario
        save_dir: Directory where results are saved

    Returns:
        Updated dashboard state
    """
    # Update dashboard
    dashboard["completed"] += 1
    dashboard["models_progress"][model] += 1

    if scenario_id not in dashboard["scenarios_completed"]:
        dashboard["scenarios_completed"].append(scenario_id)

    # Add to completed combinations for checkpointing
    combination = f"{scenario_id}_{model}"
    if combination not in dashboard["completed_combinations"]:
        dashboard["completed_combinations"].append(combination)

    # Calculate times
    start_time = datetime.fromisoformat(dashboard["start_time"])
    elapsed = datetime.now() - start_time
    dashboard["elapsed_time"] = str(elapsed).split(".")[0]  # HH:MM:SS format
    dashboard["last_updated"] = datetime.now().isoformat()

    # Estimate time remaining
    if dashboard["completed"] > 0:
        rate = elapsed.total_seconds() / dashboard["completed"]
        remaining_items = dashboard["total_combinations"] - dashboard["completed"]
        estimated_seconds = rate * remaining_items
        estimated_time = timedelta(seconds=int(estimated_seconds))
        dashboard["estimated_time_remaining"] = str(estimated_time)

    # Update dashboard file
    dashboard_path = save_dir / "dashboard.json"
    with open(dashboard_path, "w") as f:
        json.dump(dashboard, f, indent=2)

    # Update checkpoint file (we do this more frequently)
    update_checkpoint(dashboard, save_dir)

    return dashboard


def update_checkpoint(dashboard: dict[str, Any], save_dir: Path) -> None:
    """Update the checkpoint file with current progress.

    Args:
        dashboard: Current dashboard state
        save_dir: Directory where results are saved
    """
    checkpoint_path = save_dir / "checkpoint.json"

    checkpoint_data = {
        "completed_combinations": dashboard["completed_combinations"],
        "last_updated": datetime.now().isoformat(),
    }

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    logger.debug(
        f"Updated checkpoint with {len(dashboard['completed_combinations'])} completed combinations"
    )


def load_checkpoint(save_dir: Path) -> tuple[set[str], bool]:
    """Load the checkpoint file to resume an experiment.

    Args:
        save_dir: Directory where results are saved

    Returns:
        Tuple of (set of completed combinations, whether checkpoint exists)
    """
    checkpoint_path = save_dir / "checkpoint.json"

    if not checkpoint_path.exists():
        logger.info("No checkpoint file found. Starting fresh experiment.")
        return set(), False

    try:
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        completed_combinations = set(checkpoint_data.get("completed_combinations", []))
        last_updated = checkpoint_data.get("last_updated", "Unknown")

        # Convert to datetime for display
        try:
            last_update_time = datetime.fromisoformat(last_updated)
            last_updated_str = last_update_time.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            last_updated_str = last_updated

        logger.info(
            f"Loaded checkpoint from {last_updated_str} with {len(completed_combinations)} completed combinations"
        )
        return completed_combinations, True

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return set(), False


def should_skip_combination(scenario_id: str, model: str, completed_combinations: set[str]) -> bool:
    """Check if a scenario-model combination should be skipped (already processed).

    Args:
        scenario_id: ID of the scenario
        model: Name of the model
        completed_combinations: Set of completed scenario-model combinations

    Returns:
        True if the combination should be skipped, False otherwise
    """
    combination = f"{scenario_id}_{model}"
    return combination in completed_combinations
