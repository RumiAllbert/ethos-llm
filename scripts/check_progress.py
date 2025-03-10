#!/usr/bin/env python
"""Script to check the progress of a running experiment.

This script loads the dashboard.json file from the specified results directory
and displays the progress of the experiment.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Add the project root to the Python path
import fix_path  # noqa


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Check experiment progress")
    parser.add_argument(
        "--results-dir",
        "-r",
        type=str,
        default="results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Display more detailed information"
    )
    return parser.parse_args()


def load_dashboard(results_dir: str) -> dict[str, Any] | None:
    """Load the dashboard file.

    Args:
        results_dir: Directory containing experiment results

    Returns:
        Dashboard data or None if not found
    """
    dashboard_path = Path(results_dir) / "dashboard.json"

    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        return None

    try:
        with open(dashboard_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dashboard: {e}")
        return None


def format_time_delta(time_str: str) -> str:
    """Format time delta string nicely.

    Args:
        time_str: Time delta as string (HH:MM:SS)

    Returns:
        Formatted time string
    """
    if ":" not in time_str:
        return time_str

    parts = time_str.split(":")
    if len(parts) != 3:
        return time_str

    hours, minutes, seconds = map(int, parts)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def calculate_eta(dashboard: dict[str, Any]) -> tuple[str, float]:
    """Calculate estimated time of completion.

    Args:
        dashboard: Dashboard data

    Returns:
        Tuple of (formatted ETA string, completion percentage)
    """
    if "completed" not in dashboard or "total_combinations" not in dashboard:
        return "Unknown", 0.0

    completed = dashboard.get("completed", 0)
    total = dashboard.get("total_combinations", 1)  # Avoid division by zero

    if total == 0:
        return "Unknown", 0.0

    completion_percentage = (completed / total) * 100

    # If we have an estimated time remaining, use it
    if "estimated_time_remaining" in dashboard:
        eta = format_time_delta(dashboard["estimated_time_remaining"])
        return eta, completion_percentage

    return "Unknown", completion_percentage


def get_model_progress(dashboard: dict[str, Any]) -> list[tuple[str, int, int]]:
    """Get progress by model.

    Args:
        dashboard: Dashboard data

    Returns:
        List of (model name, completed, total) tuples
    """
    models_progress = dashboard.get("models_progress", {})
    total_scenarios = dashboard.get("total_scenarios", 0)

    return [(model, completed, total_scenarios) for model, completed in models_progress.items()]


def display_progress_bar(percentage: float, width: int = 50) -> str:
    """Create a simple ASCII progress bar.

    Args:
        percentage: Completion percentage (0-100)
        width: Width of the progress bar in characters

    Returns:
        Progress bar string
    """
    filled_width = int(width * percentage / 100)
    bar = "█" * filled_width + "░" * (width - filled_width)
    return f"[{bar}] {percentage:.1f}%"


def display_dashboard(dashboard: dict[str, Any], verbose: bool = False) -> None:
    """Display dashboard information.

    Args:
        dashboard: Dashboard data
        verbose: Whether to display verbose information
    """
    # Basic information
    completed = dashboard.get("completed", 0)
    total = dashboard.get("total_combinations", 0)

    # Calculate times
    start_time_str = dashboard.get("start_time", "Unknown")
    last_updated_str = dashboard.get("last_updated", "Unknown")

    try:
        start_time = datetime.fromisoformat(start_time_str)
        last_updated = datetime.fromisoformat(last_updated_str)
        start_time_readable = start_time.strftime("%Y-%m-%d %H:%M:%S")
        last_updated_readable = last_updated.strftime("%Y-%m-%d %H:%M:%S")

        # Calculate time since last update
        now = datetime.now()
        time_since_update = now - last_updated
        time_since_update_str = str(time_since_update).split(".")[0]  # HH:MM:SS format
    except (ValueError, TypeError):
        start_time_readable = start_time_str
        last_updated_readable = last_updated_str
        time_since_update_str = "Unknown"

    # Calculate ETA
    eta, completion_percentage = calculate_eta(dashboard)

    # Create progress bar
    progress_bar = display_progress_bar(completion_percentage)

    # Print summary
    print("\n=== EXPERIMENT PROGRESS ===\n")
    print(f"Progress: {completed}/{total} combinations processed ({completion_percentage:.1f}%)")
    print(progress_bar)
    print(f"Started at: {start_time_readable}")
    print(f"Last updated: {last_updated_readable} ({format_time_delta(time_since_update_str)} ago)")
    print(f"Elapsed time: {format_time_delta(dashboard.get('elapsed_time', 'Unknown'))}")
    print(f"Estimated time remaining: {eta}")

    # Print model progress
    print("\n=== PROGRESS BY MODEL ===\n")
    model_progress = get_model_progress(dashboard)
    for model, completed, total in model_progress:
        if total > 0:
            percentage = (completed / total) * 100
            print(f"{model}: {completed}/{total} ({percentage:.1f}%)")

    # Print additional information if verbose
    if verbose:
        print("\n=== DETAILED INFORMATION ===\n")
        print(f"Total scenarios: {dashboard.get('total_scenarios', 'Unknown')}")
        print(f"Total models: {dashboard.get('total_models', 'Unknown')}")
        print(f"Total frameworks: {dashboard.get('total_frameworks', 'Unknown')}")

        # Print the number of scenarios completed for each source
        scenarios_completed = dashboard.get("scenarios_completed", [])
        scenario_ids_by_source = {}

        for scenario_id in scenarios_completed:
            source = scenario_id.split("_")[0] if "_" in scenario_id else "unknown"
            if source not in scenario_ids_by_source:
                scenario_ids_by_source[source] = []
            scenario_ids_by_source[source].append(scenario_id)

        print("\nScenarios completed by source:")
        for source, ids in scenario_ids_by_source.items():
            print(f"  {source}: {len(ids)}")

    print("\n=== RESUMING INFORMATION ===\n")
    print("To resume this experiment if interrupted, run:")
    print("./run.sh --resume --output-dir [results_dir]")
    print("\nTo check progress again, run:")
    print("python scripts/check_progress.py --results-dir [results_dir]")


def main() -> None:
    """Main function."""
    args = parse_args()

    # Load dashboard
    dashboard = load_dashboard(args.results_dir)

    if dashboard:
        display_dashboard(dashboard, args.verbose)
    else:
        print("No dashboard found or error loading dashboard.")
        print("Make sure the experiment has started and the dashboard.json file exists.")


if __name__ == "__main__":
    main()
