#!/bin/bash
# Continuous monitoring script for LLM Ethics Experiment

# Default values
INTERVAL=5
VERBOSE=false
OUTPUT_DIR="results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --interval|-i)
      INTERVAL="$2"
      shift 2
      ;;
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    --output-dir|-o)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--interval N] [--verbose] [--output-dir DIR]"
      exit 1
      ;;
  esac
done

# Check if watch command is available
if ! command -v watch &> /dev/null; then
    echo "Error: 'watch' command not found."
    echo "For macOS users, install it using brew: brew install watch"
    echo "For Linux users, it should be pre-installed or available in your package manager."
    
    # Fall back to manual polling with clear and sleep
    echo "Falling back to manual polling mode. Press Ctrl+C to exit."
    while true; do
        clear
        if $VERBOSE; then
            python scripts/check_progress.py --verbose --results-dir "$OUTPUT_DIR"
        else
            python scripts/check_progress.py --results-dir "$OUTPUT_DIR"
        fi
        echo ""
        echo "Last updated: $(date). Refreshing every ${INTERVAL} seconds. Press Ctrl+C to exit."
        sleep $INTERVAL
    done
    exit 0
fi

# Build the command
CMD="python scripts/check_progress.py --results-dir \"$OUTPUT_DIR\""
if $VERBOSE; then
    CMD="$CMD --verbose"
fi

# Run the watch command
echo "Starting continuous monitoring. Press Ctrl+C to exit."
watch -n $INTERVAL -c "$CMD" 