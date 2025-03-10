#!/bin/bash
# Script to check if the experiment is running and restart monitoring

# Default values
OUTPUT_DIR="test-2"
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir|-o)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--output-dir DIR] [--verbose]"
      exit 1
      ;;
  esac
done

# Check if the dashboard.json file exists
if [ ! -f "$OUTPUT_DIR/dashboard.json" ]; then
    echo "Dashboard file not found at $OUTPUT_DIR/dashboard.json"
    echo "The experiment may not be running or the dashboard hasn't been created yet."
    
    # Check if there are any raw response files
    if [ -d "$OUTPUT_DIR/raw_responses" ] && [ "$(ls -A $OUTPUT_DIR/raw_responses)" ]; then
        echo "Found raw response files in $OUTPUT_DIR/raw_responses"
        echo "The experiment appears to be running, but the dashboard hasn't been created."
        
        # Count the number of files in raw_responses
        FILE_COUNT=$(ls -1 "$OUTPUT_DIR/raw_responses" | wc -l | tr -d ' ')
        echo "Current progress: $FILE_COUNT files generated"
        
        # List the most recent files
        echo "Most recent files:"
        ls -lt "$OUTPUT_DIR/raw_responses" | head -n 5
    else
        echo "No raw response files found. The experiment may not have started yet."
    fi
    
    exit 1
fi

# If we get here, the dashboard.json file exists, so we can run the monitor
echo "Dashboard file found. Starting monitoring..."

# Run the monitor script
if $VERBOSE; then
    ./monitor.sh --output-dir "$OUTPUT_DIR" --verbose
else
    ./monitor.sh --output-dir "$OUTPUT_DIR"
fi 