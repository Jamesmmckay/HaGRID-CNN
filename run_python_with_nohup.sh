#!/bin/bash
#Format for this script is:
#./run_python_with_nohup.sh name_of_script.py output_file.out
# Name of the Python script to execute
PYTHON_SCRIPT=$1

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: $PYTHON_SCRIPT not found in the current directory."
  exit 1
fi

# Check if the output file exists
if [ ! -f "./out/$2" ]; then
  echo "Creating $2 in ./out/"
fi

# Run the Python script using nohup
echo "Starting $PYTHON_SCRIPT with nohup..."
nohup python3 "$PYTHON_SCRIPT" > ./out/$2 2>&1 &

# Capture the process ID of the background script
PID=$!

# Inform the user
echo "Python script is running in the background with PID: $PID"
echo "Output will be written to ./out/$2"

# Optional: Monitor the output file in real-time
echo "Monitoring $2 (press Ctrl+C to stop)..."
tail -f ./out/$2
