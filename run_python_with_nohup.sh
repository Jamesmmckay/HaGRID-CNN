#!/bin/bash

# Name of the Python script to execute
PYTHON_SCRIPT=$1

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: $PYTHON_SCRIPT not found in the current directory."
  exit 1
fi

# Name of the tmux session
TMUX_SESSION=$2

# Start a new tmux session and run the script
echo "Starting $PYTHON_SCRIPT in a new tmux session ($TMUX_SESSION)..."
tmux new-session -d -s "$TMUX_SESSION" "/mnt/c/Users/james/AppData/Local/Microsoft/WindowsApps/python3.exe \"$PYTHON_SCRIPT\" > ./log/output_file.log 2>&1; exec bash"


# Inform the user
echo "Python script is running in the background in tmux session: $TMUX_SESSION"
echo "Use 'tmux attach-session -t $TMUX_SESSION' to reattach to the session."

# Optional: List active tmux sessions
echo "Active tmux sessions:"
tmux list-sessions
