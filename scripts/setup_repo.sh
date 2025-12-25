#!/bin/zsh

# Exit immediately if a command exits with a non-zero status
set -e

# Define the virtual environment directory
VENV_DIR=".venv"

# Check if the virtual environment already exists
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists in $VENV_DIR."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Use uv pip explicitly
UV_PIP="uv pip"

# Install development dependencies using uv
echo "Installing development dependencies using uv..."
if [[ -f "uv.lock" ]]; then
    uv sync
else
    echo "uv.lock not found. Please ensure it exists in the repository."
    exit 1
fi

# Print success message
echo "Repository setup complete. Virtual environment is ready."