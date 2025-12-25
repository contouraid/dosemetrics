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

# Prefer using uv (universe) when available, otherwise fall back to pip
if command -v uv > /dev/null 2>&1 && [[ -f "uv.lock" ]]; then
    echo "Found uv and uv.lock, syncing dependencies with uv..."
    uv sync --no-build-isolation || uv sync || true
else
    if ! command -v uv > /dev/null 2>&1; then
        echo "uv not found; falling back to pip-based install in the venv"
    else
        echo "uv.lock not found; falling back to pip-based install in the venv"
    fi

    # Upgrade pip and setuptools in the venv to reduce editable install issues
    echo "Upgrading pip, setuptools, and wheel in the virtual environment..."
    python3 -m pip install --upgrade pip setuptools wheel

    # Try installing the package into the venv (non-fatal if it fails)
    echo "Installing the package into the virtual environment..."
    if python3 -m pip install -e .; then
        echo "Installed package in editable mode."
    else
        echo "Editable install failed, attempting standard install..."
        python3 -m pip install . || true
    fi
fi

# Print success message
echo "Repository setup complete. Virtual environment is ready."