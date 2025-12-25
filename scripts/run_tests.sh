#!/bin/zsh

# Exit immediately if a command exits with a non-zero status
set -e

# Define the virtual environment directory
VENV_DIR=".venv"

# Check if the virtual environment is activated, if not, set it up
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment not activated. Setting it up..."

    # Create the virtual environment if it doesn't exist
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "Creating virtual environment in $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    fi

    # Activate the virtual environment
    source "$VENV_DIR/bin/activate"

    # Install dependencies if pyproject.toml exists
    if [[ -f "pyproject.toml" ]]; then
        echo "Installing dependencies from pyproject.toml..."
        if command -v poetry > /dev/null; then
            pip install -r <(poetry export --without-hashes --format=requirements.txt)
        else
            echo "poetry not found, installing package in editable mode"
            pip install -e .
        fi
    fi
else
    echo "Virtual environment already activated."
fi

# Discover and run all unittests

# Ensure src/ is on PYTHONPATH so tests can import package
export PYTHONPATH="$PWD/src"

echo "Running unittests..."
python3 -m unittest discover -s tests -p "test_*.py"