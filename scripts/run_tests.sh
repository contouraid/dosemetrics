#!/bin/zsh

# Exit immediately if a command exits with a non-zero status
set -e

# Define the virtual environment directory
VENV_DIR=".venv"

# If not already activated, run the project setup which will create the venv and install deps (prefer uv)
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment not activated. Running project setup..."

    DIR="$(cd "$(dirname "$0")" && pwd)"
    # Run setup script which will create and populate the virtual environment
    "$DIR/setup_repo.sh"

    # Activate the virtual environment in this shell
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment already activated."
fi

# Discover and run all unittests

# Ensure src/ is on PYTHONPATH so tests can import package
export PYTHONPATH="$PWD/src"

echo "Running unittests..."
python3 -m unittest discover -s tests -p "test_*.py"