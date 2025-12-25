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

# Ensure SimpleITK is available (required by data I/O)
if ! python -c "import SimpleITK" >/dev/null 2>&1; then
    echo "SimpleITK not found in environment, attempting to install via pip..."
    python3 -m pip install --no-input SimpleITK || echo "Warning: failed to install SimpleITK; tests may fail"
fi

# Ensure testing dependencies are installed
echo "Installing test dependencies..."
python3 -m pip install --no-input pytest pytest-cov nbformat nbconvert ipykernel huggingface_hub || echo "Warning: some test dependencies may not be installed"

echo "Running tests with pytest..."
python3 -m pytest tests/ -v --tb=short

# Also run unittest discovery for backward compatibility
echo ""
echo "Running additional unittest discovery..."
python3 -m unittest discover -s tests -p "test_*.py"