#!/bin/zsh

# Exit immediately if a command exits with a non-zero status
set -e

# Ensure the repository is set up by running the setup script
DIR="$(cd "$(dirname "$0")" && pwd)"
"$DIR/setup_repo.sh"

# Activate the virtual environment
source .venv/bin/activate

# Check if Streamlit is installed, if not, install it
if ! pip show streamlit > /dev/null 2>&1; then
    echo "Streamlit not found. Installing Streamlit..."
    uv pip install streamlit
fi

# Run the Streamlit app
echo "Starting the Streamlit app..."
streamlit run src/dosemetrics_app/app.py