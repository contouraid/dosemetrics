#!/bin/bash

# Upload anonymized data to HuggingFace datasets
# 
# Prerequisites:
#   1. Install huggingface_hub: pip install huggingface_hub
#   2. Login to HF: huggingface-cli login
#   3. Create dataset repo: https://huggingface.co/new-dataset
#      Suggested name: contouraid/dosemetrics-examples

set -e

echo "================================================================"
echo "DoseMetrics - Upload Anonymized Data to HuggingFace"
echo "================================================================"
echo ""

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed"
    echo "Please run: pip install huggingface_hub"
    exit 1
fi

# Check if user is logged in
if ! huggingface-cli whoami &>/dev/null; then
    echo "ERROR: Not logged in to HuggingFace"
    echo "Please run: huggingface-cli login"
    exit 1
fi

# Set variables
REPO_ID="${1:-contouraid/dosemetrics-examples}"
DATA_DIR="data_anonymized"

echo "Repository: $REPO_ID"
echo "Data directory: $DATA_DIR"
echo ""

# Verify data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Please run the anonymization script first:"
    echo "  python scripts/anonymize_data.py"
    exit 1
fi

# Check if README exists
if [ ! -f "$DATA_DIR/README.md" ]; then
    echo "WARNING: README.md not found in $DATA_DIR"
fi

# Create the dataset card as README.md for HuggingFace
if [ -f "$DATA_DIR/.dataset_card.md" ]; then
    echo "Using .dataset_card.md as README.md for HuggingFace..."
    cp "$DATA_DIR/.dataset_card.md" "$DATA_DIR/README_HF.md"
fi

echo ""
echo "================================================================"
echo "Upload Steps:"
echo "================================================================"
echo ""
echo "Option 1: Using Python API"
echo "--------------------------"
echo ""
cat << 'EOF'
from huggingface_hub import HfApi, create_repo

# Create API instance
api = HfApi()

# Create the dataset repository (if it doesn't exist)
repo_id = "contouraid/dosemetrics-examples"
try:
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    print(f"✓ Repository created/verified: {repo_id}")
except Exception as e:
    print(f"Note: {e}")

# Upload the entire directory
api.upload_folder(
    folder_path="data_anonymized",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload anonymized dosemetrics example data"
)

print(f"✓ Upload complete!")
print(f"  View at: https://huggingface.co/datasets/{repo_id}")
EOF

echo ""
echo ""
echo "Option 2: Using CLI"
echo "-------------------"
echo ""
echo "# Create the repository (web UI recommended):"
echo "  https://huggingface.co/new-dataset"
echo ""
echo "# Upload files:"
echo "  huggingface-cli upload $REPO_ID $DATA_DIR . --repo-type=dataset"
echo ""
echo ""
echo "Option 3: Using Git"
echo "-------------------"
echo ""
echo "# Clone the repository:"
echo "  git clone https://huggingface.co/datasets/$REPO_ID"
echo "  cd dosemetrics-examples"
echo ""
echo "# Copy and commit files:"
echo "  cp -r ../$DATA_DIR/* ."
echo "  git add ."
echo "  git commit -m \"Add anonymized example data\""
echo "  git push"
echo ""
echo "================================================================"
echo ""

# Offer to run the upload
read -p "Would you like to upload now using Python API? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Uploading to HuggingFace..."
    echo ""
    
    python << EOF
from huggingface_hub import HfApi, create_repo
import sys

repo_id = "$REPO_ID"
data_dir = "$DATA_DIR"

try:
    # Create API instance
    api = HfApi()
    
    # Try to create repository
    print(f"Creating/verifying repository: {repo_id}")
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"✓ Repository ready: {repo_id}")
    except Exception as e:
        print(f"Repository status: {e}")
    
    # Upload the folder
    print(f"\nUploading {data_dir}...")
    api.upload_folder(
        folder_path=data_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload anonymized dosemetrics example data",
        ignore_patterns=["__pycache__", "*.pyc", ".DS_Store"]
    )
    
    print(f"\n✅ Upload complete!")
    print(f"   View at: https://huggingface.co/datasets/{repo_id}")
    
except Exception as e:
    print(f"\n❌ Error during upload: {e}", file=sys.stderr)
    print(f"\nYou may need to:", file=sys.stderr)
    print(f"  1. Create the repository manually at:", file=sys.stderr)
    print(f"     https://huggingface.co/new-dataset", file=sys.stderr)
    print(f"  2. Ensure you have write access to: {repo_id}", file=sys.stderr)
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        echo ""
        echo "================================================================"
        echo "✅ Success! Dataset uploaded to HuggingFace"
        echo "================================================================"
        echo ""
        echo "Next steps:"
        echo "  1. Visit: https://huggingface.co/datasets/$REPO_ID"
        echo "  2. Update dataset card if needed"
        echo "  3. Test loading:"
        echo ""
        echo "     from huggingface_hub import snapshot_download"
        echo "     data_path = snapshot_download(repo_id='$REPO_ID', repo_type='dataset')"
        echo ""
    else
        echo ""
        echo "Upload failed. Please try manually or use the CLI method above."
    fi
else
    echo ""
    echo "Upload cancelled. Use one of the methods above when ready."
fi
