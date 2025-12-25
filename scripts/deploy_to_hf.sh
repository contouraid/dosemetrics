#!/bin/zsh

# Exit immediately if a command exits with a non-zero status
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
HF_SPACE="amithjkamath/dosemetrics"
HF_SPACE_URL="https://huggingface.co/spaces/${HF_SPACE}"

echo "${GREEN}=== Deploying dosemetrics to Hugging Face Space ===${NC}"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "${RED}Error: git is not installed${NC}"
    exit 1
fi

# Check if hf CLI is installed
if ! command -v hf &> /dev/null; then
    echo "${YELLOW}Warning: hf CLI is not installed${NC}"
    echo "Install it with: pip install huggingface_hub[cli]"
    echo "Then login with: hf auth login"
fi

# Create a temporary directory for deployment
DEPLOY_DIR=$(mktemp -d)
echo "Using temporary directory: ${DEPLOY_DIR}"

# Clone the Hugging Face Space repository
echo "${GREEN}Cloning Hugging Face Space repository...${NC}"
# Skip LFS files during clone since we don't have large binary files
GIT_LFS_SKIP_SMUDGE=1 git clone "https://huggingface.co/spaces/${HF_SPACE}" "${DEPLOY_DIR}"

cd "${DEPLOY_DIR}"

# Disable LFS for this repository since we don't need it
git lfs uninstall 2>/dev/null || true
git config --local filter.lfs.process ""
git config --local filter.lfs.required false

# Remove any LFS tracking from .gitattributes if it exists
if [ -f ".gitattributes" ]; then
    # Remove LFS-related lines
    grep -v "filter=lfs" .gitattributes > .gitattributes.tmp 2>/dev/null || true
    mv .gitattributes.tmp .gitattributes 2>/dev/null || true
    # Remove the file if it's now empty
    if [ ! -s ".gitattributes" ]; then
        rm .gitattributes
    fi
fi

# Copy necessary files from the current repository
echo "${GREEN}Copying project files...${NC}"

# Copy source code
if [ -d "${OLDPWD}/src" ]; then
    cp -r "${OLDPWD}/src" .
    echo "  ✓ src/"
fi

# Copy test data (but not all data - respect .gitignore)
if [ -d "${OLDPWD}/data" ]; then
    # Only copy data/ if it exists, but exclude large datasets and binary files
    # This respects the .gitignore which ignores /data/
    # For HF Space, we'll only include text data files, NOT binary medical imaging files
    mkdir -p data
    if [ -f "${OLDPWD}/data/test_subject.txt" ]; then
        cp "${OLDPWD}/data/test_subject.txt" data/
    fi
    # Skip visualization data - it contains large binary .nii.gz files
    # that HF Spaces doesn't allow without Git LFS/XET
    echo "  ✓ data/ (test text files only - skipping binary medical imaging files)"
fi

# Copy configuration files
cp "${OLDPWD}/pyproject.toml" .
echo "  ✓ pyproject.toml"

cp "${OLDPWD}/LICENSE" .
echo "  ✓ LICENSE"

if [ -f "${OLDPWD}/scripts/setup_repo.sh" ]; then
    cp "${OLDPWD}/scripts/setup_repo.sh" .
    echo "  ✓ setup_repo.sh"
fi

cp "${OLDPWD}/Dockerfile" .
echo "  ✓ Dockerfile"

# Use the Hugging Face specific README
if [ -f "${OLDPWD}/HF_README.md" ]; then
    echo "${GREEN}Using Hugging Face README...${NC}"
    cp "${OLDPWD}/HF_README.md" README.md
    echo "  ✓ README.md (from HF_README.md)"
fi

# Create .gitignore for HF Space (simpler than main repo)
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
.venv/
venv/
ENV/
.streamlit/secrets.toml
*.log
.pytest_cache/
.coverage
*.ipynb_checkpoints

# Binary medical imaging files (not allowed on HF Spaces without XET)
*.nii
*.nii.gz
*.dcm
*.nrrd
data/visualization/
EOF
echo "  ✓ .gitignore"

# Add all files
echo "${GREEN}Staging files for commit...${NC}"
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "${YELLOW}No changes to deploy${NC}"
else
    # Commit changes
    echo "${GREEN}Committing changes...${NC}"
    git commit -m "Deploy dosemetrics app - $(date '+%Y-%m-%d %H:%M:%S')"

    # Push to Hugging Face
    echo "${GREEN}Pushing to Hugging Face Space...${NC}"
    git push

    echo "${GREEN}=== Deployment complete! ===${NC}"
    echo "Your app is deploying at: ${HF_SPACE_URL}"
    echo "It may take a few minutes for the changes to be reflected."
fi

# Clean up
cd "${OLDPWD}"
rm -rf "${DEPLOY_DIR}"
echo "${GREEN}Cleaned up temporary files${NC}"
