# Deployment Guide for Hugging Face Space

This guide explains how to deploy the DoseMetrics Streamlit app to Hugging Face Space.

## Prerequisites

1. **Hugging Face Account**: Create an account at [huggingface.co](https://huggingface.co)
2. **Git Authentication**: Set up git credentials for Hugging Face
   ```bash
   # Install Hugging Face CLI
   pip install huggingface_hub[cli]
   
   # Login to Hugging Face
   hf auth login
   ```
   
   Note: The CLI command is `hf` (not `huggingface-cli`). Use `hf auth login` to authenticate.

## Files Created for Deployment

1. **Dockerfile**: Defines the Docker container configuration for running the Streamlit app
2. **HF_README.md**: README with Hugging Face metadata (YAML frontmatter) and app description
3. **.dockerignore**: Specifies files to exclude from Docker build context
4. **scripts/deploy_to_hf.sh**: Automated deployment script

## Deployment Steps

### Quick Start with Makefile (Recommended)

```bash
# Show all available commands
make help

# Deploy to Hugging Face Space
make deploy
```

### Option 1: Using the Deployment Script

Run the deployment script:

```bash
make deploy
# or directly:
./scripts/deploy_to_hf.sh
```

The script will:
1. Clone your Hugging Face Space repository
2. Copy only necessary files (respecting .gitignore patterns)
3. Commit and push changes to Hugging Face
4. Clean up temporary files

**Important**: The script only copies files that should be deployed:
- Source code (`src/`)
- Test data files (not full datasets - these are excluded via .gitignore)
- Configuration files (`pyproject.toml`, `LICENSE`, etc.)
- Docker configuration (`Dockerfile`)

Large data files and development artifacts are automatically excluded.

### Option 2: Manual Deployment

1. Clone your Hugging Face Space:
   ```bash
   git clone https://huggingface.co/spaces/amithjkamath/dosemetrics
   cd dosemetrics
   ```

2. Copy necessary files:
   ```bash
   cp -r /path/to/dosemetrics/src .
   cp -r /path/to/dosemetrics/data .
   cp /path/to/dosemetrics/Dockerfile .
   cp /path/to/dosemetrics/HF_README.md README.md
   cp /path/to/dosemetrics/pyproject.toml .
   cp /path/to/dosemetrics/LICENSE .
   cp /path/to/dosemetrics/scripts/setup_repo.sh .
   ```

3. Commit and push:
   ```bash
   git add .
   git commit -m "Deploy DoseMetrics app"
   git push
   ```

## Makefile Commands

The project includes a Makefile with common commands:

```bash
make help       # Show all available commands
make setup      # Initial setup and install dependencies
make test       # Run all tests
make run        # Run the Streamlit app locally
make deploy     # Deploy to Hugging Face Space
make clean      # Clean up cache and temporary files
make format     # Format code with black
make lint       # Run linting checks
make check      # Run all checks (format + lint + test)
make info       # Show project information
make status     # Show git status
```

## File Management

### What Gets Deployed

The deployment script respects your `.gitignore` file and only copies:
- Source code (`src/dosemetrics*`)
- Small test data files (`data/test_subject.txt`, `data/visualization/`)
- Configuration (`pyproject.toml`, `LICENSE`, `setup_repo.sh`)
- Docker setup (`Dockerfile`)
- README (`HF_README.md` → `README.md`)

### What Gets Excluded

Following `.gitignore`, these are automatically excluded:
- Large data files (`/data/` - except test files)
- Results directory (`/results/`)
- Local development files (`/local/`)
- Python cache (`__pycache__/`, `*.pyc`)
- Virtual environments (`.venv/`, `venv/`)
- IDE files (`.vscode/`, `.idea/`)
- Test artifacts (`.pytest_cache/`, `.coverage`)

## Verification

After deployment:

1. Visit your Space: https://huggingface.co/spaces/amithjkamath/dosemetrics
2. Wait for the Docker build to complete (may take 5-10 minutes)
3. Test the app by uploading sample dose and mask files

## Troubleshooting

### Build Failures

- Check the build logs in the Hugging Face Space UI
- Ensure all dependencies in `pyproject.toml` are compatible
- Verify Dockerfile syntax

### Runtime Errors

- Check the app logs in the Space UI
- Test locally first: `docker build -t dosemetrics . && docker run -p 7860:7860 dosemetrics`

### Memory Issues

If the app runs out of memory:
1. Request an upgrade to a larger Space tier in Hugging Face settings
2. Optimize data loading in the app
3. Add pagination or lazy loading for large datasets

## Configuration

### Port Configuration

The Dockerfile is configured to use port 7860 (Hugging Face default):
```dockerfile
ENV STREAMLIT_SERVER_PORT=7860
EXPOSE 7860
```

### Streamlit Settings

Settings are configured in the Dockerfile:
- Headless mode enabled
- CORS disabled for Hugging Face compatibility
- Usage stats collection disabled

## Updating the Deployment

To update your deployment:

1. Make changes to your local code
2. Run tests: `./scripts/run_tests.sh` (run `make setup` first if you haven't installed dependencies; `make setup` prefers `uv` when available)
3. Re-run deployment: `./scripts/deploy_to_hf.sh`

## Data Considerations

- The `data/` folder includes test data for demonstration
- For production, consider using Hugging Face Datasets for large data files
- Sensitive patient data should NOT be included in the deployment

## Security Notes

- The authentication is currently disabled in `app.py` for public access
- To enable authentication, uncomment the `check_password()` section in `app.py`
- Add secrets via Hugging Face Space settings (not in the code!)

## Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Docker Spaces Documentation](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Streamlit Documentation](https://docs.streamlit.io/)
