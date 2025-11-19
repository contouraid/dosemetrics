# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for medical imaging libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY LICENSE ./
COPY README.md ./
COPY setup_repo.sh ./
COPY src/ ./src/
COPY data/ ./data/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Expose Streamlit default port
EXPOSE 7860

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create .streamlit directory and config
RUN mkdir -p /root/.streamlit
RUN echo "\
    [server]\n\
    headless = true\n\
    port = 7860\n\
    address = \"0.0.0.0\"\n\
    enableCORS = false\n\
    enableXsrfProtection = false\n\
    \n\
    [browser]\n\
    gatherUsageStats = false\n\
    " > /root/.streamlit/config.toml

# Run the Streamlit app
CMD ["streamlit", "run", "src/dosemetrics_app/app.py"]
