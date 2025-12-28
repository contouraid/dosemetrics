#!/bin/bash
# Local documentation build test script
# Mirrors the GitHub Actions workflow for documentation builds

set -e

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Testing Documentation Build${NC}"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python not found${NC}"
    exit 1
fi

# Set python command
PYTHON=python3

# Check if we're in the right directory
if [ ! -f "mkdocs.yml" ]; then
    echo -e "${RED}❌ mkdocs.yml not found. Please run from repository root.${NC}"
    exit 1
fi

echo -e "${BLUE}📦 Installing dependencies...${NC}"
$PYTHON -m pip install --upgrade pip --quiet
pip install -e ".[docs]" --quiet
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

echo -e "${BLUE}🏗️  Building documentation...${NC}"
# Clean previous build
if [ -d "site" ]; then
    rm -rf site
fi

# Build documentation (this is what GitHub Actions runs)
if mkdocs build; then
    echo ""
    echo -e "${GREEN}✅ Documentation build successful!${NC}"
    echo ""
    echo -e "${YELLOW}📂 Site generated in: ./site/${NC}"
    echo -e "${YELLOW}💡 To preview locally, run: make docs-serve${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}❌ Documentation build failed${NC}"
    echo ""
    echo -e "${YELLOW}💡 Check the error messages above for details${NC}"
    exit 1
fi
