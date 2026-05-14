# Makefile for DoseMetrics
.PHONY: help setup run test test-cov test-cov-html deploy clean format lint install check update info app docs docs-serve docs-build docs-test

# Default target
.DEFAULT_GOAL := help

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

help: ## Show this help message
	@echo "$(BLUE)DoseMetrics - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

setup: ## Initial setup - install dependencies and configure environment
	@echo "$(BLUE)🔧 Running setup...$(NC)"
	@chmod +x scripts/setup_repo.sh
	@./scripts/setup_repo.sh
	@echo "$(GREEN)✅ Setup complete$(NC)"

install: ## Install/update dependencies
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	@if command -v uv > /dev/null; then \
		uv sync --no-build-isolation; \
	else \
		pip install -e .; \
	fi
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

run: app ## Run the Streamlit app locally

app: ## Run the Streamlit app locally
	@echo "$(BLUE)🚀 Starting Streamlit app...$(NC)"
	@chmod +x scripts/run_streamlit_app.sh
	@./scripts/run_streamlit_app.sh

test: ## Run all tests
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	@chmod +x scripts/run_tests.sh
	@./scripts/run_tests.sh

test-cov: ## Run tests with coverage and show text report
	@echo "$(BLUE)🧪 Running tests with coverage...$(NC)"
	@if command -v uv > /dev/null; then \
		uv run pytest tests/ --cov=src/dosemetrics --cov-report=term-missing --cov-report=html --ignore=tests/metrics/test_gamma_performance.py; \
	else \
		pytest tests/ --cov=src/dosemetrics --cov-report=term-missing --cov-report=html --ignore=tests/metrics/test_gamma_performance.py; \
	fi
	@echo ""
	@echo "$(GREEN)✅ Coverage report generated$(NC)"
	@echo "$(YELLOW)💡 Text report shown above$(NC)"
	@echo "$(YELLOW)💡 HTML report: open htmlcov/index.html$(NC)"

test-cov-html: ## Generate HTML coverage report and open in browser
	@echo "$(BLUE)🧪 Generating HTML coverage report...$(NC)"
	@if command -v uv > /dev/null; then \
		uv run pytest tests/ --cov=src/dosemetrics --cov-report=html --cov-report=term-missing:skip-covered --ignore=tests/metrics/test_gamma_performance.py; \
	else \
		pytest tests/ --cov=src/dosemetrics --cov-report=html --cov-report=term-missing:skip-covered --ignore=tests/metrics/test_gamma_performance.py; \
	fi
	@echo ""
	@echo "$(GREEN)✅ HTML coverage report generated in htmlcov/$(NC)"
	@echo "$(YELLOW)💡 Opening in browser...$(NC)"
	@open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || echo "$(YELLOW)⚠️  Please open htmlcov/index.html manually$(NC)"

deploy: ## Deploy to Hugging Face Space
	@echo "$(BLUE)🚀 Deploying to Hugging Face Space...$(NC)"
	@chmod +x scripts/deploy_to_hf.sh
	@./scripts/deploy_to_hf.sh

docs-serve: ## Serve documentation locally with live-reload
	@echo "$(BLUE)📚 Starting documentation server...$(NC)"
	@if command -v uv > /dev/null; then \
		uv run mkdocs serve; \
	else \
		mkdocs serve; \
	fi

docs-build: ## Build static documentation site
	@echo "$(BLUE)📚 Building documentation...$(NC)"
	@if command -v uv > /dev/null; then \
		uv run mkdocs build; \
	else \
		mkdocs build; \
	fi
	@echo "$(GREEN)✅ Documentation built in site/$(NC)"

docs-test: ## Test documentation build (same as GitHub Actions)
	@echo "$(BLUE)🧪 Testing documentation build...$(NC)"
	@chmod +x scripts/test_docs_build.sh
	@./scripts/test_docs_build.sh

docs: docs-serve ## Alias for docs-serve

clean: ## Clean up cache and temporary files
	@echo "$(BLUE)🧹 Cleaning up...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ site/ 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

format: ## Format code with black (if installed)
	@echo "$(BLUE)🎨 Formatting code...$(NC)"
	@if command -v uv > /dev/null; then \
		uv run black src/ tests/ examples/ 2>/dev/null || echo "$(YELLOW)⚠️  black not in uv environment$(NC)"; \
		echo "$(GREEN)✅ Code formatted$(NC)"; \
	else \
		if command -v black > /dev/null; then \
			black src/ tests/ examples/; \
			echo "$(GREEN)✅ Code formatted$(NC)"; \
		else \
			echo "$(YELLOW)⚠️  black not installed, skipping$(NC)"; \
		fi; \
	fi

lint: ## Run linting checks with ruff (if installed)
	@echo "$(BLUE)🔍 Running linting...$(NC)"
	@if command -v uv > /dev/null; then \
		uv run ruff check src/ tests/ examples/ 2>/dev/null || echo "$(YELLOW)⚠️  ruff not in uv environment$(NC)"; \
	else \
		if command -v ruff > /dev/null; then \
			ruff check src/ tests/ examples/; \
		else \
			echo "$(YELLOW)⚠️  ruff not installed, skipping$(NC)"; \
		fi; \
	fi

check: format lint test ## Run all checks (format + lint + test)
	@echo "$(GREEN)✅ All checks complete$(NC)"

update: ## Update all dependencies
	@echo "$(BLUE)⬆️  Updating dependencies...$(NC)"
	@if command -v uv > /dev/null; then \
		uv sync --upgrade --no-build-isolation; \
	else \
		pip install --upgrade -e .; \
	fi
	@echo "$(GREEN)✅ Dependencies updated$(NC)"

info: ## Show project information
	@echo "$(BLUE)📊 Project Information$(NC)"
	@echo ""
	@echo "  Name:        DoseMetrics"
	@echo "  Version:     $$(grep '^version' pyproject.toml | cut -d'"' -f2)"
	@echo "  Python:      $$(python3 --version | awk '{print $$2}')"
	@echo "  UV:          $$(uv --version 2>/dev/null || echo 'not installed')"
	@echo "  Repository:  https://github.com/contouraid/dosemetrics"
	@echo "  HF Space:    https://huggingface.co/spaces/contouraid/dosemetrics"
	@echo ""

status: ## Show git and deployment status
	@echo "$(BLUE)📊 Repository Status$(NC)"
	@echo ""
	@echo "Git branch:  $$(git branch --show-current)"
	@echo "Git status:"
	@git status --short
	@echo ""

build-docker: ## Build Docker image locally
	@echo "$(BLUE)🐳 Building Docker image...$(NC)"
	@docker build -t dosemetrics .
	@echo "$(GREEN)✅ Docker image built$(NC)"

run-docker: ## Run Docker container locally
	@echo "$(BLUE)🐳 Running Docker container...$(NC)"
	@docker run -p 7860:7860 dosemetrics

example: ## Run an example script
	@echo "$(BLUE)📝 Available examples:$(NC)"
	@echo ""
	@ls -1 examples/*.py | sed 's/examples\//  - /' | sed 's/.py//'
	@echo ""
	@echo "Run with: python examples/<example_name>.py"
