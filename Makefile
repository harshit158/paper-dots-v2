# PaperDots — developer tasks
#
# `make dev` runs the Streamlit app.
# `make help` lists everything else.

UV      ?= uv
APP     := src/app.py
UI_PORT ?= 8501
DB_PATH ?= data/paperdots.db

STREAMLIT := $(UV) run streamlit run $(APP) --server.port $(UI_PORT)

.DEFAULT_GOAL := help

.PHONY: help install dev test lint format check reset clean

help: ## Show this help
	@echo "PaperDots targets:"
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install dependencies
	$(UV) sync

dev: ## Run the Streamlit app (Ctrl-C stops it)
	@echo "UI -> http://localhost:$(UI_PORT)"
	@$(STREAMLIT)

test: ## Run the test suite
	$(UV) run pytest

lint: ## Lint and auto-fix
	$(UV) run ruff check . --fix

format: ## Format the code
	$(UV) run ruff format .

check: ## Verify lint, formatting and tests (no writes)
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run pytest

reset: ## Delete the local database (start with an empty library)
	rm -f $(DB_PATH)
	@echo "Removed $(DB_PATH)"

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
