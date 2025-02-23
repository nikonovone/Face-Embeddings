# Makefile for Face Embeddings Project

# Basic settings
PYTHON := python3.12
UV := uv
PROJECT_NAME := Face Embeddings

# Paths
SRC_DIR := src
VENV_DIR := .venv

# Environment variables
export PYTHONPATH := $(PWD)

.PHONY: setup train inference clean

# Setup virtual environment and install dependencies
setup:
	@echo "Creating virtual environment and installing dependencies..."
	$(UV) venv $(VENV_DIR) --python=$(PYTHON)
	. $(VENV_DIR)/bin/activate && \
	$(UV) pip install -e .
	@echo "Environment setup complete!"
	@echo "Activate it with: source $(VENV_DIR)/bin/activate"

# Training pipeline
train:
	@echo "Starting model training..."
	. $(VENV_DIR)/bin/activate && \
	$(PYTHON) -m $(SRC_DIR).train

# Inference/evaluation pipeline
inference:
	@echo "Running inference..."
	. $(VENV_DIR)/bin/activate && \
	$(PYTHON) -m $(SRC_DIR).inference

# Clean up
clean:
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "Cleaned up virtual environment and cache files"

# Show help
help:
	@echo "Available commands:"
	@echo "  make setup      - Create virtual environment and install dependencies"
	@echo "  make train      - Run model training"
	@echo "  make inference  - Run model inference"
	@echo "  make clean      - Remove virtual environment and cache files"