#!/bin/bash
# Setup script for project environment
# Creates virtual environment, installs dependencies, and starts Docker containers (MinIO and MLFlow)

set -e

echo "Setting up project environment..."

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Install project in editable mode
echo "Installing project in editable mode..."
pip install -e . --quiet

# Install pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit hooks..."
    pre-commit install
else
    echo "Warning: pre-commit not found, skipping hook installation"
fi

# Start Docker containers (MinIO and MLFlow)
echo "Starting Docker containers (MinIO and MLFlow)..."
docker-compose up -d

# Wait for containers to start
echo "Waiting for containers to start..."
sleep 5

# Check container status
echo "Checking container status..."
docker-compose ps

echo ""
echo "Environment setup complete."
echo "Activate virtual environment with: source .venv/bin/activate"
