# Grail

GRAIL – Guaranteed Rollout Authenticity via Inference Ledger

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Create a venv
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install GRAIL
uv pip install -e .
```

## Mining

```bash
# Copy then fill out env items.
cp .env.example .env

# Run miner locally.
grail mine
```

## Validating

```bash
# Copy then fill out env items.
cp .env.example .env

# Run valdiator locally.
grail validate
```
