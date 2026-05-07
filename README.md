# Upstox Trading Research

A paper trading bot for Indian markets focusing on live market data streaming and strategy research.

## Setup

### Prerequisites
- Python 3.11+
- Upstox API Credentials (set in `.env`)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install development dependencies and setup pre-commit hooks:
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```

### Usage
Run the live market data feed:
```bash
PYTHONPATH=. python src/main.py
```

## Security
- No API keys are logged or committed.
- `.env` is ignored by git.
- Pre-commit hooks check for secrets and private keys.
