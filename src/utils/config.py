import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if accessible
try:
    load_dotenv()
except Exception:
    # If .env is not accessible (e.g. strict permissions in Docker), 
    # we rely on environment variables already set by the host/Docker
    pass

def get_env_var(key: str, default: str = "") -> str:
    """Retrieve an environment variable or return a default value."""
    return os.getenv(key, default)

def get_required_env_var(key: str) -> str:
    """Retrieve a required environment variable or raise an error if missing."""
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value

# API Config
UPSTOX_API_KEY = get_env_var("UPSTOX_API_KEY")
UPSTOX_API_SECRET = get_env_var("UPSTOX_API_SECRET")
UPSTOX_REDIRECT_URI = get_env_var("UPSTOX_REDIRECT_URI")

# App Config
LOG_LEVEL = get_env_var("LOG_LEVEL", "INFO")
DB_PATH = get_env_var("DB_PATH", "data/trading_bot.db")
