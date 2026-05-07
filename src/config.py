import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Root
ROOT_DIR = Path(__file__).parent.parent

# API Credentials
API_KEY = os.getenv("UPSTOX_API_KEY")
API_SECRET = os.getenv("UPSTOX_API_SECRET")
REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:8000/callback")

# Session Persistence
TOKEN_FILE = ROOT_DIR / ".token"

def validate_config() -> None:
    """
    Validates that all necessary environment variables are set.
    """
    if not API_KEY or not API_SECRET:
        raise ValueError("UPSTOX_API_KEY and UPSTOX_API_SECRET must be set in .env")
