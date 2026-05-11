import aiosqlite
import os
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from src.utils.logger import logger
from src.utils.config import DB_PATH

# Encryption setup
ENCRYPTION_KEY = os.getenv("TOKEN_ENCRYPTION_KEY")
cipher = Fernet(ENCRYPTION_KEY.encode()) if ENCRYPTION_KEY else None

async def init_db():
    """Initialize the SQLite database and ensure directory exists."""
    abs_path = os.path.abspath(DB_PATH)
    logger.info(f"Initializing database at: {abs_path}")
    
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    
    # Set secure permissions for the DB file directory
    if os.name != 'nt':
        os.chmod(os.path.dirname(abs_path), 0o700)
    
    async with aiosqlite.connect(abs_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY,
                access_token BLOB,
                issued_at TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        await db.commit()
    
    # Ensure DB file itself is protected
    if os.name != 'nt' and os.path.exists(abs_path):
        os.chmod(abs_path, 0o600)
    
    logger.info("Database initialized and secured")

async def save_token(access_token: str):
    """Encrypt and save the access token to the database."""
    if not cipher:
        raise ValueError("TOKEN_ENCRYPTION_KEY not set")
        
    abs_path = os.path.abspath(DB_PATH)
    encrypted_token = cipher.encrypt(access_token.encode())
    issued_at = datetime.now()
    expires_at = issued_at + timedelta(days=1)
    
    async with aiosqlite.connect(abs_path) as db:
        # Clear old tokens first
        await db.execute("DELETE FROM tokens")
        await db.execute(
            "INSERT INTO tokens (access_token, issued_at, expires_at) VALUES (?, ?, ?)",
            (encrypted_token, issued_at, expires_at)
        )
        await db.commit()
    logger.info("Token encrypted and saved successfully")

async def get_valid_token() -> str | None:
    """Retrieve and decrypt the access token if it hasn't expired."""
    if not cipher:
        return None
        
    abs_path = os.path.abspath(DB_PATH)
    async with aiosqlite.connect(abs_path) as db:
        async with db.execute(
            "SELECT access_token, expires_at FROM tokens ORDER BY id DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                encrypted_token, expires_at_str = row
                expires_at = datetime.fromisoformat(expires_at_str)
                if expires_at > datetime.now():
                    try:
                        decrypted_token = cipher.decrypt(encrypted_token).decode()
                        return decrypted_token
                    except Exception as e:
                        logger.error(f"Decryption failed: {type(e).__name__}")
                        return None
    return None
