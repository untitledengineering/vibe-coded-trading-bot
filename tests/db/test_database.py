import pytest
import os
import asyncio
from datetime import datetime, timedelta
import aiosqlite
from src.db.database import save_token, get_valid_token, init_db, DB_PATH

@pytest.fixture(autouse=True)
async def setup_db(mocker):
    """Ensure database is initialized and clean for each test."""
    # Set a dummy encryption key for tests
    mocker.patch("src.db.database.ENCRYPTION_KEY", "N9oS-KrI_mc9E2SW-iPlQ8H_suAc3j63ttOBxrw8Lc4=")
    from cryptography.fernet import Fernet
    mocker.patch("src.db.database.cipher", Fernet("N9oS-KrI_mc9E2SW-iPlQ8H_suAc3j63ttOBxrw8Lc4=".encode()))
    
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    await init_db()
    yield
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

@pytest.mark.asyncio
async def test_token_save_and_retrieve():
    """Verify that a saved token can be retrieved."""
    await save_token("test_access_token")
    token = await get_valid_token()
    assert token == "test_access_token"

@pytest.mark.asyncio
async def test_token_expiry(mocker):
    """Verify that an expired token is not returned."""
    from cryptography.fernet import Fernet
    cipher = Fernet("N9oS-KrI_mc9E2SW-iPlQ8H_suAc3j63ttOBxrw8Lc4=".encode())
    
    issued_at = datetime.now() - timedelta(days=2)
    expires_at = datetime.now() - timedelta(days=1)
    
    # Manually inject an encrypted but expired token
    encrypted_token = cipher.encrypt(b"expired_token")
    
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO tokens (access_token, issued_at, expires_at) VALUES (?, ?, ?)",
            (encrypted_token, issued_at.isoformat(), expires_at.isoformat())
        )
        await db.commit()
        
    token = await get_valid_token()
    assert token is None
