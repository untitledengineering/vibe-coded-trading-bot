from datetime import datetime, timedelta

import aiosqlite
import pytest

from src.db.database import (
    save_token,
    get_valid_token,
    get_token_info,
    clear_tokens,
    init_db,
)

@pytest.fixture(autouse=True)
async def setup_db(mocker, tmp_path):
    """Each test gets its own DB file. We monkeypatch DB_PATH everywhere it is
    imported so production data/trading_bot.db is never touched by the suite."""
    test_db_path = str(tmp_path / "trading_bot.db")
    # Patch in every module that captured DB_PATH at import time.
    mocker.patch("src.db.database.DB_PATH", test_db_path)
    mocker.patch("src.utils.config.DB_PATH", test_db_path)

    # Dummy encryption key for tests (fixed so values stay stable across runs).
    mocker.patch(
        "src.db.database.ENCRYPTION_KEY",
        "N9oS-KrI_mc9E2SW-iPlQ8H_suAc3j63ttOBxrw8Lc4=",
    )
    from cryptography.fernet import Fernet
    mocker.patch(
        "src.db.database.cipher",
        Fernet("N9oS-KrI_mc9E2SW-iPlQ8H_suAc3j63ttOBxrw8Lc4=".encode()),
    )

    await init_db()
    yield test_db_path
    # tmp_path is cleaned up by pytest automatically.

@pytest.mark.asyncio
async def test_token_save_and_retrieve():
    """Verify that a saved token can be retrieved."""
    await save_token("test_access_token")
    token = await get_valid_token()
    assert token == "test_access_token"

@pytest.mark.asyncio
async def test_token_expiry(setup_db):
    """Verify that an expired token is not returned."""
    from cryptography.fernet import Fernet
    cipher = Fernet("N9oS-KrI_mc9E2SW-iPlQ8H_suAc3j63ttOBxrw8Lc4=".encode())

    issued_at = datetime.now() - timedelta(days=2)
    expires_at = datetime.now() - timedelta(days=1)
    encrypted_token = cipher.encrypt(b"expired_token")

    async with aiosqlite.connect(setup_db) as db:
        await db.execute(
            "INSERT INTO tokens (access_token, issued_at, expires_at) VALUES (?, ?, ?)",
            (encrypted_token, issued_at.isoformat(), expires_at.isoformat()),
        )
        await db.commit()

    token = await get_valid_token()
    assert token is None


@pytest.mark.asyncio
async def test_get_token_info_returns_none_when_empty():
    """No rows → None."""
    info = await get_token_info()
    assert info is None


@pytest.mark.asyncio
async def test_get_token_info_returns_metadata_without_token_value():
    """get_token_info exposes timestamps + validity but never the token bytes."""
    await save_token("a_secret_token")
    info = await get_token_info()
    assert info is not None
    assert info["valid"] is True
    assert info["seconds_until_expiry"] > 0
    assert "issued_at" in info and "expires_at" in info
    # The whole point: metadata, not the secret itself.
    assert "access_token" not in info
    assert "a_secret_token" not in str(info)


@pytest.mark.asyncio
async def test_get_token_info_flags_expired(setup_db):
    """An expired row reports valid=False and 0s remaining."""
    from cryptography.fernet import Fernet
    cipher = Fernet("N9oS-KrI_mc9E2SW-iPlQ8H_suAc3j63ttOBxrw8Lc4=".encode())
    issued_at = datetime.now() - timedelta(days=2)
    expires_at = datetime.now() - timedelta(days=1)
    async with aiosqlite.connect(setup_db) as db:
        await db.execute(
            "INSERT INTO tokens (access_token, issued_at, expires_at) VALUES (?, ?, ?)",
            (cipher.encrypt(b"x"), issued_at.isoformat(), expires_at.isoformat()),
        )
        await db.commit()
    info = await get_token_info()
    assert info is not None
    assert info["valid"] is False
    assert info["seconds_until_expiry"] == 0


def test_next_upstox_expiry_before_330_returns_today_330():
    """A token issued at 02:00 IST dies at 03:30 IST same morning."""
    from src.db.database import _next_upstox_expiry
    now = datetime(2026, 5, 15, 2, 0, 0)
    expiry = _next_upstox_expiry(now)
    assert expiry == datetime(2026, 5, 15, 3, 30, 0)
    # ~1.5h, not 24h.
    assert (expiry - now).total_seconds() == pytest.approx(1.5 * 3600)


def test_next_upstox_expiry_after_330_returns_tomorrow_330():
    """A token issued at 10:00 IST is valid until 03:30 IST the next morning."""
    from src.db.database import _next_upstox_expiry
    now = datetime(2026, 5, 15, 10, 0, 0)
    expiry = _next_upstox_expiry(now)
    assert expiry == datetime(2026, 5, 16, 3, 30, 0)


def test_next_upstox_expiry_at_exactly_330_rolls_forward():
    """Edge case: issuing at 03:30:00 sharp returns the next day's 03:30."""
    from src.db.database import _next_upstox_expiry
    now = datetime(2026, 5, 15, 3, 30, 0)
    expiry = _next_upstox_expiry(now)
    assert expiry == datetime(2026, 5, 16, 3, 30, 0)


@pytest.mark.asyncio
async def test_clear_tokens_removes_all_rows():
    """clear_tokens leaves the table empty and get_valid_token returns None."""
    await save_token("doomed_token")
    assert await get_valid_token() == "doomed_token"
    await clear_tokens()
    assert await get_valid_token() is None
    assert await get_token_info() is None
