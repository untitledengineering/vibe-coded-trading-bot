import aiosqlite
import os
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from src.utils.logger import logger
from src.utils.config import DB_PATH

# Encryption setup
ENCRYPTION_KEY = os.getenv("TOKEN_ENCRYPTION_KEY")
cipher = Fernet(ENCRYPTION_KEY.encode()) if ENCRYPTION_KEY else None

# Sentiment columns added to the news table after Sprint 1. ALTER TABLE ADD COLUMN
# is the SQLite migration idiom; we guard each one against existing columns so
# init_db() stays idempotent across upgrades.
_NEWS_SENTIMENT_COLUMNS = (
    ("sentiment_score", "REAL"),
    ("sentiment_confidence", "REAL"),
    ("sentiment_decay_minutes", "INTEGER"),
    ("sentiment_model", "TEXT"),
    ("sentiment_at", "INTEGER"),
)

_PAPER_POSITIONS_EXTRA_COLUMNS = (
    ("entry_sentiment_score", "REAL"),
)


async def _ensure_columns(db, table: str, columns: tuple) -> None:
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        existing = {row[1] for row in await cur.fetchall()}
    for name, sql_type in columns:
        if name not in existing:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


async def _ensure_news_sentiment_columns(db) -> None:
    await _ensure_columns(db, "news", _NEWS_SENTIMENT_COLUMNS)


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
        # Historical 1-min OHLCV bars (backfilled from Upstox V3 historical-candle API).
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bars_1m (
                instrument_key TEXT NOT NULL,
                minute_ts      INTEGER NOT NULL,
                open           REAL,
                high           REAL,
                low            REAL,
                close          REAL,
                volume         INTEGER,
                PRIMARY KEY (instrument_key, minute_ts)
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS ix_bars_1m_ts ON bars_1m (minute_ts)")
        # Live 1-min OHLCV bars (aggregated from the WebSocket tick stream).
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bars_live (
                instrument_key TEXT NOT NULL,
                minute_ts      INTEGER NOT NULL,
                open           REAL,
                high           REAL,
                low            REAL,
                close          REAL,
                volume         INTEGER,
                PRIMARY KEY (instrument_key, minute_ts)
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS ix_bars_live_ts ON bars_live (minute_ts)")
        # News headlines (market-wide + per-ticker). UNIQUE(source, url) for dedup.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                published_at   INTEGER NOT NULL,
                fetched_at     INTEGER NOT NULL,
                source         TEXT NOT NULL,
                headline       TEXT NOT NULL,
                url            TEXT,
                instrument_key TEXT,
                UNIQUE(source, url)
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS ix_news_published ON news (published_at)")
        await db.execute("CREATE INDEX IF NOT EXISTS ix_news_instrument ON news (instrument_key)")
        # Sentiment columns. We add them via ALTER so existing news rows survive.
        await _ensure_news_sentiment_columns(db)
        await db.execute("CREATE INDEX IF NOT EXISTS ix_news_sentiment_at ON news (sentiment_at)")
        # Paper-trading state (Sprint 3). One row per position. UNIQUE
        # (instrument_key, entry_ts) blocks accidental double-writes from a
        # buggy retry path.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument_key      TEXT NOT NULL,
                side                TEXT NOT NULL,
                qty                 INTEGER NOT NULL,
                entry_ts            INTEGER NOT NULL,
                entry_price         REAL NOT NULL,
                stop_loss_price     REAL NOT NULL,
                target_price        REAL NOT NULL,
                exit_ts             INTEGER,
                exit_price          REAL,
                exit_reason         TEXT,
                realised_pnl_inr    REAL,
                predicted_return    REAL,
                model_name          TEXT,
                UNIQUE(instrument_key, entry_ts)
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS ix_paper_positions_entry_ts ON paper_positions (entry_ts)")
        await db.execute("CREATE INDEX IF NOT EXISTS ix_paper_positions_exit_ts  ON paper_positions (exit_ts)")
        await _ensure_columns(db, "paper_positions", _PAPER_POSITIONS_EXTRA_COLUMNS)
        # One row per trading day for halt/resume state.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS paper_session_state (
                session_date    TEXT PRIMARY KEY,
                halted          INTEGER NOT NULL DEFAULT 0,
                halt_reason     TEXT,
                halted_at       INTEGER,
                started_at      INTEGER
            )
        """)
        # ---------- Sprint 7: options trading ----------
        # Per-contract metadata. Refreshed daily from Upstox's options chain
        # endpoint. instrument_key uniquely identifies a contract for trading
        # and historical-data lookups.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS option_contracts (
                instrument_key      TEXT PRIMARY KEY,
                underlying          TEXT NOT NULL,
                underlying_symbol   TEXT,
                expiry_ts           INTEGER NOT NULL,
                expiry_date         TEXT NOT NULL,
                strike_price        REAL NOT NULL,
                instrument_type     TEXT NOT NULL,
                lot_size            INTEGER NOT NULL,
                last_seen_ts        INTEGER,
                is_expired          INTEGER NOT NULL DEFAULT 0
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS ix_oc_underlying_expiry ON option_contracts (underlying, expiry_ts)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS ix_oc_expiry_date ON option_contracts (expiry_date)"
        )
        # 1-min OHLCV per option contract. Same shape as bars_1m for equities,
        # plus open_interest. PRIMARY KEY enforces idempotent backfill.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS option_bars_1m (
                instrument_key  TEXT NOT NULL,
                minute_ts       INTEGER NOT NULL,
                open            REAL,
                high            REAL,
                low             REAL,
                close           REAL,
                volume          INTEGER,
                open_interest   INTEGER,
                PRIMARY KEY (instrument_key, minute_ts)
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS ix_obm_ts ON option_bars_1m (minute_ts)"
        )
        # Two-leg spread positions. The credit, max_loss and margin are derived
        # at entry from the two leg prices + strikes. We persist the leg-level
        # detail so the paper engine can recompute mark-to-market exits without
        # cross-table joins.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS paper_spreads (
                id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                underlying               TEXT NOT NULL,
                spread_type              TEXT NOT NULL,
                expiry_date              TEXT NOT NULL,
                entry_ts                 INTEGER NOT NULL,
                short_leg_key            TEXT NOT NULL,
                short_strike             REAL NOT NULL,
                short_entry_price        REAL NOT NULL,
                long_leg_key             TEXT NOT NULL,
                long_strike              REAL NOT NULL,
                long_entry_price         REAL NOT NULL,
                qty_lots                 INTEGER NOT NULL,
                lot_size                 INTEGER NOT NULL,
                credit_received_per_lot  REAL NOT NULL,
                max_loss_per_lot         REAL NOT NULL,
                stop_loss_pct_of_credit  REAL NOT NULL DEFAULT 2.0,
                target_pct_of_credit     REAL NOT NULL DEFAULT 0.5,
                exit_ts                  INTEGER,
                short_exit_price         REAL,
                long_exit_price          REAL,
                realised_pnl_inr         REAL,
                exit_reason              TEXT,
                UNIQUE(short_leg_key, long_leg_key, entry_ts)
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS ix_ps_entry_ts ON paper_spreads (entry_ts)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS ix_ps_exit_ts ON paper_spreads (exit_ts)"
        )
        await db.commit()
    
    # Ensure DB file itself is protected
    if os.name != 'nt' and os.path.exists(abs_path):
        os.chmod(abs_path, 0o600)
    
    logger.info("Database initialized and secured")

def _next_upstox_expiry(now: datetime) -> datetime:
    """Upstox access tokens expire at 03:30 IST every day, regardless of when issued.
    A token issued at 02:00 IST is dead by 03:30 IST the same morning. We assume
    the host clock is IST (we're running this bot for the Indian market)."""
    boundary = now.replace(hour=3, minute=30, second=0, microsecond=0)
    if now >= boundary:
        boundary = boundary + timedelta(days=1)
    return boundary


async def save_token(access_token: str):
    """Encrypt and save the access token to the database."""
    if not cipher:
        raise ValueError("TOKEN_ENCRYPTION_KEY not set")

    abs_path = os.path.abspath(DB_PATH)
    encrypted_token = cipher.encrypt(access_token.encode())
    issued_at = datetime.now()
    expires_at = _next_upstox_expiry(issued_at)
    
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


async def get_token_info() -> dict | None:
    """Return token metadata (NOT the token value) for the latest row, or None."""
    abs_path = os.path.abspath(DB_PATH)
    async with aiosqlite.connect(abs_path) as db:
        async with db.execute(
            "SELECT issued_at, expires_at FROM tokens ORDER BY id DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            issued_at_str, expires_at_str = row
            expires_at = datetime.fromisoformat(expires_at_str)
            now = datetime.now()
            return {
                "issued_at": issued_at_str,
                "expires_at": expires_at_str,
                "valid": expires_at > now,
                "seconds_until_expiry": max(0, int((expires_at - now).total_seconds())),
            }


async def clear_tokens() -> None:
    """Delete every stored token. Used on logout."""
    abs_path = os.path.abspath(DB_PATH)
    async with aiosqlite.connect(abs_path) as db:
        await db.execute("DELETE FROM tokens")
        await db.commit()
    logger.info("All tokens cleared")
