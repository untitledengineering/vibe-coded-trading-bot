import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from src.api import auth, paper as paper_router, stream, streamer_manager
from src.api import market as market_api
from src.api import stock as stock_api
from src.data.news import market_news_loop
from src.db.database import init_db, get_valid_token
from src.features.sentiment import get_default_scorer, score_unscored_news
from src.paper import loop as paper_loop
from src.utils.logger import logger

# Security: Uvicorn access log filter for /callback?code=...
class _StripCallbackQuery(logging.Filter):
    def filter(self, record):
        if hasattr(record, "args") and len(record.args) > 2:
            path = str(record.args[2])
            if "/callback" in path and "code=" in path:
                # Redact the entire query string for security
                record.args = list(record.args)
                record.args[2] = path.split('?')[0] + "?<redacted>"
                record.args = tuple(record.args)
        return True

# Apply filter to uvicorn access logger
logging.getLogger("uvicorn.access").addFilter(_StripCallbackQuery())

def security_check():
    """Verify file permissions and environment security."""
    # C1: Check .env file permissions (Mac/Linux only)
    if os.name != 'nt':
        try:
            mode = os.stat('.env').st_mode & 0o777
            if mode > 0o600:
                logger.warning(f"SECURITY WARNING: .env file is world-readable (mode {oct(mode)}). Run 'chmod 600 .env'")
        except FileNotFoundError:
            pass

async def _sentiment_scoring_loop(interval_seconds: int = 300) -> None:
    """Score any unscored headlines every `interval_seconds`. Runs for the process lifetime."""
    scorer = get_default_scorer()
    logger.info(f"Sentiment scorer ready ({scorer.model_name}), interval={interval_seconds}s")
    while True:
        try:
            n = await score_unscored_news(scorer, limit=30, concurrency=2)
            if n:
                logger.info(f"Sentiment: scored {n} new headlines")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Sentiment scoring loop error: {type(e).__name__}: {e}")
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    security_check()
    await init_db()

    # Try to start streamer if token is valid
    token = await get_valid_token()
    if token:
        logger.info("Valid token found on startup, starting streamer...")
        asyncio.create_task(streamer_manager.start_market_data_streamer(token))
    else:
        logger.info("No valid token found, waiting for login.")

    # Watchdog: restarts the streamer if ticks stop flowing during market hours
    # and the SDK's own auto-reconnect has given up.
    streamer_manager.start_health_monitor()

    # News pipeline: fetch headlines every 15 min, score new ones every 5 min.
    news_task = asyncio.create_task(market_news_loop(interval_seconds=900), name="news_fetcher")
    sentiment_task = asyncio.create_task(_sentiment_scoring_loop(interval_seconds=300), name="sentiment_scorer")

    # Paper-trading engine. Starts the per-minute decision loop + SL/TP watcher.
    # No-op if model_v1.json is missing (engine logs a warning and stays dormant).
    await paper_loop.start_paper_engine()

    yield

    # Shutdown logic
    await paper_loop.stop_paper_engine()
    news_task.cancel()
    sentiment_task.cancel()
    streamer_manager.stop_health_monitor()
    streamer_manager.stop_market_data_streamer()

app = FastAPI(
    title="Upstox Live Dashboard", 
    lifespan=lifespan,
    # M1: User wants to keep /docs open for local dev, but we explicitly note it
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Include routers
app.include_router(auth.router)
app.include_router(stream.router)
app.include_router(paper_router.router)
app.include_router(market_api.router)
app.include_router(stock_api.router)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the dashboard HTML efficiently (L5)."""
    return FileResponse("src/static/index.html")


@app.get("/streamer/health")
async def streamer_health_endpoint():
    """Honest snapshot of the upstream WS pipe. The dashboard polls this to
    decide whether the 'stream' pill should say live / stalled / idle.

    Returns:
        streamer_running         bool   — process holds a streamer instance
        last_tick_seconds_ago    int|None — seconds since most recent tick (None if none yet)
        market_open              bool
        stale                    bool   — running but no ticks during market hours
    """
    return streamer_manager.streamer_health()

if __name__ == "__main__":
    import uvicorn
    # H2: Start with access log enabled but filtered
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=True)
