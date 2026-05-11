import asyncio
import os
import logging
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from src.api import auth, stream, streamer_manager
from src.db.database import init_db, get_valid_token
from src.utils.logger import logger
from contextlib import asynccontextmanager

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
    
    yield
    
    # Shutdown logic
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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the dashboard HTML efficiently (L5)."""
    return FileResponse("src/static/index.html")

if __name__ == "__main__":
    import uvicorn
    # H2: Start with access log enabled but filtered
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=True)
