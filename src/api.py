from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from src.auth import get_login_url, exchange_code_for_token, get_access_token, is_token_expired
from src.market_data import MarketDataModule
from src.logger import logger
from src.config import REDIRECT_URI

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """
    Serves the main dashboard page.
    """
    with open("src/static/index.html") as f:
        return f.read()

@app.get("/login")
async def login():
    """
    Redirects to Upstox login page.
    """
    return RedirectResponse(get_login_url())

@app.get("/callback")
async def callback(code: str):
    """
    Handles the OAuth callback from Upstox.
    """
    try:
        exchange_code_for_token(code)
        return RedirectResponse(url="/")
    except Exception as e:
        return {"error": f"Authentication failed: {str(e)}"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket proxy for real-time market data.
    """
    await websocket.accept()

    token = get_access_token()
    if not token or is_token_expired():
        await websocket.send_json({"error": "unauthorized"})
        await websocket.close()
        return

    md_module = MarketDataModule(token)
    symbols = ["RELIANCE", "HDFCBANK", "TCS"]

    symbol_to_key = md_module.get_instrument_keys(symbols)
    key_to_symbol = {v: k for k, v in symbol_to_key.items()}

    # Get the running loop to safely pass data from the SDK thread
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def handle_tick(tick):
        # Log that we received a tick for debugging
        logger.info(f"Tick received from Upstox for {len(tick.get('feeds', {}))} instruments")
        loop.call_soon_threadsafe(queue.put_nowait, tick)

    try:
        # Start streamer in a background thread as it might be blocking
        asyncio.create_task(asyncio.to_thread(md_module.start_stream, list(symbol_to_key.values()), handle_tick))
        logger.info("Market data streamer started in background. Waiting for ticks...")

        while True:
            tick = await queue.get()
            processed_feeds = {}

            if tick and 'feeds' in tick:
                for key, feed in tick['feeds'].items():
                    if key in key_to_symbol:
                        symbol = key_to_symbol[key]
                        try:
                            # In LTPC mode, ltpc is directly in the feed
                            ltpc = feed.get('ltpc')
                            if ltpc:
                                processed_feeds[symbol] = {
                                    'lp': ltpc.get('ltp'),
                                    'cp': ltpc.get('cp')
                                }
                        except Exception:
                            pass

            if processed_feeds:
                await websocket.send_json(processed_feeds)

    except WebSocketDisconnect:
        logger.info("Web client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error occurred")
    finally:
        # Note: Ideally md_module should have a stop_stream method
        pass
