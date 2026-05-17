import asyncio
import time
from typing import Optional

from src.api import rest_client, stream, websocket_client
from src.data.bar_builder import BarBuilder
from src.db.database import get_valid_token
from src.utils.logger import logger

# Process-global handles. Held here (not in main.py) so /auth/logout can tear them down.
streamer: Optional[websocket_client.UpstoxWebsocketClient] = None
bar_builder: Optional[BarBuilder] = None
_health_monitor_task: Optional[asyncio.Task] = None

# Wall-clock of the most recent tick that arrived from Upstox. 0 = no tick yet.
last_tick_at: float = 0.0

# Latest LTP per instrument_key. Updated on every tick in _fanout. The paper
# engine reads this for sub-minute mark-to-market — fresher than bars_live
# (which only writes on minute rollover).
last_quote_by_symbol: dict[str, float] = {}

# Health-monitor thresholds. The monitor restarts the streamer if no tick has
# arrived during market hours within this window. Empirically 60s of silence
# during market hours = a broken upstream pipe; we allow a 90s grace window.
STALE_TICK_THRESHOLD_SECONDS = 90
HEALTH_CHECK_INTERVAL_SECONDS = 30

# IST trading window. Duplicated here to avoid importing from src.features
# (keeps src/api -> src/features one-directional).
IST_OFFSET_SECONDS = 5 * 3600 + 30 * 60

def _load_stream_symbols() -> list[str]:
    """Full F&O equity universe + Nifty 50 index for the dashboard pill.
    Falls back to a minimal hardcoded list if universe.json is missing."""
    try:
        from src.data.universe import load_universe
        keys = [u["instrument_key"] for u in load_universe()]
        return ["NSE_INDEX|Nifty 50"] + keys
    except Exception:
        return [
            "NSE_INDEX|Nifty 50",
            "NSE_EQ|INE002A01018",
            "NSE_EQ|INE040A01034",
            "NSE_EQ|INE009A01021",
            "NSE_EQ|INE467B01029",
            "NSE_EQ|INE090A01021",
        ]

STREAM_SYMBOLS: list[str] = _load_stream_symbols()


def _record_quotes(tick_data) -> None:
    """Pull LTP out of every feed in the tick payload and stash in
    last_quote_by_symbol. Used by the paper engine's mark-to-market loop."""
    if not isinstance(tick_data, dict):
        return
    feeds = tick_data.get("feeds")
    if not isinstance(feeds, dict):
        return
    for key, payload in feeds.items():
        if not isinstance(payload, dict):
            continue
        ltpc = payload.get("ltpc") or (payload.get("ff", {}) or {}).get("marketFF", {}).get("ltpc")
        if not ltpc:
            continue
        ltp = ltpc.get("ltp")
        if ltp is None:
            continue
        try:
            last_quote_by_symbol[key] = float(ltp)
        except (TypeError, ValueError):
            pass


def _fanout(tick_data) -> None:
    """Run inside the asyncio loop (via call_soon_threadsafe). Pushes the tick to
    every interested consumer. Each branch is wrapped so one failure can't drop
    the others — the bar builder errors out of curiosity should not break the SSE."""
    global last_tick_at
    last_tick_at = time.time()
    _record_quotes(tick_data)
    try:
        stream.broadcast_tick(tick_data)
    except Exception as e:
        logger.error(f"SSE broadcast failed: {type(e).__name__}")
    if bar_builder is not None:
        try:
            bar_builder.enqueue(tick_data)
        except Exception as e:
            logger.error(f"BarBuilder enqueue failed: {type(e).__name__}")


def is_market_open(now_ts: Optional[float] = None) -> bool:
    """NSE trading hours: 09:15–15:30 IST Mon–Fri. We don't track holidays."""
    now = now_ts if now_ts is not None else time.time()
    ist = time.gmtime(now + IST_OFFSET_SECONDS)
    if ist.tm_wday >= 5:  # 0=Mon..6=Sun
        return False
    minutes = ist.tm_hour * 60 + ist.tm_min
    return 9 * 60 + 15 <= minutes < 15 * 60 + 30


def streamer_health() -> dict:
    """Snapshot for the dashboard's stream pill. Single source of truth for
    'is the upstream pipe actually flowing?'."""
    seconds_ago: Optional[int]
    if last_tick_at == 0.0:
        seconds_ago = None
    else:
        seconds_ago = int(time.time() - last_tick_at)
    return {
        "streamer_running": streamer is not None,
        "last_tick_seconds_ago": seconds_ago,
        "market_open": is_market_open(),
        "stale": (
            streamer is not None
            and is_market_open()
            and (seconds_ago is None or seconds_ago > STALE_TICK_THRESHOLD_SECONDS)
        ),
    }


async def _health_monitor_loop() -> None:
    """Watchdog. If we're in market hours and ticks have stopped for longer
    than STALE_TICK_THRESHOLD_SECONDS, restart the streamer. Belt-and-suspenders
    on top of the SDK's own auto_reconnect, which the empirical 2026-05-15
    overnight drop showed could give up after enough DNS failures."""
    try:
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL_SECONDS)
            if not is_market_open():
                continue
            if streamer is None:
                continue
            since_last = time.time() - last_tick_at if last_tick_at else float("inf")
            if since_last <= STALE_TICK_THRESHOLD_SECONDS:
                continue
            logger.warning(
                f"Streamer stale during market hours ({since_last:.0f}s since last tick). "
                f"Restarting upstream connection."
            )
            token = await get_valid_token()
            if not token:
                logger.error(
                    "Cannot restart streamer: no valid token in DB. User must re-authenticate."
                )
                continue
            try:
                await start_market_data_streamer(token)
            except Exception as e:
                logger.error(f"Streamer auto-restart failed: {type(e).__name__}: {e}")
    except asyncio.CancelledError:
        return


def start_health_monitor() -> None:
    """Spawn the watchdog once. Called from FastAPI's lifespan at startup."""
    global _health_monitor_task
    if _health_monitor_task is not None and not _health_monitor_task.done():
        return
    _health_monitor_task = asyncio.create_task(
        _health_monitor_loop(), name="streamer_health_monitor"
    )
    logger.info("Streamer health monitor started")


def stop_health_monitor() -> None:
    global _health_monitor_task
    if _health_monitor_task is not None:
        _health_monitor_task.cancel()
        _health_monitor_task = None


async def start_market_data_streamer(token: str):
    """Start (or restart) the streamer and the bar builder."""
    global streamer, bar_builder

    if streamer is not None:
        logger.info("Streamer already running, reconnecting...")
        streamer.disconnect()
        streamer = None
    if bar_builder is not None:
        await bar_builder.stop()
        bar_builder = None

    try:
        client = rest_client.UpstoxRestClient(access_token=token)

        bar_builder = BarBuilder()
        bar_builder.start()

        streamer = websocket_client.UpstoxWebsocketClient(
            api_client=client.api_client,
            instrument_keys=STREAM_SYMBOLS,
            broadcast_callback=_fanout,
        )
        streamer.connect()
        logger.info(f"Streamer started for {len(STREAM_SYMBOLS)} symbols (bar builder live)")
    except Exception as e:
        logger.error(f"Failed to start streamer: {type(e).__name__}")
        # Best-effort cleanup so a half-init doesn't leave dangling state.
        if bar_builder is not None:
            try:
                await bar_builder.stop()
            finally:
                bar_builder = None


def stop_market_data_streamer():
    """Synchronous shutdown. Schedules bar-builder finalisation on the running loop."""
    global streamer, bar_builder

    if streamer is not None:
        streamer.disconnect()
        streamer = None
        logger.info("Streamer stopped")

    if bar_builder is not None:
        # We cannot await here (callers are synchronous); schedule the flush instead.
        bb = bar_builder
        bar_builder = None
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(bb.stop())
        except RuntimeError:
            # No running loop (process shutdown). Best effort: skip async flush.
            logger.warning("BarBuilder stop skipped: no running event loop")
