import asyncio
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from src.utils.logger import logger
from src.db.database import get_valid_token

router = APIRouter()

# List of active client queues
active_clients = []
logger.info(f"Stream module initialized (id: {id(active_clients)})")

_LOGGED_TICK_SAMPLE = False  # one-shot diagnostic; flipped after the first tick is logged


def _sanitize_tick(tick_data) -> dict:
    """Best-effort JSON-safe conversion. Upstox V3 protobuf-decoded payloads can
    include bytes/enum/non-dict types that json.dumps refuses. We coerce to str
    on anything it can't handle, and recurse through dicts/lists."""
    if isinstance(tick_data, dict):
        return {str(k): _sanitize_tick(v) for k, v in tick_data.items()}
    if isinstance(tick_data, (list, tuple)):
        return [_sanitize_tick(x) for x in tick_data]
    if isinstance(tick_data, (str, int, float, bool)) or tick_data is None:
        return tick_data
    if isinstance(tick_data, bytes):
        try:
            return tick_data.decode("utf-8", errors="replace")
        except Exception:
            return repr(tick_data)
    # Enums, custom proto types, etc.
    return str(tick_data)


async def tick_event_generator(request: Request, queue: asyncio.Queue):
    """Generator that yields tick data from a per-client queue. Tolerant of
    serialization errors — one bad tick must NOT close the SSE."""
    global _LOGGED_TICK_SAMPLE
    try:
        while True:
            if await request.is_disconnected():
                break

            try:
                # Wait for data with a timeout to check for disconnection
                tick_data = await asyncio.wait_for(queue.get(), timeout=1.0)

                # First tick after restart: log its type/keys so we know what we got.
                if not _LOGGED_TICK_SAMPLE:
                    logger.info(
                        f"First tick on SSE pipe: type={type(tick_data).__name__} "
                        f"keys={list(tick_data.keys()) if isinstance(tick_data, dict) else 'n/a'}"
                    )
                    _LOGGED_TICK_SAMPLE = True

                try:
                    payload = json.dumps(tick_data)
                except (TypeError, ValueError) as e:
                    # Fall back to sanitised serialisation. Don't kill the SSE
                    # because one tick had a weird field.
                    logger.warning(
                        f"Tick not directly JSON-serializable ({type(e).__name__}: {e}); "
                        f"falling back to sanitised payload"
                    )
                    payload = json.dumps(_sanitize_tick(tick_data))

                yield f"data: {payload}\n\n"
                queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
    finally:
        if queue in active_clients:
            active_clients.remove(queue)
        logger.info("SSE client disconnected, queue removed")

@router.get("/stream")
async def stream_ticks(request: Request):
    """Endpoint for Server-Sent Events (SSE) streaming with authentication check."""
    # M2: Ensure the user is authenticated before allowing stream access
    token = await get_valid_token()
    if not token:
        logger.warning("Unauthorized SSE connection attempt")
        raise HTTPException(status_code=401, detail="Authentication required")
        
    # Create a new queue for this client
    client_queue = asyncio.Queue(maxsize=100)
    active_clients.append(client_queue)
    logger.info(f"New SSE client connected. Total clients: {len(active_clients)}")
    
    return StreamingResponse(
        tick_event_generator(request, client_queue),
        media_type="text/event-stream"
    )

def broadcast_tick(tick_data: dict):
    """Push tick data to all active client queues. 
    Synchronous to be compatible with loop.call_soon_threadsafe."""
    if not active_clients:
        return
        
    logger.debug(f"Broadcasting tick to {len(active_clients)} clients")
    for queue in active_clients:
        try:
            # Non-blocking put, skip if queue is full
            queue.put_nowait(tick_data)
        except asyncio.QueueFull:
            # If a client is too slow, we can't let it block everyone
            pass
