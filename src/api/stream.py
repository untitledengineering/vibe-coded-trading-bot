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

async def tick_event_generator(request: Request, queue: asyncio.Queue):
    """Generator that yields tick data from a per-client queue."""
    try:
        while True:
            if await request.is_disconnected():
                break

            try:
                # Wait for data with a timeout to check for disconnection
                tick_data = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield f"data: {json.dumps(tick_data)}\n\n"
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
