import pytest
import json
import asyncio
from fastapi.testclient import TestClient
from src.main import app

def test_stream_endpoint(mocker):
    """Verify that /stream yields data from the queue in SSE format."""
    client = TestClient(app)
    
    # Mock tick_queue.get to return a value and then a dummy to stop the loop
    fake_tick = {"feeds": {"NSE_EQ|RELIANCE": {"ltp": 2500.0}}}
    
    # We mock the generator's dependency or the generator itself
    # A cleaner way is to mock tick_event_generator
    async def mock_generator(request, queue):
        yield f"data: {json.dumps(fake_tick)}\n\n"
        
    mocker.patch("src.api.stream.get_valid_token", return_value="valid_token")
    mocker.patch("src.api.stream.tick_event_generator", side_effect=mock_generator)
    
    with client.stream("GET", "/stream") as response:
        for line in response.iter_lines():
            if line:
                assert line.startswith("data: ")
                data = json.loads(line.replace("data: ", ""))
                assert data["feeds"]["NSE_EQ|RELIANCE"]["ltp"] == 2500.0
                break
