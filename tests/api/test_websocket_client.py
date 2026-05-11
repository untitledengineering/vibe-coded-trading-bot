import pytest
import asyncio
from src.api.websocket_client import UpstoxWebsocketClient

def test_websocket_connect(mocker):
    """Verify that connect initializes the MarketDataStreamerV3."""
    mock_streamer = mocker.patch("upstox_client.MarketDataStreamerV3")
    mocker.patch("asyncio.get_running_loop")
    
    mock_api_client = mocker.Mock()
    callback = mocker.Mock()
    client = UpstoxWebsocketClient(api_client=mock_api_client, instrument_keys=["NSE_EQ|RELIANCE"], broadcast_callback=callback)
    
    client.connect()
    
    mock_streamer.assert_called_once()
    assert client.streamer is not None

def test_websocket_on_message(mocker):
    """Verify that _on_message calls the broadcast callback."""
    mock_api_client = mocker.Mock()
    callback = mocker.Mock()
    
    # Mock the event loop
    mock_loop = mocker.Mock()
    mocker.patch("asyncio.get_event_loop", return_value=mock_loop)
    
    client = UpstoxWebsocketClient(api_client=mock_api_client, instrument_keys=["NSE_EQ|RELIANCE"], broadcast_callback=callback)
    client.loop = mock_loop
    
    fake_data = {"feeds": {"NSE_EQ|RELIANCE": {"ltp": 2500.0}}}
    client._on_message(fake_data)
    
    # Verify call_soon_threadsafe was called with the callback
    mock_loop.call_soon_threadsafe.assert_called_once_with(callback, fake_data)

def test_websocket_disconnect(mocker):
    """Verify that disconnect calls streamer.disconnect."""
    mock_streamer = mocker.Mock()
    client = UpstoxWebsocketClient(api_client=mocker.Mock(), instrument_keys=[], broadcast_callback=mocker.Mock())
    client.streamer = mock_streamer
    
    client.disconnect()
    
    mock_streamer.disconnect.assert_called_once()
