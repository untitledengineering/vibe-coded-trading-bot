import pytest
from unittest.mock import MagicMock, patch
from src.market_data import MarketDataModule

def test_get_instrument_keys():
    """
    Test that instrument keys are fetched and mapped correctly.
    """
    module = MarketDataModule("test_token")
    mock_api = MagicMock()
    module.instrument_api = mock_api

    # Mock search response
    mock_item = MagicMock()
    mock_item.trading_symbol = "RELIANCE"
    mock_item.instrument_key = "NSE_EQ|INE002A01018"

    mock_response = MagicMock()
    mock_response.data = [mock_item]
    mock_api.search_instrument.return_value = mock_response

    keys = module.get_instrument_keys(["RELIANCE"])

    assert keys["RELIANCE"] == "NSE_EQ|INE002A01018"
    mock_api.search_instrument.assert_called_with(
        query="RELIANCE",
        exchanges="NSE"
    )

@patch("upstox_client.MarketDataStreamerV3")
def test_start_stream(mock_streamer_class):
    """
    Test that the streamer is initialized and connected.
    """
    module = MarketDataModule("test_token")
    mock_streamer = mock_streamer_class.return_value

    def on_tick(tick):
        pass

    module.start_stream(["key1"], on_tick)

    mock_streamer_class.assert_called_once()
    mock_streamer.connect.assert_called_once()
    assert module.streamer == mock_streamer
