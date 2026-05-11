import asyncio
from typing import Optional, List
from src.api import rest_client, websocket_client, stream
from src.utils.logger import logger

# Global streamer instance
streamer: Optional[websocket_client.UpstoxWebsocketClient] = None

async def start_market_data_streamer(token: str):
    """Start the Upstox market data streamer in the background."""
    global streamer
    
    # If already running, disconnect first
    if streamer:
        logger.info("Streamer already running, reconnecting...")
        streamer.disconnect()
        
    try:
        client = rest_client.UpstoxRestClient(access_token=token)
        
        # Symbols for subscription
        symbols = [
            "NSE_INDEX|Nifty 50",
            "NSE_EQ|INE002A01018", # Reliance
            "NSE_EQ|INE040A01034", # HDFC Bank
            "NSE_EQ|INE009A01021", # Infosys
            "NSE_EQ|INE467B01029", # TCS
            "NSE_EQ|INE090A01021"  # ICICI Bank
        ]
        
        streamer = websocket_client.UpstoxWebsocketClient(
            api_client=client.api_client,
            instrument_keys=symbols,
            broadcast_callback=stream.broadcast_tick
        )
        streamer.connect()
        logger.info(f"Streamer started for {len(symbols)} symbols")
    except Exception as e:
        logger.error(f"Failed to start background streamer: {e}")

def stop_market_data_streamer():
    """Stop the global streamer instance."""
    global streamer
    if streamer:
        streamer.disconnect()
        streamer = None
        logger.info("Streamer stopped")
