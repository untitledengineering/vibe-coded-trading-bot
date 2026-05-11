import upstox_client
import asyncio
from typing import List, Dict, Any, Callable
from src.utils.logger import logger

class UpstoxWebsocketClient:
    """Wrapper for Upstox Market Data Streamer V3."""

    def __init__(self, api_client: upstox_client.ApiClient, instrument_keys: List[str], broadcast_callback: Callable):
        """Initialize the streamer with API client, keys, and a broadcast callback."""
        self.api_client = api_client
        self.instrument_keys = instrument_keys
        self.broadcast_callback = broadcast_callback
        self.streamer = None
        self.loop = None

    def connect(self) -> None:
        """Initialize and connect the Market Data Streamer."""
        try:
            # Capture the running loop inside connect to avoid deprecation issues
            self.loop = asyncio.get_running_loop()
            
            self.streamer = upstox_client.MarketDataStreamerV3(
                api_client=self.api_client,
                instrumentKeys=self.instrument_keys,
                mode='ltpc'
            )

            self.streamer.on('open', self._on_open)
            self.streamer.on('message', self._on_message)
            self.streamer.on('error', self._on_error)
            self.streamer.on('close', self._on_close)

            logger.info(f"Connecting to Upstox stream for {len(self.instrument_keys)} symbols")
            self.streamer.connect()
        except Exception as e:
            logger.error(f"Failed to connect streamer: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect the streamer."""
        if self.streamer:
            self.streamer.disconnect()
            logger.info("Streamer disconnected")

    def _on_open(self) -> None:
        """Handle connection open event."""
        logger.info("Upstox Streamer Connection Opened")
        if self.streamer:
            self.streamer.subscribe(self.instrument_keys, 'ltpc')

    def _on_message(self, data: Any) -> None:
        """Handle incoming tick data and broadcast to all clients."""
        logger.debug(f"Tick received: {data}")
        try:
            if self.loop and self.loop.is_running():
                # Use threadsafe call to invoke the broadcast callback in the main loop
                self.loop.call_soon_threadsafe(self.broadcast_callback, data)
        except Exception as e:
            logger.error(f"Error broadcasting tick: {type(e).__name__}")

    def _on_error(self, error: Any) -> None:
        """Handle streamer errors."""
        logger.error(f"Upstox Streamer Error: {error}")

    def _on_close(self, code: int, reason: str) -> None:
        """Handle connection close event."""
        logger.info(f"Upstox Streamer Closed: {code} - {reason}")
