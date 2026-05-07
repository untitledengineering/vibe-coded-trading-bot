import upstox_client
from upstox_client.rest import ApiException
from typing import List, Dict, Any, Callable
import json
from google.protobuf.json_format import MessageToDict
import upstox_client.feeder.proto.MarketDataFeedV3_pb2 as pb
from src.logger import logger

class MarketDataModule:
    """
    Handles instrument lookup and live market data streaming.
    """

    def __init__(self, access_token: str):
        """
        Initializes the MarketDataModule with an access token.
        """
        self.configuration = upstox_client.Configuration()
        self.configuration.access_token = access_token.strip()
        # Set as default to ensure all SDK components use this config
        upstox_client.Configuration.set_default(self.configuration)

        self.api_client = upstox_client.ApiClient(self.configuration)
        self.api_client.set_default_header('Authorization', f'Bearer {access_token.strip()}')
        self.api_client.set_default_header('Api-Version', '2.0')
        self.instrument_api = upstox_client.InstrumentsApi(self.api_client)
        self.streamer = None

    def get_instrument_keys(self, symbols: List[str]) -> Dict[str, str]:
        """
        Fetches instrument keys for the given symbols.
        Returns a mapping of symbol -> instrument_key.
        """
        results = {}
        for symbol in symbols:
            try:
                # Try NSE first, then fallback to NSE_INDEX for indices
                found = False
                for exchange in ['NSE', 'NSE_INDEX']:
                    if found: break
                    try:
                        search_response = self.instrument_api.search_instrument(
                            query=symbol,
                            exchanges=exchange
                        )

                        if not search_response or not search_response.data:
                            continue

                        for item in search_response.data:
                            # Some SDK versions return dicts, others return objects
                            trading_symbol = item.get('trading_symbol', '') if isinstance(item, dict) else getattr(item, 'trading_symbol', '')
                            short_name = item.get('short_name', '') if isinstance(item, dict) else getattr(item, 'short_name', '')
                            instrument_key = item.get('instrument_key', '') if isinstance(item, dict) else getattr(item, 'instrument_key', '')

                            # Match logic: exact or NIFTY 50 specific variations
                            target = symbol.upper()
                            if target == "NIFTY 50":
                                match = trading_symbol.upper() in ["NIFTY 50", "NIFTY_50", "NIFTY50"]
                            else:
                                match = trading_symbol.upper() == target or short_name.upper() == target

                            if match:
                                results[symbol] = instrument_key
                                found = True
                                break
                    except ApiException as e:
                        logger.debug(f"Search in {exchange} failed for {symbol}: {e}")
                        continue

                if not found:
                    # Log candidates to help debug matching issues
                    candidates = []
                    try:
                        search_response = self.instrument_api.search_instrument(query=symbol, exchanges='NSE_INDEX')
                        if search_response and search_response.data:
                            candidates = [ (item.get('trading_symbol') if isinstance(item, dict) else getattr(item, 'trading_symbol', '')) for item in search_response.data[:5] ]
                    except:
                        pass
                    logger.warning(f"Symbol {symbol} not found. Top candidates in NSE_INDEX: {candidates}")

            except ApiException as e:
                logger.error(f"Exception when calling InstrumentApi->search_instrument for {symbol}: {e}")

        return results

    def start_stream(self, instrument_keys: List[str], on_tick: Callable[[Any], None]) -> None:
        """
        Starts the WebSocket stream for the given instrument keys.
        """
        # upstox-python-sdk v2 uses EventEmitter pattern
        self.streamer = upstox_client.MarketDataStreamerV3(
            api_client=self.api_client,
            instrumentKeys=instrument_keys,
            mode='ltpc'
        )

        def on_open_callback():
            import time
            logger.info("Streamer Connection Opened. Waiting 1s for stability...")
            time.sleep(1)
            logger.info("Subscribing to instruments...")
            self.streamer.subscribe(instrument_keys, 'ltpc')

        # Register listeners using the native .on() method
        self.streamer.on('open', on_open_callback)
        self.streamer.on('message', lambda data: on_tick(data))
        self.streamer.on('error', lambda err: logger.error(f"Streamer Error: {err}"))
        self.streamer.on('close', lambda code, reason: logger.info(f"Streamer Closed: {code} - {reason}"))

        logger.info(f"Connecting to live feed for {len(instrument_keys)} instruments...")
        self.streamer.connect()

    def _parse_message(self, message: Any) -> Dict[str, Any]:
        """
        No longer needed as MarketDataStreamerV3 handles decoding natively.
        """
        return message
