import time
import sys
from src.config import validate_config
from src.auth import get_login_url, exchange_code_for_token, get_access_token
from src.market_data import MarketDataModule
from src.utils import clear_screen, print_data_table
from src.logger import logger

def authenticate() -> str:
    """
    Handles the OAuth2 authentication flow.
    """
    token = get_access_token()
    if token:
        return token

    logger.info("Authentication required.")
    print(f"1. Open this URL in your browser: {get_login_url()}")
    print("2. Log in and authorize the app.")
    print("3. Copy the 'code' parameter from the redirect URL.")
    code = input("Enter the authorization code: ").strip()
    try:
        token = exchange_code_for_token(code)
        logger.info("Authentication successful!")
        return token
    except Exception:
        logger.error("Authentication failed")
        sys.exit(1)

def main():
    """
    Main entry point for the live market data bot.
    """
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)

    try:
        while True:
            token = authenticate()

            # 2. Market Data Setup
            md_module = MarketDataModule(token)
            symbols = ["NIFTY 50", "RELIANCE", "HDFCBANK", "TCS"]

            logger.info("Searching for instrument keys...")
            symbol_to_key = md_module.get_instrument_keys(symbols)
            key_to_symbol = {v: k for k, v in symbol_to_key.items()}

            if not symbol_to_key:
                logger.error("Error: Could not find any instrument keys.")
                sys.exit(1)

            # State to hold latest prices
            market_data = {symbol: {'lp': 0.0, 'close': 0.0} for symbol in symbol_to_key.keys()}

            def handle_tick(tick):
                """
                Callback for each tick received from WebSocket.
                """
                if not tick or 'feeds' not in tick:
                    return

                for key, feed in tick['feeds'].items():
                    if key in key_to_symbol:
                        symbol = key_to_symbol[key]
                        try:
                            ltpc = feed.get('ltpc')
                            if ltpc:
                                market_data[symbol]['lp'] = ltpc.get('ltp', market_data[symbol]['lp'])
                                market_data[symbol]['close'] = ltpc.get('cp', market_data[symbol]['close'])
                        except Exception:
                            pass

                print_data_table(market_data)

            # 3. Start Streaming
            clear_screen()
            try:
                md_module.start_stream(list(symbol_to_key.values()), handle_tick)

                # Keep alive and check for expiration
                while True:
                    if is_token_expired():
                        logger.warning("Token expired. Re-authenticating...")
                        break
                    time.sleep(1)
            except Exception as e:
                # Check if it's a 401/Unauthorized error
                if "401" in str(e) or "Unauthorized" in str(e):
                    logger.warning("Unauthorized access. Forcing re-authentication...")
                    # Clear token file to force re-auth
                    from src.config import TOKEN_FILE
                    if TOKEN_FILE.exists():
                        TOKEN_FILE.unlink()
                    continue
                else:
                    logger.error(f"Stream error occurred")
                    break
    except KeyboardInterrupt:
        print("\nStopping feed...")
        sys.exit(0)

if __name__ == "__main__":
    main()
