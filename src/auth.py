import upstox_client
from upstox_client.rest import ApiException
import json
import time
import urllib.parse
from src.config import API_KEY, API_SECRET, REDIRECT_URI, TOKEN_FILE
from src.logger import logger

def get_login_url() -> str:
    """
    Generates the Upstox OAuth login URL with encoded parameters.
    """
    encoded_uri = urllib.parse.quote(REDIRECT_URI, safe='')
    return f"https://api.upstox.com/v2/login/authorization/dialog?client_id={API_KEY}&redirect_uri={encoded_uri}"

def exchange_code_for_token(code: str) -> str:
    """
    Exchanges the authorization code for an access token using Upstox SDK.
    """
    api_instance = upstox_client.LoginApi()
    try:
        # api_instance.token() requires api_version as the first argument
        api_response = api_instance.token(
            '2.0',
            code=code,
            client_id=API_KEY,
            client_secret=API_SECRET,
            redirect_uri=REDIRECT_URI,
            grant_type='authorization_code'
        )
        access_token = api_response.access_token
        logger.info(f"Received new token: {access_token[:5]}... (length: {len(access_token)})")
        # Calculate expiration (default to 24h if not provided)
        expires_in = getattr(api_response, 'expires_in', 86400)
        expires_at = int(time.time()) + expires_in

        _save_token_data(access_token, expires_at)
        return access_token
    except ApiException as e:
        logger.error(f"Exception when calling LoginApi->token: {e}")
        raise RuntimeError(f"Authentication failed: {e}")

def get_access_token() -> str | None:
    """
    Retrieves the access token from the local cache file if not expired.
    """
    if TOKEN_FILE.exists():
        try:
            data = json.loads(TOKEN_FILE.read_text())
            if data.get("expires_at", 0) > time.time():
                token = data.get("access_token")
                if token:
                    logger.info(f"Loaded token from cache: {token[:5]}...")
                return token
            else:
                logger.info("Cached token has expired.")
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to decode token cache file.")
    return None

def is_token_expired() -> bool:
    """
    Checks if the current token is expired.
    """
    if TOKEN_FILE.exists():
        try:
            data = json.loads(TOKEN_FILE.read_text())
            return data.get("expires_at", 0) <= time.time()
        except (json.JSONDecodeError, KeyError):
            return True
    return True

def _save_token_data(token: str, expires_at: int) -> None:
    """
    Saves the access token and expiration to the local cache file.
    """
    import os
    data = {
        "access_token": token,
        "expires_at": expires_at
    }
    TOKEN_FILE.write_text(json.dumps(data))
    os.chmod(TOKEN_FILE, 0o600)
