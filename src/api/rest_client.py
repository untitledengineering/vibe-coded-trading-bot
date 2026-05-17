import upstox_client
from upstox_client.api import user_api, order_api, login_api
from src.utils.logger import logger
from src.utils.config import UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_REDIRECT_URI

class UpstoxRestClient:
    """Wrapper for Upstox REST API operations including OAuth."""

    def __init__(self, access_token: str = ""):
        """Initialize the client, optionally with an access token."""
        self.configuration = upstox_client.Configuration()
        if access_token:
            self.configuration.access_token = access_token
        self.api_client = upstox_client.ApiClient(self.configuration)
        self.user_api = user_api.UserApi(self.api_client)
        self.order_api = order_api.OrderApi(self.api_client)
        self.login_api = login_api.LoginApi(self.api_client)

    def get_authorize_url(self, state: str | None = None) -> str:
        """Construct the Upstox authorization URL, optionally including a CSRF state."""
        import urllib.parse
        encoded_redirect = urllib.parse.quote(UPSTOX_REDIRECT_URI, safe='')
        url = (
            f"https://api.upstox.com/v2/login/authorization/dialog"
            f"?response_type=code&client_id={UPSTOX_API_KEY}"
            f"&redirect_uri={encoded_redirect}"
        )
        if state:
            url += f"&state={urllib.parse.quote(state, safe='')}"
        return url

    async def exchange_code_for_token(self, code: str) -> str:
        """Exchange the authorization code for an access token."""
        try:
            # Note: SDK might have a sync implementation, we wrap it or use it as is
            # For Day 1, we assume the SDK provides this method
            response = self.login_api.token(
                api_version="2.0",
                code=code,
                client_id=UPSTOX_API_KEY,
                client_secret=UPSTOX_API_SECRET,
                redirect_uri=UPSTOX_REDIRECT_URI,
                grant_type="authorization_code"
            )
            logger.info("Successfully exchanged code for token")
            return response.access_token
        except Exception as e:
            logger.error(f"Failed to exchange code for token: {e}")
            raise

    def get_profile(self) -> dict:
        """Fetch the user profile information."""
        try:
            response = self.user_api.get_profile(api_version="2.0")
            logger.info("Successfully fetched user profile")
            return response.to_dict()
        except Exception as e:
            logger.error(f"Failed to fetch user profile: {e}")
            raise
