import asyncio
import secrets
import re
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, JSONResponse
from src.api.rest_client import UpstoxRestClient
from src.db.database import save_token, get_token_info, clear_tokens
from src.utils.logger import logger
from src.api import streamer_manager

router = APIRouter()

@router.get("/auth/status")
async def auth_status():
    """Return authentication state plus token metadata (never the token value itself)."""
    info = await get_token_info()
    if not info or not info["valid"]:
        return {"authenticated": False}
    return {
        "authenticated": True,
        "issued_at": info["issued_at"],
        "expires_at": info["expires_at"],
        "seconds_until_expiry": info["seconds_until_expiry"],
    }


@router.post("/auth/logout")
async def logout():
    """Stop the live streamer and wipe stored tokens."""
    streamer_manager.stop_market_data_streamer()
    await clear_tokens()
    logger.info("Logout complete: streamer stopped, tokens cleared")
    return {"ok": True}

@router.get("/auth/login")
async def login():
    """Redirect the user to the Upstox login page with a state parameter."""
    state = secrets.token_urlsafe(32)
    client = UpstoxRestClient()
    auth_url = client.get_authorize_url(state=state)

    logger.info("Initiating login flow")
    response = RedirectResponse(auth_url)
    
    # Store state in an HTTP-only cookie
    response.set_cookie(
        "oauth_state", 
        state, 
        httponly=True, 
        secure=False, # Set to True in production with HTTPS
        samesite="lax", 
        max_age=600,
        path="/"
    )
    return response

@router.get("/callback")
async def callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    """Handle the OAuth callback with state validation and error handling."""
    
    # Handle error redirected from Upstox. error is from a fixed OAuth enum; we never
    # surface the provider's error_description to the client or the log line (treat it
    # as untrusted text — log injection vector).
    if error:
        safe_error = error[:64] if isinstance(error, str) else "unknown"
        logger.warning("Upstox auth error received: %s", safe_error)
        return JSONResponse(
            status_code=400,
            content={"error": "Authentication failed at provider"},
        )

    if not code or len(code) > 512 or not re.match(r"^[A-Za-z0-9._\-]+$", code):
        logger.warning("OAuth callback missing or malformed code")
        return JSONResponse(status_code=400, content={"error": "Invalid code"})

    cookie_state = request.cookies.get("oauth_state")
    if (
        not state
        or not cookie_state
        or len(state) > 128
        or not secrets.compare_digest(state, cookie_state)
    ):
        # Do NOT log the state values — replay window is short but free to skip.
        logger.warning("OAuth state validation failed")
        return JSONResponse(status_code=400, content={"error": "Invalid state"})

    try:
        client = UpstoxRestClient()
        access_token = await client.exchange_code_for_token(code)
        await save_token(access_token)
        logger.info("Successfully authenticated and saved token")

        # Start streamer in background
        asyncio.create_task(streamer_manager.start_market_data_streamer(access_token))

        response = RedirectResponse(url="/")
        response.delete_cookie("oauth_state", path="/")
        return response
    except Exception as e:
        logger.error("Callback exchange failed: %s", type(e).__name__)
        return JSONResponse(status_code=400, content={"error": "Authentication failed"})
