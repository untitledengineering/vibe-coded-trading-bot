import asyncio
import secrets
import re
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from src.api.rest_client import UpstoxRestClient
from src.db.database import save_token, get_valid_token
from src.utils.logger import logger
from src.api import streamer_manager

router = APIRouter()

@router.get("/auth/status")
async def auth_status():
    """Check if the user is authenticated with a valid token."""
    token = await get_valid_token()
    return {"authenticated": token is not None}

@router.get("/auth/login")
async def login():
    """Redirect the user to the Upstox login page with a state parameter."""
    client = UpstoxRestClient()
    auth_url = client.get_authorize_url()
    
    # Generate random state for CSRF protection
    state = secrets.token_urlsafe(32)
    
    logger.info("Initiating login flow with state")
    response = RedirectResponse(f"{auth_url}&state={state}")
    
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
    error_description: str | None = None
):
    """Handle the OAuth callback with state validation and error handling."""
    
    # Handle error redirected from Upstox
    if error:
        logger.error(f"Upstox auth error: {error} - {error_description}")
        return JSONResponse(
            status_code=400, 
            content={"error": error, "description": error_description}
        )

    # Validate state parameter (CSRF protection)
    cookie_state = request.cookies.get("oauth_state")
    logger.debug(f"State validation: received={state}, cookie={cookie_state}")
    
    if not state or not cookie_state or state != cookie_state:
        logger.warning(f"OAuth state mismatch or missing. Received: {state}, Cookie: {cookie_state}")
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    # Basic shape validation for code
    if len(code) > 512 or not re.match(r"^[A-Za-z0-9._-]+$", code):
        logger.warning("Malformed authorization code received")
        raise HTTPException(status_code=400, detail="Invalid code format")

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
        import traceback
        logger.error(f"Callback error: {type(e).__name__} - {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=400, content={"error": "Authentication failed", "detail": str(e)})
