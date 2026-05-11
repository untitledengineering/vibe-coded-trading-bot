import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_auth_status_unauthenticated(mocker):
    """Verify /auth/status returns false when no token exists."""
    mocker.patch("src.api.auth.get_valid_token", return_value=None)
    response = client.get("/auth/status")
    assert response.status_code == 200
    assert response.json() == {"authenticated": False}

def test_auth_status_authenticated(mocker):
    """Verify /auth/status returns true when a valid token exists."""
    mocker.patch("src.api.auth.get_valid_token", return_value="valid_token")
    response = client.get("/auth/status")
    assert response.status_code == 200
    assert response.json() == {"authenticated": True}

def test_login_redirect(mocker):
    """Verify /auth/login redirects to Upstox auth URL."""
    # Mock RestClient to return a specific URL
    mock_client = mocker.patch("src.api.auth.UpstoxRestClient")
    mock_client.return_value.get_authorize_url.return_value = "https://upstox.com/auth"
    
    response = client.get("/auth/login", follow_redirects=False)
    assert response.status_code == 307 # Temporary Redirect
    assert response.headers["location"].startswith("https://upstox.com/auth")
    assert "&state=" in response.headers["location"]
    assert "oauth_state" in response.cookies
