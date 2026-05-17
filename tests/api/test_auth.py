from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_auth_status_unauthenticated(mocker):
    """When no token info is available, /auth/status reports unauthenticated."""
    mocker.patch("src.api.auth.get_token_info", new_callable=mocker.AsyncMock, return_value=None)
    response = client.get("/auth/status")
    assert response.status_code == 200
    assert response.json() == {"authenticated": False}


def test_auth_status_unauthenticated_when_expired(mocker):
    """Expired token info → unauthenticated, no metadata leaked."""
    mocker.patch(
        "src.api.auth.get_token_info",
        new_callable=mocker.AsyncMock,
        return_value={
            "issued_at": "2024-01-01T00:00:00",
            "expires_at": "2024-01-02T00:00:00",
            "valid": False,
            "seconds_until_expiry": 0,
        },
    )
    response = client.get("/auth/status")
    assert response.status_code == 200
    assert response.json() == {"authenticated": False}


def test_auth_status_authenticated_includes_metadata(mocker):
    """Authenticated response carries issued_at / expires_at / countdown but never the token."""
    info = {
        "issued_at": "2025-05-11T10:00:00",
        "expires_at": "2025-05-12T10:00:00",
        "valid": True,
        "seconds_until_expiry": 86400,
    }
    mocker.patch("src.api.auth.get_token_info", new_callable=mocker.AsyncMock, return_value=info)
    response = client.get("/auth/status")
    assert response.status_code == 200
    body = response.json()
    assert body["authenticated"] is True
    assert body["issued_at"] == info["issued_at"]
    assert body["expires_at"] == info["expires_at"]
    assert body["seconds_until_expiry"] == 86400
    assert "access_token" not in body  # critical: never leak the token over the wire


def test_logout_clears_tokens_and_stops_streamer(mocker):
    """POST /auth/logout must stop the streamer and wipe stored tokens."""
    clear = mocker.patch("src.api.auth.clear_tokens", new_callable=mocker.AsyncMock)
    stop = mocker.patch("src.api.auth.streamer_manager.stop_market_data_streamer")
    response = client.post("/auth/logout")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    clear.assert_awaited_once()
    stop.assert_called_once()


def test_logout_rejects_get(mocker):
    """Logout must not respond to GET (basic CSRF defence)."""
    mocker.patch("src.api.auth.clear_tokens", new_callable=mocker.AsyncMock)
    response = client.get("/auth/logout")
    assert response.status_code == 405


def test_login_redirects_and_sets_state_cookie(mocker):
    """/auth/login should redirect to Upstox and set the oauth_state cookie."""
    mock_client = mocker.patch("src.api.auth.UpstoxRestClient")
    # Echo the state back in the URL so we can assert it propagates.
    mock_client.return_value.get_authorize_url.side_effect = (
        lambda state=None: f"https://upstox.com/auth?state={state}"
    )

    response = client.get("/auth/login", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"].startswith("https://upstox.com/auth?state=")
    cookie = response.cookies.get("oauth_state")
    assert cookie is not None and len(cookie) >= 32
    # Same state in the URL and the cookie.
    assert f"state={cookie}" in response.headers["location"]


def test_callback_rejects_provider_error(mocker):
    """If Upstox sends ?error=..., /callback must short-circuit with 400 and not call the exchange."""
    mock_exchange = mocker.patch("src.api.auth.UpstoxRestClient")
    response = client.get(
        "/callback?error=access_denied",
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert response.json() == {"error": "Authentication failed at provider"}
    mock_exchange.return_value.exchange_code_for_token.assert_not_called()


def test_callback_rejects_missing_code(mocker):
    """No code → 400, no exchange attempted."""
    mock_exchange = mocker.patch("src.api.auth.UpstoxRestClient")
    response = client.get("/callback", follow_redirects=False)
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid code"}
    mock_exchange.return_value.exchange_code_for_token.assert_not_called()


def test_callback_rejects_malformed_code(mocker):
    """Code with disallowed characters → 400."""
    mock_exchange = mocker.patch("src.api.auth.UpstoxRestClient")
    response = client.get(
        "/callback?code=has spaces&state=anystate",
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid code"}
    mock_exchange.return_value.exchange_code_for_token.assert_not_called()


def test_callback_rejects_missing_state(mocker):
    """Code present, state missing → 400, no exchange attempted."""
    mock_exchange = mocker.patch("src.api.auth.UpstoxRestClient")
    response = client.get("/callback?code=goodcode", follow_redirects=False)
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid state"}
    mock_exchange.return_value.exchange_code_for_token.assert_not_called()


def test_callback_rejects_state_mismatch(mocker):
    """State in URL doesn't match cookie → 400, no exchange attempted."""
    mock_exchange = mocker.patch("src.api.auth.UpstoxRestClient")
    response = client.get(
        "/callback?code=goodcode&state=urlstate",
        cookies={"oauth_state": "different_cookie_state"},
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid state"}
    mock_exchange.return_value.exchange_code_for_token.assert_not_called()


def test_callback_success_exchanges_and_redirects(mocker):
    """Matching state + valid code → exchange runs, token saved, redirect to /."""
    save_token = mocker.patch(
        "src.api.auth.save_token", new_callable=mocker.AsyncMock
    )
    mock_client = mocker.patch("src.api.auth.UpstoxRestClient")
    mock_client.return_value.exchange_code_for_token = mocker.AsyncMock(
        return_value="real_access_token"
    )
    mocker.patch(
        "src.api.auth.streamer_manager.start_market_data_streamer",
        new_callable=mocker.AsyncMock,
    )

    state = "matching_state_value_at_least_sixteen"
    response = client.get(
        f"/callback?code=goodcode&state={state}",
        cookies={"oauth_state": state},
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert response.headers["location"] == "/"
    save_token.assert_awaited_once_with("real_access_token")
