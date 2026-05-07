import pytest
import json
from unittest.mock import MagicMock, patch
from src.auth import get_login_url, exchange_code_for_token, get_access_token

@patch("src.auth.API_KEY", "test_key")
@patch("src.auth.REDIRECT_URI", "http://test.com")
def test_get_login_url():
    """
    Test that the login URL is generated correctly.
    """
    url = get_login_url()
    assert "client_id=test_key" in url
    assert "redirect_uri=http%3A%2F%2Ftest.com" in url

@patch("upstox_client.LoginApi")
@patch("src.auth.API_KEY", "test_key")
@patch("src.auth.API_SECRET", "test_secret")
@patch("src.auth.REDIRECT_URI", "http://test.com")
@patch("src.auth._save_token_data")
def test_exchange_code_for_token(mock_save, mock_login_api):
    """
    Test that the code is exchanged for a token and saved.
    """
    mock_instance = mock_login_api.return_value
    mock_response = MagicMock()
    mock_response.access_token = "test_token"
    mock_response.expires_in = 3600
    mock_instance.token.return_value = mock_response

    with patch("time.time", return_value=1000):
        token = exchange_code_for_token("test_code")

    assert token == "test_token"
    mock_save.assert_called_once_with("test_token", 4600)

def test_get_access_token_exists(tmp_path):
    """
    Test retrieving token when file exists and not expired.
    """
    token_file = tmp_path / ".token"
    data = {"access_token": "file_token", "expires_at": 2000}
    token_file.write_text(json.dumps(data))

    with patch("src.auth.TOKEN_FILE", token_file), patch("time.time", return_value=1000):
        assert get_access_token() == "file_token"

def test_get_access_token_expired(tmp_path):
    """
    Test retrieving token when file exists but expired.
    """
    token_file = tmp_path / ".token"
    data = {"access_token": "file_token", "expires_at": 500}
    token_file.write_text(json.dumps(data))

    with patch("src.auth.TOKEN_FILE", token_file), patch("time.time", return_value=1000):
        assert get_access_token() is None

def test_get_access_token_not_exists(tmp_path):
    """
    Test retrieving token when file doesn't exist.
    """
    token_file = tmp_path / ".token"

    with patch("src.auth.TOKEN_FILE", token_file):
        assert get_access_token() is None
