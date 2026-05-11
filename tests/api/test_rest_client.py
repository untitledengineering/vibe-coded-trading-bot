import pytest
from src.api.rest_client import UpstoxRestClient

def test_get_profile_success(mocker):
    """Verify get_profile returns data when the API call is successful."""
    # Mock the UserApi instance
    mock_user_api = mocker.patch("upstox_client.api.user_api.UserApi")
    mock_response = mocker.Mock()
    mock_response.to_dict.return_value = {"data": {"user_id": "TEST1234"}}
    mock_user_api.return_value.get_profile.return_value = mock_response

    client = UpstoxRestClient(access_token="fake_token")
    profile = client.get_profile()

    assert profile["data"]["user_id"] == "TEST1234"
    mock_user_api.return_value.get_profile.assert_called_once_with(api_version="2.0")

def test_get_profile_failure(mocker):
    """Verify get_profile raises an exception when the API call fails."""
    mock_user_api = mocker.patch("upstox_client.api.user_api.UserApi")
    mock_user_api.return_value.get_profile.side_effect = Exception("API Error")

    client = UpstoxRestClient(access_token="fake_token")
    
    with pytest.raises(Exception) as excinfo:
        client.get_profile()
    
    assert "API Error" in str(excinfo.value)
