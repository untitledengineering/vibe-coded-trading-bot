from src.utils.config import get_env_var

def test_get_env_var_default():
    """Verify that get_env_var returns the default value when key is missing."""
    assert get_env_var("NON_EXISTENT_VAR", "default_val") == "default_val"

def test_get_env_var_loaded(monkeypatch):
    """Verify that get_env_var retrieves a value set in the environment."""
    monkeypatch.setenv("TEST_VAR", "test_val")
    assert get_env_var("TEST_VAR") == "test_val"
