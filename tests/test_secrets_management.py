"""
Unit tests for secrets management, .env loading, masking, and .gitignore rules
"""

import os
from pathlib import Path
import pytest

from src.config import AppConfig, load_config, mask_secrets


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_env_example_file_structure():
    """Verify .env.example exists and documents essential security and configuration keys"""
    example_path = REPO_ROOT / ".env.example"
    assert example_path.exists(), ".env.example must exist in the root directory"

    content = example_path.read_text(encoding="utf-8")
    expected_keys = [
        "API_SECRET_KEY",
        "API_HOST",
        "API_PORT",
        "CORS_ORIGINS",
        "TICKERS",
        "LOOKBACK_YEARS",
        "EVENT_WINDOW",
        "ESTIMATION_WINDOW",
        "LABEL_THRESHOLD",
        "TEST_SIZE",
        "MODELS_TO_TRAIN",
        "DATA_DIR",
        "MODELS_DIR",
        "VIZ_DIR",
    ]
    for key in expected_keys:
        assert f"{key}=" in content, f"Expected {key} in .env.example"


def test_gitignore_ignores_env_files():
    """Verify .gitignore properly blocks .env and allows .env.example"""
    gitignore_path = REPO_ROOT / ".gitignore"
    assert gitignore_path.exists(), ".gitignore must exist"

    content = gitignore_path.read_text(encoding="utf-8")
    assert ".env" in content
    assert "!.env.example" in content
    # Ensure raw markdown formatting artifacts aren't in .gitignore
    assert not content.strip().startswith("```"), ".gitignore should not start with markdown code fence"


def test_dotenv_loading_via_load_config(tmp_path):
    """Verify load_config correctly loads sensitive variables from a custom .env file"""
    env_file = tmp_path / ".env"
    secret_val = "super-secret-jwt-token-xyz-12345"
    env_file.write_text(
        f"API_SECRET_KEY={secret_val}\n"
        f"APP_PORT=9090\n"
        f"TICKERS=NVDA,AMD,INTC\n",
        encoding="utf-8"
    )

    # Load with custom env_file
    config = load_config(env_file=env_file)
    assert config["api_secret_key"] == secret_val
    assert config["tickers"] == ["NVDA", "AMD", "INTC"]


def test_secret_masking_in_repr_and_helper():
    """Verify mask_secrets and AppConfig.__repr__ hide secret keys"""
    raw_cfg = {
        "api_secret_key": "very-secret-token",
        "jwt_password": "super-sensitive-pass",
        "database_url": "postgresql://user:pass@localhost:5432/db",
        "tickers": ["AAPL", "MSFT"],
        "lookback_years": 5,
    }

    masked = mask_secrets(raw_cfg)
    assert masked["api_secret_key"] == "***"
    assert masked["jwt_password"] == "***"
    assert masked["tickers"] == ["AAPL", "MSFT"]
    assert masked["lookback_years"] == 5

    app_config = AppConfig(raw_cfg)
    repr_str = repr(app_config)
    assert "very-secret-token" not in repr_str
    assert "super-sensitive-pass" not in repr_str
    assert "***" in repr_str
