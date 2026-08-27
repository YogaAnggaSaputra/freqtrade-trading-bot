import os
from pathlib import Path
from typing import Optional

SECRETS_DIR = Path("/run/secrets")


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    secret_path = SECRETS_DIR / name
    if secret_path.exists():
        return secret_path.read_text().strip()
    env_value = os.getenv(name.upper())
    if env_value:
        return env_value
    return default


def load_secrets_into_env(prefix: str = "") -> None:
    if not SECRETS_DIR.exists():
        return
    for secret_file in SECRETS_DIR.iterdir():
        if secret_file.is_file():
            key = f"{prefix}{secret_file.name.upper()}"
            os.environ[key] = secret_file.read_text().strip()


class SecretManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
        return cls._instance

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        if name in self._cache:
            return self._cache[name]
        value = get_secret(name, default)
        self._cache[name] = value
        return value

    def clear_cache(self):
        self._cache.clear()