import os
from typing import Optional

def get_env_variable(key: str, default_value: Optional[str] = None) -> str:
    value = os.getenv(key, default_value)
    if value is None:
        raise ValueError(f"Environment variable '{key}' is not set.")
    return value
