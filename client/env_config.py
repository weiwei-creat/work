import json
import os
from pathlib import Path


CLIENT_DIR = Path(__file__).resolve().parent
SAGE_ROOT = CLIENT_DIR.parent


def _clean_env_value(value: str) -> str:
    value = value.strip().strip('"').strip("'")
    value = value.replace("[1m", "").rstrip("]")
    return value


def _load_dotenv(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_env_value(value)
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def _load_key_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_client_config() -> dict:
    env = _load_dotenv(SAGE_ROOT / ".env")
    key_json = _load_key_json(CLIENT_DIR / "key.json")

    def get(name: str, default: str = "") -> str:
        return os.getenv(name) or env.get(name) or key_json.get(name, default)

    return {
        "API_TOKEN": get("API_TOKEN") or get("QWEN_API_KEY") or get("DASHSCOPE_API_KEY"),
        "API_URL_QWEN": get("API_URL_QWEN") or get("QWEN_BASE_URL"),
        "MODEL_NAME": get("QWEN_MODEL") or get("MODEL_NAME", "qwen-vl-max-latest"),
    }
