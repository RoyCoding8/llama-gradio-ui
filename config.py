"""Configuration values loaded from environment and local .env file."""

from __future__ import annotations

import os
from pathlib import Path


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = _strip_wrapping_quotes(value.strip())
        os.environ.setdefault(key, value)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = _env(key, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


_BASE_DIR = Path(__file__).parent
_load_dotenv(_BASE_DIR / ".env")

HOST = _env("LLAMA_HOST", "127.0.0.1")
PORT = _env_int("LLAMA_PORT", 8080)
API = f"http://{HOST}:{PORT}"

LLAMA_SERVER_DIR = _env("LLAMA_SERVER_DIR")
GGUF_DIR = _env("GGUF_DIR", LLAMA_SERVER_DIR)

UI_HOST = _env("UI_HOST", "127.0.0.1")
UI_PORT = _env_int("UI_PORT", 7860)
UI_SHARE = _env_bool("UI_SHARE", default=False)
ALLOW_REMOTE_TOOLS = _env_bool("ALLOW_REMOTE_TOOLS", default=False)

CTX_SIZE = _env_int("CTX_SIZE", 4096)
GPU_LAYERS = _env_int("GPU_LAYERS", -1)
KV_CACHE_TYPE_K = _env("KV_CACHE_TYPE_K", "").lower()
KV_CACHE_TYPE_V = _env("KV_CACHE_TYPE_V", "").lower()

MCP_CONFIG_PATH = str(_BASE_DIR / "mcp_servers.json")
