"""Runtime management for llama-server."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import httpx

_KV_TYPES = {
    "f32",
    "f16",
    "bf16",
    "q8_0",
    "q4_0",
    "q4_1",
    "iq4_nl",
    "q5_0",
    "q5_1",
}


class ServerRuntime:
    def __init__(
        self,
        api: str,
        host: str,
        port: int,
        server_dir: str,
        gguf_dir: str,
        ctx_size: int,
        gpu_layers: int,
        kv_cache_type_k: str,
        kv_cache_type_v: str,
    ) -> None:
        self.api = api
        self.host = host
        self.port = port
        self.server_dir = server_dir
        self.gguf_dir = gguf_dir
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers
        self.kv_cache_type_k = kv_cache_type_k
        self.kv_cache_type_v = kv_cache_type_v
        self._server_proc: subprocess.Popen | None = None
        self._cached_model_id = ""

    @property
    def cached_model_id(self) -> str:
        return self._cached_model_id

    def fetch_models(self) -> list[str] | None:
        try:
            response = httpx.get(f"{self.api}/v1/models", timeout=2)
            if response.status_code != 200:
                return None
            payload = response.json()
        except Exception:
            return None

        model_ids = []
        for model in payload.get("data", []):
            model_id = model.get("id") if isinstance(model, dict) else None
            if isinstance(model_id, str) and model_id:
                model_ids.append(model_id)
        return model_ids

    def get_model_id(self) -> str:
        models = self.fetch_models() or []
        self._cached_model_id = models[0] if models else ""
        return self._cached_model_id

    def server_status(self) -> str:
        models = self.fetch_models()
        if models is None:
            return "**Server offline** - load a model to start."
        if models:
            return f"**Server online** - model loaded: `{models[0]}`"
        return "**Server online** - no model loaded"

    def refresh_status(self) -> str:
        self.get_model_id()
        return self.server_status()

    def scan_gguf_files(self) -> list[str]:
        directory = Path(self.gguf_dir) if self.gguf_dir else None
        if not directory or not directory.is_dir():
            return []
        return sorted(path.name for path in directory.glob("*.gguf"))

    def start_server(self, gguf_name: str, custom_path: str) -> str:
        model_path = self._resolve_model_path(gguf_name, custom_path)
        if isinstance(model_path, str):
            return model_path

        executable = self._resolve_server_executable()
        if not executable:
            return (
                f"**Error:** llama-server executable not found in `{self.server_dir}`"
            )

        self.stop_server()

        ngl = str(self.gpu_layers) if self.gpu_layers >= 0 else "999"
        command = [
            str(executable),
            "-m",
            str(model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-ngl",
            ngl,
            "--ctx-size",
            str(self.ctx_size),
        ]

        cache_k = self._validated_kv_type(self.kv_cache_type_k)
        cache_v = self._validated_kv_type(self.kv_cache_type_v)
        if cache_k:
            command.extend(["--cache-type-k", cache_k])
        if cache_v:
            command.extend(["--cache-type-v", cache_v])

        env = os.environ.copy()
        if "sycl" in (self.server_dir or "").lower():
            env.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
            env.setdefault("ZES_ENABLE_SYSMAN", "1")
            env.setdefault("SYCL_CACHE_PERSISTENT", "1")

        try:
            self._server_proc = subprocess.Popen(
                command,
                cwd=str(executable.parent),
                env=env,
            )
        except Exception as exc:
            return f"**Failed to start:** {exc}"

        for _ in range(30):
            time.sleep(1)
            if self.fetch_models() is not None:
                return f"Server started - `{model_path.name}` loaded"
            if self._server_proc.poll() is not None:
                return f"**Server exited** (code {self._server_proc.returncode})"

        return "Server started but not responding yet. Click Refresh Status."

    def stop_server(self) -> str:
        if self._server_proc is None:
            return "No server process to stop."

        try:
            self._server_proc.terminate()
            self._server_proc.wait(timeout=5)
        except Exception:
            try:
                self._server_proc.kill()
                self._server_proc.wait(timeout=5)
            except Exception:
                pass
        finally:
            self._server_proc = None

        return "Server stopped. VRAM freed."

    def cleanup(self) -> None:
        if self._server_proc and self._server_proc.poll() is None:
            try:
                self._server_proc.terminate()
                self._server_proc.wait(timeout=3)
            except Exception:
                try:
                    self._server_proc.kill()
                    self._server_proc.wait(timeout=3)
                except Exception:
                    pass
        self._server_proc = None

    def _resolve_model_path(self, gguf_name: str, custom_path: str) -> Path | str:
        raw_custom_path = (custom_path or "").strip().strip("\"'")
        if raw_custom_path:
            model_path = Path(raw_custom_path)
            if not model_path.exists():
                return f"**Error:** File not found: `{raw_custom_path}`"
            if model_path.suffix.lower() != ".gguf":
                return f"**Error:** Not a GGUF file: `{model_path.name}`"
            return model_path

        if gguf_name and gguf_name.strip():
            model_path = Path(self.gguf_dir) / gguf_name
            if not model_path.exists():
                return f"**Error:** `{gguf_name}` not found in `{self.gguf_dir}`"
            return model_path

        return "Select a model from the dropdown or paste a GGUF path."

    def _resolve_server_executable(self) -> Path | None:
        names = (
            ["llama-server.exe", "llama-server"]
            if os.name == "nt"
            else ["llama-server", "llama-server.exe"]
        )
        base = Path(self.server_dir) if self.server_dir else Path()
        for name in names:
            candidate = base / name
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _validated_kv_type(value: str) -> str:
        return value if value in _KV_TYPES else ""
