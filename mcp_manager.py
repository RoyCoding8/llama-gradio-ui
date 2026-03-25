"""MCP server manager — connects to MCP servers and exposes tools to the LLM.

Runs an asyncio event loop in a background thread for MCP protocol operations.
All public methods are synchronous and safe to call from Gradio handlers.
"""

import asyncio
import json
import logging
import threading
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
_KNOWN_FIELDS: set[str] | None = None


@dataclass
class ServerConfig:
    id: str
    name: str
    transport: str  # stdio | sse | http
    command: str  # executable (stdio) or URL (sse/http)
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    enabled: bool = True
    autostart: bool = False


def _filter_fields(d: dict) -> dict:
    global _KNOWN_FIELDS
    if _KNOWN_FIELDS is None:
        _KNOWN_FIELDS = {f.name for f in fields(ServerConfig)}
    return {k: v for k, v in d.items() if k in _KNOWN_FIELDS}


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class MCPManager:
    def __init__(self, config_path: str | Path):
        self._config_path = Path(config_path)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self.servers: dict[str, ServerConfig] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._conn_tasks: dict[str, asyncio.Task] = {}
        self._tool_map: dict[str, str] = {}  # tool_name -> server_id
        self.load()

    def _run(self, coro, timeout=60):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(
            timeout=timeout
        )

    # --- Persistence --------------------------------------------------------

    def load(self):
        if not self._config_path.exists():
            return
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            for sid, raw in data.get("servers", {}).items():
                raw["id"] = sid
                self.servers[sid] = ServerConfig(**_filter_fields(raw))
        except Exception as e:
            log.warning("Failed to load MCP config: %s", e)

    def save(self):
        out = {}
        for sid, cfg in self.servers.items():
            d = asdict(cfg)
            d.pop("id")
            out[sid] = d
        self._config_path.write_text(
            json.dumps({"servers": out}, indent=2), encoding="utf-8"
        )

    # --- Server CRUD --------------------------------------------------------

    def add_server(self, cfg: ServerConfig) -> str:
        if cfg.id in self.servers:
            raise ValueError(f"Server '{cfg.id}' already exists")
        self.servers[cfg.id] = cfg
        self.save()
        return cfg.id

    def remove_server(self, sid: str):
        if sid in self._sessions:
            self.disconnect(sid)
        self.servers.pop(sid, None)
        self.save()

    def set_enabled(self, sid: str, enabled: bool):
        if sid not in self.servers:
            return
        self.servers[sid].enabled = enabled
        if not enabled and sid in self._sessions:
            self.disconnect(sid)
        self.save()

    def status(self, sid: str) -> str:
        if sid in self._sessions:
            return "connected"
        return "disconnected" if sid in self.servers else "unknown"

    # --- Connect / Disconnect -----------------------------------------------

    def connect(self, sid: str) -> list[str]:
        """Connect to server, returns list of discovered tool names."""
        cfg = self.servers.get(sid)
        if not cfg:
            raise ValueError(f"Unknown server: {sid}")
        return self._run(self._async_connect(cfg))

    def disconnect(self, sid: str):
        self._run(self._async_disconnect(sid))

    async def _async_connect(self, cfg: ServerConfig) -> list[str]:
        if cfg.id in self._sessions:
            await self._async_disconnect(cfg.id)

        from mcp import ClientSession, StdioServerParameters
        from mcp import types as mcp_types

        stop_event = asyncio.Event()
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            self._connection_task(
                cfg, stop_event, ready, ClientSession, StdioServerParameters, mcp_types
            )
        )
        try:
            session, tools = await ready
            names = []
            for t in tools:
                if t.name in self._tool_map:
                    log.warning(
                        "Tool '%s' already registered by '%s', skipping",
                        t.name,
                        self._tool_map[t.name],
                    )
                    continue
                self._tool_map[t.name] = cfg.id
                names.append(t.name)
            self._sessions[cfg.id] = {"session": session, "tools": tools}
            self._stop_events[cfg.id] = stop_event
            self._conn_tasks[cfg.id] = task
            return names
        except Exception:
            stop_event.set()
            task.cancel()
            try:
                await task
            except Exception:
                pass
            raise

    async def _connection_task(
        self, cfg, stop_event, ready, client_session_cls, stdio_params_cls, mcp_types
    ):
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            if cfg.transport == "stdio":
                from mcp.client.stdio import stdio_client

                params = stdio_params_cls(
                    command=cfg.command,
                    args=cfg.args or [],
                    env=cfg.env or None,
                    cwd=cfg.cwd or None,
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            elif cfg.transport == "sse":
                from mcp.client.sse import sse_client

                read, write = await stack.enter_async_context(
                    sse_client(url=cfg.command)
                )
            elif cfg.transport == "http":
                from mcp.client.streamable_http import streamablehttp_client

                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(url=cfg.command)
                )
            else:
                raise ValueError(f"Unsupported transport: {cfg.transport}")

            session = await stack.enter_async_context(
                client_session_cls(
                    read_stream=read,
                    write_stream=write,
                    client_info=mcp_types.Implementation(
                        name="llama-gradio-ui", version="0.1.0"
                    ),
                )
            )
            await session.initialize()
            tools = (await session.list_tools()).tools
            if not ready.done():
                ready.set_result((session, tools))
            await stop_event.wait()
        except Exception as e:
            if not ready.done():
                ready.set_exception(e)
            else:
                log.warning("MCP connection task error for %s: %s", cfg.id, e)
        finally:
            try:
                await stack.aclose()
            except Exception as e:
                log.warning("Error closing stack for %s: %s", cfg.id, e)

    async def _async_disconnect(self, sid: str):
        self._tool_map = {k: v for k, v in self._tool_map.items() if v != sid}
        self._sessions.pop(sid, None)
        stop_event = self._stop_events.pop(sid, None)
        task = self._conn_tasks.pop(sid, None)
        if stop_event:
            stop_event.set()
        if task:
            try:
                await task
            except Exception as e:
                log.warning("Error disconnecting %s: %s", sid, e)

    # --- Tools --------------------------------------------------------------

    def get_tools(self) -> list[dict]:
        """OpenAI-format tool list from all enabled + connected servers."""
        tools = []
        for sid, data in self._sessions.items():
            cfg = self.servers.get(sid)
            if not cfg or not cfg.enabled:
                continue
            for t in data["tools"]:
                if self._tool_map.get(t.name) != sid:
                    continue
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": t.inputSchema
                            or {"type": "object", "properties": {}},
                        },
                    }
                )
        return tools

    def get_server_tools(self, sid: str) -> list[dict]:
        data = self._sessions.get(sid)
        if not data:
            return []
        return [
            {"name": t.name, "description": t.description or ""} for t in data["tools"]
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        return self._run(self._async_call_tool(name, arguments))

    async def _async_call_tool(self, name: str, arguments: dict) -> str:
        sid = self._tool_map.get(name)
        if not sid or sid not in self._sessions:
            return f"Error: tool '{name}' not found or server disconnected"
        try:
            result = await self._sessions[sid]["session"].call_tool(name, arguments)
            parts = []
            for c in result.content:
                if hasattr(c, "text"):
                    parts.append(c.text)
                elif hasattr(c, "data"):
                    parts.append(f"[binary: {getattr(c, 'mimeType', 'unknown')}]")
                else:
                    parts.append(str(c))
            return "\n".join(parts) or "(empty result)"
        except Exception as e:
            return f"Error calling '{name}': {e}"

    # --- Import -------------------------------------------------------------

    def import_config(self, json_str: str) -> list[str]:
        """Import Claude/Cursor-format config. Returns added server IDs."""
        data = json.loads(json_str)
        servers = data.get("mcpServers", data.get("servers", {}))
        if not isinstance(servers, dict):
            raise ValueError("Invalid config: 'servers' must be an object")

        added = []
        for name, raw in servers.items():
            if not isinstance(raw, dict):
                continue

            sid = name.lower().replace(" ", "-").replace("_", "-")
            if sid in self.servers:
                continue

            transport = str(raw.get("transport", "stdio")).lower().strip() or "stdio"
            if transport not in {"stdio", "sse", "http"}:
                transport = "stdio"

            command = raw.get("command")
            if not command:
                command = raw.get("url", "")
                if transport == "stdio" and command:
                    transport = "sse"

            if not isinstance(command, str) or not command.strip():
                continue

            raw_args = raw.get("args", [])
            args = [str(a) for a in raw_args] if isinstance(raw_args, list) else []
            env = raw.get("env", {}) if isinstance(raw.get("env", {}), dict) else {}

            self.add_server(
                ServerConfig(
                    id=sid,
                    name=name,
                    transport=transport,
                    command=command.strip(),
                    args=args,
                    env=env,
                    enabled=bool(raw.get("enabled", True)),
                    autostart=bool(raw.get("autostart", False)),
                )
            )
            added.append(sid)
        return added

    # --- Lifecycle ----------------------------------------------------------

    def autostart(self):
        for sid, cfg in self.servers.items():
            if cfg.enabled and cfg.autostart:
                try:
                    self.connect(sid)
                except Exception as e:
                    log.warning("Autostart failed for %s: %s", sid, e)

    def shutdown(self):
        for sid in list(self._sessions):
            try:
                self.disconnect(sid)
            except Exception as e:
                log.warning("Shutdown error for %s: %s", sid, e)
        self._loop.call_soon_threadsafe(self._loop.stop)
