"""Chat streaming and MCP tool-call orchestration."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

import httpx

from mcp_manager import MCPManager
from server_runtime import ServerRuntime

TOOL_LOG_SEP = "\n\n---\n\n"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_TAIL_RE = re.compile(r"<think>.*$", re.DOTALL)
_GENERATING_DOTS = "Generating..."


class ChatEngine:
    def __init__(
        self,
        api: str,
        runtime: ServerRuntime,
        mcp: MCPManager,
        max_tool_iters: int = 5,
        tools_enabled: bool = True,
    ) -> None:
        self.api = api
        self.runtime = runtime
        self.mcp = mcp
        self.max_tool_iters = max_tool_iters
        self.tools_enabled = tools_enabled

    def openai_stream(self, messages: list[dict], model_id: str = ""):
        if not model_id:
            model_id = self.runtime.get_model_id() or "default"

        in_reasoning = False
        timeout = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)
        with httpx.stream(
            "POST",
            f"{self.api}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                try:
                    response.read()
                    detail = response.text[:300]
                except Exception:
                    detail = f"HTTP {response.status_code}"
                yield f"\n\n*[Server error: {detail}]*"
                return

            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(payload).get("choices", [{}])[0].get("delta", {})
                    reasoning = delta.get("reasoning_content")
                    if reasoning is not None:
                        if not in_reasoning:
                            in_reasoning = True
                            yield "<think>"
                        yield reasoning
                    token = delta.get("content")
                    if token is not None:
                        if in_reasoning:
                            in_reasoning = False
                            yield "</think>"
                        yield token
                except json.JSONDecodeError:
                    continue

        if in_reasoning:
            yield "</think>"

    def chat_stream(self, message: str, history: list, think_on: bool = False):
        if self.runtime.fetch_models() is None:
            yield "**Server offline** - please go to the Server tab to load a model."
            return

        if not message.strip():
            return

        model_id = self.runtime.cached_model_id or self.runtime.get_model_id()
        if not model_id:
            yield "**No model loaded** - please go to the Server tab and load a model first."
            return

        tools = self.mcp.get_tools() if self.tools_enabled else []
        system_prompt = self._system_prompt(bool(tools))

        messages = [{"role": "system", "content": system_prompt}]
        for item in history:
            role = self._message_role(item)
            content = self.normalize_content(self._message_content(item))
            if role == "user":
                messages.append({"role": "user", "content": content})
                continue
            if role == "assistant":
                clean = self.clean_history_content(content)
                if clean:
                    messages.append({"role": "assistant", "content": clean})

        messages.append({"role": "user", "content": message})

        if not tools:
            yield from self._stream_plain(messages, model_id, think_on)
            return

        tool_log: list[dict[str, Any]] = []
        for _ in range(self.max_tool_iters):
            yield self.format_tool_log(tool_log, _GENERATING_DOTS)
            try:
                response = httpx.post(
                    f"{self.api}/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                yield self.format_tool_log(tool_log, f"*[Error: {exc}]*")
                return

            choice = data.get("choices", [{}])[0]
            assistant_message = choice.get("message", {})
            tool_calls = assistant_message.get("tool_calls")

            if not tool_calls:
                prefix = self.format_tool_log(tool_log, "")
                full = prefix + TOOL_LOG_SEP if prefix else ""
                clean_messages = self._prepare_final_stream_messages(messages)
                streamed_any = False
                try:
                    for token in self.openai_stream(clean_messages, model_id):
                        streamed_any = True
                        full += token
                        yield self.strip_think(full, think_on)
                except httpx.RemoteProtocolError:
                    pass
                except Exception as exc:
                    full += f"\n\n*[Error: {exc}]*"

                if not streamed_any:
                    content = assistant_message.get("content", "") or ""
                    reasoning = assistant_message.get("reasoning_content", "") or ""
                    if reasoning:
                        content = f"<think>{reasoning}</think>{content}"
                    full += content

                yield self.strip_think(full, think_on)
                return

            messages.append(assistant_message)
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = function.get("name", "?")
                args_str = function.get("arguments", "{}")
                args = self.coerce_tool_args(name, self.parse_tool_args(args_str))
                yield self.format_tool_log(tool_log, f"Calling `{name}`...")
                result = self.mcp.call_tool(name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "content": str(result),
                    }
                )
                tool_log.append(
                    {
                        "name": name,
                        "args": args_str[:100],
                        "result": str(result)[:300],
                    }
                )

        clean_messages = self._prepare_final_stream_messages(messages)
        prefix = self.format_tool_log(tool_log, "")
        full = prefix + TOOL_LOG_SEP if prefix else ""
        try:
            for token in self.openai_stream(clean_messages, model_id):
                full += token
                yield self.strip_think(full, think_on)
        except httpx.RemoteProtocolError:
            pass
        except Exception as exc:
            full += f"\n\n*[Error: {exc}]*"

        yield self.strip_think(full, think_on)

    @staticmethod
    def _system_prompt(has_tools: bool) -> str:
        if has_tools:
            return (
                "You are a helpful assistant with access to tools. "
                "Call tools only when needed for verification or external data. "
                "If you call a tool, produce strict JSON arguments matching the tool schema exactly. "
                "Do not encode arrays or objects as strings. Be concise."
            )
        return (
            "You are a helpful assistant. "
            "Keep reasoning concise and answer clearly with minimal verbosity."
        )

    @staticmethod
    def normalize_content(content: Any) -> str:
        if isinstance(content, list):
            parts = []
            for item in content:
                parts.append(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                )
            return " ".join(part for part in parts if part)
        return content if isinstance(content, str) else str(content)

    @staticmethod
    def parse_tool_args(args_str: str) -> dict:
        raw = (args_str or "").strip()
        if not raw:
            return {}
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}

    @classmethod
    def coerce_tool_args(cls, name: str, args: dict) -> dict:
        coerced = cls._coerce_tool_value(args)
        if not isinstance(coerced, dict):
            return {}

        if name == "execute_python":
            deps = coerced.get("dependencies")
            if deps is None:
                coerced["dependencies"] = []
            elif isinstance(deps, str):
                dep_str = deps.strip()
                if not dep_str:
                    coerced["dependencies"] = []
                else:
                    parsed = cls._coerce_tool_value(dep_str)
                    coerced["dependencies"] = (
                        [str(item) for item in parsed]
                        if isinstance(parsed, list)
                        else [dep_str]
                    )
            elif isinstance(deps, list):
                coerced["dependencies"] = [str(item) for item in deps]
            else:
                coerced["dependencies"] = [str(deps)]

            timeout = coerced.get("timeout_seconds")
            if isinstance(timeout, (int, float)):
                coerced["timeout_seconds"] = int(timeout)

        if name == "run_process":
            if "command_line" not in coerced:
                command = coerced.get("cmd") or coerced.get("command")
                if isinstance(command, str) and command.strip():
                    coerced["command_line"] = command.strip()

            mode = coerced.get("mode")
            if not isinstance(mode, str) or mode not in {"shell", "executable"}:
                coerced["mode"] = "executable" if "argv" in coerced else "shell"

            if "timeout_ms" in coerced and isinstance(
                coerced["timeout_ms"], (int, float)
            ):
                coerced["timeout_ms"] = int(coerced["timeout_ms"])
            elif "timeout_seconds" in coerced and isinstance(
                coerced["timeout_seconds"], (int, float)
            ):
                coerced["timeout_ms"] = int(coerced["timeout_seconds"] * 1000)

        return coerced

    @classmethod
    def _coerce_tool_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._coerce_tool_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._coerce_tool_value(item) for item in value]
        if not isinstance(value, str):
            return value

        raw = value.strip()
        low = raw.lower()

        if low in {"null", "none"}:
            return None
        if low == "true":
            return True
        if low == "false":
            return False

        if raw.startswith(("[", "{")):
            for parser in (json.loads, ast.literal_eval):
                try:
                    return cls._coerce_tool_value(parser(raw))
                except Exception:
                    continue

        if re.fullmatch(r"-?\d+", raw):
            try:
                return int(raw)
            except Exception:
                pass

        if re.fullmatch(r"-?\d+\.\d+", raw):
            try:
                return float(raw)
            except Exception:
                pass

        return value

    @staticmethod
    def format_tool_log(log: list[dict], status: str) -> str:
        if not log and not status:
            return ""
        lines = []
        for entry in log:
            lines.append(f"[tool] {entry['name']}({entry['args']})")
            lines.append(f"  -> {entry['result'].replace(chr(10), ' ')}")
        if status:
            lines.append(f"  {status}")
        body = "\n".join(lines)
        return f"```tool-log\n{body}\n```"

    @staticmethod
    def _has_unclosed_think(text: str) -> bool:
        open_idx = text.rfind("<think>")
        return open_idx >= 0 and text.rfind("</think>") < open_idx

    @classmethod
    def strip_think(cls, text: str, think_on: bool = False) -> str:
        if think_on:
            return text + "</think>" if cls._has_unclosed_think(text) else text
        without_closed = _THINK_RE.sub("", text)
        return _OPEN_THINK_TAIL_RE.sub("", without_closed).strip()

    @staticmethod
    def clean_history_content(content: str) -> str:
        if TOOL_LOG_SEP in content:
            content = content.split(TOOL_LOG_SEP, 1)[1]
        return content.strip()

    @staticmethod
    def _prepare_final_stream_messages(messages: list[dict]) -> list[dict]:
        clean_messages: list[dict] = []
        tool_parts: list[str] = []

        for message in messages:
            role = ChatEngine._message_role(message)
            if role == "tool":
                tool_parts.append(
                    f"[Tool result: {str(ChatEngine._message_content(message))[:200]}]"
                )
                continue

            if role == "assistant":
                tool_calls = (
                    message.get("tool_calls") if isinstance(message, dict) else None
                )
                content = ChatEngine._message_content(message)
                if tool_calls and not content:
                    continue

            clean_messages.append(message)

        if tool_parts:
            clean_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Tool results summary:\n"
                        + "\n".join(tool_parts)
                        + "\n\nPlease provide your final answer based on the above."
                    ),
                }
            )

        return clean_messages

    @staticmethod
    def _message_role(message: Any) -> str:
        if isinstance(message, dict):
            role = message.get("role", "")
        else:
            role = getattr(message, "role", "")
        return role if isinstance(role, str) else str(role)

    @staticmethod
    def _message_content(message: Any) -> Any:
        if isinstance(message, dict):
            return message.get("content", "")
        return getattr(message, "content", "")

    def _stream_plain(
        self, messages: list[dict], model_id: str, think_on: bool = False
    ):
        full = ""
        try:
            for token in self.openai_stream(messages, model_id):
                full += token
                yield self.strip_think(full, think_on)
        except httpx.RemoteProtocolError:
            pass
        except Exception as exc:
            full += f"\n\n*[Error: {exc}]*"
        yield self.strip_think(full, think_on)
