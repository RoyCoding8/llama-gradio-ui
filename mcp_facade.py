"""Synchronous facade used by UI handlers for MCP operations."""

from __future__ import annotations

import json

import gradio as gr

from mcp_manager import MCPManager, ServerConfig


class MCPFacade:
    def __init__(
        self,
        manager: MCPManager,
        allow_tool_execution: bool = True,
        remote_mode: bool = False,
    ) -> None:
        self.manager = manager
        self.allow_tool_execution = allow_tool_execution
        self.remote_mode = remote_mode

    def _execution_blocked_message(self) -> str:
        if self.remote_mode:
            return (
                "Tool execution is disabled while UI sharing is enabled. "
                "Set `ALLOW_REMOTE_TOOLS=1` to enable it."
            )
        return "Tool execution is disabled by configuration."

    def refresh(self):
        rows: list[str] = []
        server_ids: list[str] = []
        for server_id, cfg in self.manager.servers.items():
            rows.append(
                "| "
                + " | ".join(
                    [
                        cfg.name,
                        cfg.transport,
                        self.manager.status(server_id),
                        "Yes" if cfg.enabled else "No",
                        "Yes" if cfg.autostart else "No",
                    ]
                )
                + " |"
            )
            server_ids.append(server_id)

        if rows:
            servers_table = (
                "| Name | Transport | Status | Enabled | Autostart |\n"
                "|---|---|---|---|---|\n" + "\n".join(rows)
            )
        else:
            servers_table = (
                "No MCP servers configured. Add one below or import a config."
            )

        tools = self.manager.get_tools()
        if tools:
            tool_rows: list[str] = []
            tool_names: list[str] = []
            for tool in tools:
                function = tool["function"]
                tool_rows.append(
                    f"| {function['name']} | {function.get('description', '')[:80]} |"
                )
                tool_names.append(function["name"])
            tools_table = "| Tool | Description |\n|---|---|\n" + "\n".join(tool_rows)
        else:
            tools_table = "No tools available. Connect an enabled MCP server first."
            tool_names = []

        return (
            servers_table,
            gr.update(choices=server_ids, value=server_ids[0] if server_ids else None),
            tools_table,
            gr.update(choices=tool_names, value=tool_names[0] if tool_names else None),
        )

    def do_connect(self, server_id: str) -> str:
        if not server_id:
            return "No server selected."
        if not self.allow_tool_execution:
            return self._execution_blocked_message()
        try:
            names = self.manager.connect(server_id)
        except Exception as exc:
            return f"**Error:** {exc}"
        return f"Connected `{server_id}` - tools: {', '.join(names) or '(none)'}"

    def do_disconnect(self, server_id: str) -> str:
        if not server_id:
            return "No server selected."
        try:
            self.manager.disconnect(server_id)
        except Exception as exc:
            return f"**Error:** {exc}"
        return f"Disconnected `{server_id}`."

    def do_enable(self, server_id: str) -> str:
        if not server_id:
            return "No server selected."
        self.manager.set_enabled(server_id, True)
        return f"Enabled `{server_id}`."

    def do_disable(self, server_id: str) -> str:
        if not server_id:
            return "No server selected."
        self.manager.set_enabled(server_id, False)
        return f"Disabled `{server_id}` (disconnected if active)."

    def do_remove(self, server_id: str) -> str:
        if not server_id:
            return "No server selected."
        self.manager.remove_server(server_id)
        return f"Removed `{server_id}`."

    def do_toggle_autostart(self, server_id: str) -> str:
        if not server_id or server_id not in self.manager.servers:
            return "No server selected."
        cfg = self.manager.servers[server_id]
        cfg.autostart = not cfg.autostart
        self.manager.save()
        state = "enabled" if cfg.autostart else "disabled"
        return f"`{server_id}` autostart {state}."

    def do_add(self, name: str, transport: str, command: str, args_json: str) -> str:
        normalized_name = (name or "").strip()
        normalized_command = (command or "").strip()
        if not normalized_name or not normalized_command:
            return "Name and command/URL are required."

        args: list[str] = []
        if args_json and args_json.strip():
            try:
                parsed_args = json.loads(args_json)
            except json.JSONDecodeError as exc:
                return f"Invalid JSON for args: {exc}"
            if not isinstance(parsed_args, list):
                return "Args must be a JSON array."
            args = [str(item) for item in parsed_args]

        server_id = normalized_name.lower().replace(" ", "-").replace("_", "-")
        if server_id in self.manager.servers:
            return f"Server `{server_id}` already exists. Choose a unique name."

        self.manager.add_server(
            ServerConfig(
                id=server_id,
                name=normalized_name,
                transport=transport,
                command=normalized_command,
                args=args,
            )
        )
        return f"Added server `{server_id}`. Click **Connect** to start it."

    def do_import(self, json_str: str) -> str:
        if not json_str or not json_str.strip():
            return "Paste a JSON config to import."
        try:
            added = self.manager.import_config(json_str)
        except Exception as exc:
            return f"**Error:** {exc}"
        return f"Imported {len(added)} server(s): {', '.join(added)}"

    def do_test(self, tool_name: str, args_json: str):
        if not tool_name:
            return "No tool selected."
        if not self.allow_tool_execution:
            return self._execution_blocked_message()

        args: dict = {}
        if args_json and args_json.strip():
            try:
                args = json.loads(args_json)
            except json.JSONDecodeError as exc:
                return f"Invalid JSON: {exc}"

        try:
            return self.manager.call_tool(tool_name, args)
        except Exception as exc:
            return f"Error: {exc}"
