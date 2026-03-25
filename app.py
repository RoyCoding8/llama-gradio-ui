"""Gradio application entry point for local llama-server chat."""

from __future__ import annotations

import atexit
import sys
from pathlib import Path

import gradio as gr

from chat_engine import ChatEngine
from config import (
    ALLOW_REMOTE_TOOLS,
    API,
    CTX_SIZE,
    GGUF_DIR,
    GPU_LAYERS,
    HOST,
    KV_CACHE_TYPE_K,
    KV_CACHE_TYPE_V,
    LLAMA_SERVER_DIR,
    MCP_CONFIG_PATH,
    PORT,
    UI_HOST,
    UI_PORT,
    UI_SHARE,
)
from mcp_facade import MCPFacade
from mcp_manager import MCPManager
from privacy_shield import PrivacyShield
from server_runtime import ServerRuntime

MAX_TOOL_ITERS = 5

runtime = ServerRuntime(
    api=API,
    host=HOST,
    port=PORT,
    server_dir=LLAMA_SERVER_DIR,
    gguf_dir=GGUF_DIR,
    ctx_size=CTX_SIZE,
    gpu_layers=GPU_LAYERS,
    kv_cache_type_k=KV_CACHE_TYPE_K,
    kv_cache_type_v=KV_CACHE_TYPE_V,
)

mcp = MCPManager(MCP_CONFIG_PATH)

REMOTE_MODE = UI_SHARE
TOOLS_ENABLED = not REMOTE_MODE or ALLOW_REMOTE_TOOLS
if REMOTE_MODE and not ALLOW_REMOTE_TOOLS:
    REMOTE_TOOLS_WARNING = (
        "MCP tool execution is disabled while `UI_SHARE=1`. "
        "Set `ALLOW_REMOTE_TOOLS=1` only if you trust everyone with access to the shared link."
    )
else:
    REMOTE_TOOLS_WARNING = ""

chat_engine = ChatEngine(
    api=API,
    runtime=runtime,
    mcp=mcp,
    max_tool_iters=MAX_TOOL_ITERS,
    tools_enabled=TOOLS_ENABLED,
)

privacy = PrivacyShield(
    fetch_models=runtime.fetch_models, openai_stream=chat_engine.openai_stream
)
mcp_ui = MCPFacade(mcp, allow_tool_execution=TOOLS_ENABLED, remote_mode=REMOTE_MODE)


def refresh_status() -> str:
    return runtime.refresh_status()


def start_server(gguf_name: str, custom_path: str) -> str:
    return runtime.start_server(gguf_name, custom_path)


def stop_server() -> str:
    return runtime.stop_server()


def chat_stream(message: str, history: list, think_on: bool = False):
    yield from chat_engine.chat_stream(message, history, think_on)


def scrub_pii(raw_text: str):
    return privacy.scrub_pii(raw_text)


def restyle_text(scrubbed_text: str):
    yield from privacy.restyle_text(scrubbed_text)


def mcp_refresh():
    return mcp_ui.refresh()


def mcp_do_connect(server_id: str):
    return mcp_ui.do_connect(server_id)


def mcp_do_disconnect(server_id: str):
    return mcp_ui.do_disconnect(server_id)


def mcp_do_enable(server_id: str):
    return mcp_ui.do_enable(server_id)


def mcp_do_disable(server_id: str):
    return mcp_ui.do_disable(server_id)


def mcp_do_remove(server_id: str):
    return mcp_ui.do_remove(server_id)


def mcp_do_toggle_autostart(server_id: str):
    return mcp_ui.do_toggle_autostart(server_id)


def mcp_do_add(name: str, transport: str, command: str, args_json: str):
    return mcp_ui.do_add(name, transport, command, args_json)


def mcp_do_import(json_str: str):
    return mcp_ui.do_import(json_str)


def mcp_do_test(tool_name: str, args_json: str):
    return mcp_ui.do_test(tool_name, args_json)


@atexit.register
def _cleanup() -> None:
    runtime.cleanup()
    mcp.shutdown()


def create_ui():
    css_file = Path(__file__).parent / "style.css"
    css = css_file.read_text(encoding="utf-8") if css_file.exists() else ""

    with gr.Blocks(title="Local LLM Chat", elem_id="rf-app") as app:
        with gr.Row(elem_id="rf-header-row"):
            gr.Markdown(
                "# Local LLM Chat\nRuns against your local llama.cpp server.",
                elem_id="rf-header",
            )
            dark_btn = gr.Button("🌙", elem_id="rf-theme-toggle", scale=0, min_width=42)

        dark_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            js="""() => {
                const root = document.documentElement;
                const body = document.body;
                const dark = !root.classList.contains('dark');
                root.classList.toggle('dark', dark);
                if (body) body.classList.toggle('dark', dark);
                localStorage.setItem('rf-theme', dark ? 'dark' : 'light');
                const btn = document.querySelector('#rf-theme-toggle button') || document.querySelector('#rf-theme-toggle');
                if (btn) {
                    btn.textContent = dark ? '☀️' : '🌙';
                    btn.title = dark ? 'Switch to light mode' : 'Switch to dark mode';
                    btn.setAttribute('aria-label', btn.title);
                }
            }""",
        )
        app.load(
            fn=None,
            inputs=None,
            outputs=None,
            js="""() => {
                const root = document.documentElement;
                const body = document.body;
                const saved = localStorage.getItem('rf-theme');
                const dark = saved === 'dark';
                root.classList.toggle('dark', dark);
                if (body) body.classList.toggle('dark', dark);
                const btn = document.querySelector('#rf-theme-toggle button') || document.querySelector('#rf-theme-toggle');
                if (btn) {
                    btn.textContent = dark ? '☀️' : '🌙';
                    btn.title = dark ? 'Switch to light mode' : 'Switch to dark mode';
                    btn.setAttribute('aria-label', btn.title);
                }
            }""",
        )

        with gr.Tabs():
            with gr.Tab("Server", id="server-tab"):
                gr.Markdown("### Server")
                status_md = gr.Markdown("Checking...")
                refresh_btn = gr.Button("Refresh Status", scale=0)

                gr.Markdown("---\n### Load Model")
                gguf_files = runtime.scan_gguf_files()
                gguf_dd = gr.Dropdown(
                    label="Models in GGUF folder",
                    choices=gguf_files,
                    value=gguf_files[0] if gguf_files else None,
                    interactive=True,
                )
                custom_path = gr.Textbox(
                    label="Or paste a custom GGUF path",
                    placeholder=r"e.g. C:\models\Qwen3.5-9B.gguf",
                    lines=1,
                    interactive=True,
                )
                with gr.Row():
                    load_btn = gr.Button("Load Model", variant="primary", scale=0)
                    stop_btn = gr.Button(
                        "Stop Server", elem_classes=["stop-btn"], scale=0
                    )
                    gr.Column(scale=1)
                action_md = gr.Markdown("")

                gr.Markdown(
                    "---\n### Getting Started\n\n"
                    "1. Select a `.gguf` model or paste a custom path\n"
                    "2. Click **Load Model** to launch the server\n"
                    "3. Open the **Chat** tab and start a conversation\n\n"
                    f"**Server:** `{API}`  \n"
                    f"**llama.cpp dir:** `{runtime.server_dir or '(not set)'}`  \n"
                    f"**GGUF dir:** `{runtime.gguf_dir or '(not set)'}`"
                )

            with gr.Tab("Chat", id="chat-tab"):
                if REMOTE_TOOLS_WARNING:
                    gr.Markdown(f"⚠️ {REMOTE_TOOLS_WARNING}")

                think_state = gr.State(False)
                chatbot = gr.Chatbot(
                    height=600,
                    render_markdown=True,
                    reasoning_tags=[("<think>", "</think>")],
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                        {"left": "\\\\[", "right": "\\\\]", "display": True},
                        {"left": "\\\\(", "right": "\\\\)", "display": False},
                    ],
                    placeholder="Load a model in the Server tab, then start chatting.",
                )
                gr.ChatInterface(
                    fn=chat_stream, chatbot=chatbot, additional_inputs=[think_state]
                )
                with gr.Row(elem_id="rf-think-row"):
                    think_btn = gr.Button(
                        "Think: OFF",
                        elem_id="rf-think-toggle",
                        elem_classes=["secondary"],
                        scale=0,
                        min_width=120,
                    )

                def _toggle_think(current):
                    new_value = not current
                    return new_value, gr.update(
                        value=f"Think: {'ON' if new_value else 'OFF'}"
                    )

                think_btn.click(
                    _toggle_think,
                    inputs=[think_state],
                    outputs=[think_state, think_btn],
                )

            with gr.Tab("MCP Servers", id="mcp-tab"):
                gr.Markdown(
                    "### MCP Server Manager\n"
                    "Attach MCP servers so the model can call tools "
                    "(file system, web, databases, and more)."
                )
                if REMOTE_TOOLS_WARNING:
                    gr.Markdown(f"⚠️ {REMOTE_TOOLS_WARNING}")

                mcp_status_md = gr.Markdown("Click **Refresh** to load server list.")
                with gr.Row():
                    mcp_refresh_btn = gr.Button("Refresh", scale=0)

                gr.Markdown("---\n#### Server Actions")
                mcp_server_dd = gr.Dropdown(
                    label="Select Server", choices=[], interactive=True
                )
                with gr.Row():
                    mcp_connect_btn = gr.Button("Connect", variant="primary", scale=0)
                    mcp_disconnect_btn = gr.Button("Disconnect", scale=0)
                    mcp_enable_btn = gr.Button("Enable", scale=0)
                    mcp_disable_btn = gr.Button("Disable", scale=0)
                    mcp_autostart_btn = gr.Button("Toggle Autostart", scale=0)
                    mcp_remove_btn = gr.Button(
                        "Remove", elem_classes=["stop-btn"], scale=0
                    )
                    gr.Column(scale=1)
                mcp_action_md = gr.Markdown("")

                gr.Markdown("---\n#### Add Server")
                with gr.Row():
                    mcp_name = gr.Textbox(
                        label="Name", placeholder="my-server", scale=2
                    )
                    mcp_transport = gr.Dropdown(
                        label="Transport",
                        choices=["stdio", "sse", "http"],
                        value="stdio",
                        scale=1,
                        interactive=True,
                    )
                mcp_command = gr.Textbox(
                    label="Command executable (stdio) or URL (sse/http)",
                    placeholder="npx",
                )
                mcp_args = gr.Textbox(
                    label="Args (JSON array, stdio only)",
                    placeholder='["-y", "@modelcontextprotocol/server-filesystem", "C:/Users/me/documents"]',
                )
                with gr.Row():
                    mcp_add_btn = gr.Button("Add Server", variant="primary", scale=0)
                    gr.Column(scale=1)

                gr.Markdown("---\n#### Import Config (Claude / Cursor format)")
                mcp_import_text = gr.Textbox(
                    label="Paste JSON config",
                    lines=4,
                    placeholder='{\n  "mcpServers": {\n    "filesystem": {\n      "command": "npx",\n      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/docs"]\n    }\n  }\n}',
                )
                with gr.Row():
                    mcp_import_btn = gr.Button("Import", variant="primary", scale=0)
                    gr.Column(scale=1)

                gr.Markdown("---\n#### Available Tools")
                mcp_tools_md = gr.Markdown("No tools available.")

                gr.Markdown("---\n#### Test Tool Call")
                mcp_tool_dd = gr.Dropdown(label="Tool", choices=[], interactive=True)
                mcp_tool_args = gr.Textbox(
                    label="Arguments (JSON)",
                    lines=2,
                    placeholder='{"path": "C:/Users/me/documents"}',
                )
                with gr.Row():
                    mcp_test_btn = gr.Button("Call Tool", variant="primary", scale=0)
                    gr.Column(scale=1)
                mcp_test_result = gr.Textbox(label="Result", lines=6, interactive=False)

            with gr.Tab("Privacy Shield", id="shield-tab"):
                gr.Markdown(
                    "### Privacy Shield\n\n"
                    "Scrub PII and mask writing style before pasting text into external services.\n\n"
                    "**Step 1:** Paste text and click **Scrub PII**\n"
                    "**Step 2:** Click **Restyle** to flatten writing style\n"
                    "**Step 3:** Copy the final output"
                )
                gr.Markdown("---\n#### Step 1: Paste and Scrub PII")
                shield_input = gr.Textbox(
                    label="Raw Prompt",
                    placeholder="Paste here...",
                    lines=8,
                    interactive=True,
                )
                with gr.Row():
                    scrub_btn = gr.Button("Scrub PII", variant="primary", scale=0)
                    gr.Column(scale=1)
                scrub_report = gr.Markdown("")
                scrubbed_out = gr.Textbox(
                    label="After PII Scrubbing (editable)", lines=6, interactive=True
                )

                gr.Markdown("---\n#### Step 2: Restyle and Anonymize")
                with gr.Row():
                    restyle_btn = gr.Button("Restyle", variant="primary", scale=0)
                    gr.Column(scale=1)
                final_out = gr.Textbox(
                    label="Final Safe-to-Paste Output", lines=8, interactive=True
                )

        refresh_btn.click(refresh_status, outputs=[status_md])
        load_btn.click(
            start_server, inputs=[gguf_dd, custom_path], outputs=[action_md]
        ).then(
            refresh_status,
            outputs=[status_md],
        )
        stop_btn.click(stop_server, outputs=[action_md]).then(
            refresh_status, outputs=[status_md]
        )

        scrub_btn.click(
            scrub_pii, inputs=[shield_input], outputs=[scrubbed_out, scrub_report]
        )
        restyle_btn.click(restyle_text, inputs=[scrubbed_out], outputs=[final_out])

        mcp_outputs = [mcp_status_md, mcp_server_dd, mcp_tools_md, mcp_tool_dd]
        mcp_refresh_btn.click(mcp_refresh, outputs=mcp_outputs)

        for button, handler in [
            (mcp_connect_btn, mcp_do_connect),
            (mcp_disconnect_btn, mcp_do_disconnect),
            (mcp_enable_btn, mcp_do_enable),
            (mcp_disable_btn, mcp_do_disable),
            (mcp_autostart_btn, mcp_do_toggle_autostart),
            (mcp_remove_btn, mcp_do_remove),
        ]:
            button.click(handler, inputs=[mcp_server_dd], outputs=[mcp_action_md]).then(
                mcp_refresh,
                outputs=mcp_outputs,
            )

        mcp_add_btn.click(
            mcp_do_add,
            inputs=[mcp_name, mcp_transport, mcp_command, mcp_args],
            outputs=[mcp_action_md],
        ).then(mcp_refresh, outputs=mcp_outputs)

        mcp_import_btn.click(
            mcp_do_import, inputs=[mcp_import_text], outputs=[mcp_action_md]
        ).then(
            mcp_refresh,
            outputs=mcp_outputs,
        )

        mcp_test_btn.click(
            mcp_do_test, inputs=[mcp_tool_dd, mcp_tool_args], outputs=[mcp_test_result]
        )

    return app, css


if __name__ == "__main__":
    print()
    if not runtime.server_dir:
        print("ERROR: No llama-server directory configured.")
        print("Set LLAMA_SERVER_DIR in .env")
        sys.exit(1)

    print(f"Dir     : {runtime.server_dir}")
    print(f"Server  : {API}")
    print("Think   : toggle in Chat tab")
    print(f"MCP cfg : {MCP_CONFIG_PATH}")
    if REMOTE_TOOLS_WARNING:
        print(f"Warning : {REMOTE_TOOLS_WARNING}")

    configured_servers = len(mcp.servers)
    if configured_servers:
        print(f"MCP     : {configured_servers} server(s) configured")
    print()

    if TOOLS_ENABLED:
        mcp.autostart()

    app, css = create_ui()
    app.launch(
        server_name=UI_HOST,
        server_port=UI_PORT,
        inbrowser=True,
        share=UI_SHARE,
        show_error=True,
        css=css,
    )
