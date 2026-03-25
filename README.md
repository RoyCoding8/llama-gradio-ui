# llama-gradio-ui

A local chat UI for `llama.cpp` server with Gradio 6, MCP tool calling, and optional privacy processing.

## What You Can Do

- Start and stop `llama-server` from the UI and load models from a GGUF folder or custom path
- Stream chat responses from the OpenAI-compatible `/v1/chat/completions` endpoint
- Connect MCP servers over stdio, SSE, or HTTP and let the model call tools
- Run a two-step privacy flow: redact PII with Presidio, then restyle text locally

## Requirements

- Python 3.10+
- `llama-server` / `llama-server.exe` from [llama.cpp](https://github.com/ggerganov/llama.cpp)
- At least one GGUF model file
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

This repository is a `uv` project (`pyproject.toml` + `uv.lock`).

## Quick Start

1. Clone and enter the project

   ```bash
   git clone https://github.com/RoyCoding8/llama-gradio-ui.git
   cd llama-gradio-ui
   ```

2. Create `.env` from the example

   ```bash
   cp .env.example .env
   ```

3. Update `.env` with your local paths and defaults

   ```env
   LLAMA_SERVER_DIR=C:\path\to\llama-cpp-build
   GGUF_DIR=C:\path\to\models

   GPU_LAYERS=-1
   CTX_SIZE=4096
   KV_CACHE_TYPE_K=f16
   KV_CACHE_TYPE_V=f16
   ```

4. Install dependencies

   ```bash
   uv sync
   ```

   Or with pip:

   ```bash
   pip install -e .
   python -m spacy download en_core_web_lg
   ```

5. Run the app

   ```bash
   uv run python app.py
   ```

   On Windows you can also use `start.bat`.

By default, the UI is available at `http://127.0.0.1:7860`.

## Configuration

Key values in `.env`:

- `LLAMA_HOST`, `LLAMA_PORT`: target `llama-server` host and port
- `LLAMA_SERVER_DIR`: directory that contains `llama-server`
- `GGUF_DIR`: directory scanned for `.gguf` models
- `UI_HOST`, `UI_PORT`, `UI_SHARE`: Gradio host, port, and public share mode
- `CTX_SIZE`, `GPU_LAYERS`: default runtime settings for `llama-server`
- `KV_CACHE_TYPE_K`, `KV_CACHE_TYPE_V`: KV cache quantization settings
- `ALLOW_REMOTE_TOOLS`: when `UI_SHARE=1`, set this to `1` only if you explicitly want remote tool execution

## MCP Server Setup

You can add MCP servers in the UI or edit `mcp_servers.json` directly.

```json
{
  "servers": {
    "filesystem": {
      "name": "filesystem",
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/docs"],
      "enabled": true,
      "autostart": false
    }
  }
}
```

The MCP tab also accepts Claude/Cursor-format imports.

## Tool-Calling Flow

1. The model receives chat history plus OpenAI-format tool schemas from connected MCP servers
2. If it emits tool calls, the app dispatches them to MCP and records results
3. Tool results are appended to conversation context
4. The model runs again, up to 5 rounds
5. Final output streams back to the chat UI

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Application entry point and Gradio wiring |
| `server_runtime.py` | `llama-server` process lifecycle and model discovery |
| `chat_engine.py` | Streaming chat and MCP tool-call loop |
| `mcp_manager.py` | Async MCP client manager and server connections |
| `mcp_facade.py` | UI-facing MCP actions and response formatting |
| `privacy_shield.py` | PII redaction and local restyling flow |
| `config.py` | Environment and `.env` parsing |
| `style.css` | UI styling |

## License

Apache 2.0
