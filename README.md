# ChatJimmy OpenAI-Compatible API Adapter

An OpenAI-compatible HTTP API (`/v1/`) that proxies requests to [chatjimmy.ai](https://chatjimmy.ai) running `llama3.1-8B`.

## Features

- **OpenAI-compatible endpoints**: `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`
- **Streaming & non-streaming** responses (SSE format)
- **Parameter mapping**: `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `response_format`, `tools`
- **Function/tool calling** via system prompt injection + response parsing
- **JSON mode** (`response_format: { "type": "json_object" }`)
- **Usage tracking** from upstream stats (`prompt_tokens`, `completion_tokens`, `total_tokens`)
- **Optional API key authentication** (set `CHATJIMMY_API_KEY` env var)
- **OpenAI-style error responses**

## Quick Start

```bash
# Install dependencies
pip install fastapi pydantic requests uvicorn

# Run server
uvicorn server:app --host 0.0.0.0 --port 8000

# API base URL
http://localhost:8000/v1/
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `CHATJIMMY_API_KEY` | Enable Bearer token auth | None (auth disabled) |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List available models |
| GET | `/v1/models/{id}` | Get model info |
| POST | `/v1/chat/completions` | Chat completions (OpenAI format) |
| POST | `/v1/completions` | Legacy completions |
| GET | `/health` | Health check |

## Example Requests

**Non-streaming:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1-8B", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Streaming:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1-8B", "stream": true, "messages": [{"role": "user", "content": "Hello!"}]}'
```

**With tools (function calling):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1-8B",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
      }
    }]
  }'
```

## Supported Parameters

| Parameter | Supported | Notes |
|-----------|-----------|-------|
| `model` | ✅ | Must be `llama3.1-8B` |
| `messages` | ✅ | system, user, assistant, tool, function |
| `temperature` | ✅ | Mapped to upstream `topK` |
| `top_p` | ⚠️ | Logged, not directly supported upstream |
| `max_tokens` / `max_completion_tokens` | ⚠️ | Post-generation truncation only |
| `stream` | ✅ | Buffered upstream → SSE output |
| `stop` | ❌ | Not supported upstream |
| `response_format` | ✅ | `json_object` injects JSON instruction |
| `tools` / `tool_choice` | ✅ | Injected into system prompt |
| `seed` | ❌ | Not supported upstream |
| `n` > 1 | ❌ | Returns error |

## Architecture

```
Client → FastAPI Adapter → chatjimmy.ai/api/chat
              │
              ├─ OpenAI request parsing
              ├─ Parameter mapping
              ├─ Message conversion
              ├─ Upstream call (buffered)
              ├─ Stats extraction (<|stats|> block)
              ├─ Usage calculation
              └─ OpenAI response formatting
```

## Known Limitations

1. **No true streaming** — Upstream delivers full response at once (~4s latency). Adapter emits SSE-compatible chunks but not token-by-token.
2. **Single model** — Only `llama3.1-8B` available upstream.
3. **Limited sampling params** — Upstream only exposes `topK` (mapped from `temperature`).
4. **No stop sequences** — Not supported by upstream.
5. **Tool calling is simulated** — Parsed from model's JSON output, not native function calling.

## License

MIT