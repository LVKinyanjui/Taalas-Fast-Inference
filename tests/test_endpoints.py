"""Integration tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json


class TestModelsEndpoint:
    """Tests for /v1/models endpoints."""

    def test_list_models(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) >= 1
        assert data["data"][0]["id"] == "llama3.1-8B"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "chatjimmy"

    def test_get_model_exists(self, client):
        response = client.get("/v1/models/llama3.1-8B")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == "llama3.1-8B"
        assert data["object"] == "model"

    def test_get_model_not_found(self, client):
        response = client.get("/v1/models/nonexistent")
        assert response.status_code == 404

        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["code"] == "model_not_found"


class TestChatCompletionsEndpoint:
    """Tests for /v1/chat/completions endpoint."""

    def test_non_streaming_basic(self, client, mock_upstream_call):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "llama3.1-8B"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] == 18
        assert data["usage"]["completion_tokens"] == 26
        assert data["usage"]["total_tokens"] == 44

    def test_streaming_basic(self, client, mock_upstream_call):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE stream
        content = response.text
        lines = content.strip().split("\n\n")
        assert len(lines) >= 3  # First chunk, final chunk, [DONE]

        # First chunk has content
        first_chunk = json.loads(lines[0].replace("data: ", ""))
        assert first_chunk["object"] == "chat.completion.chunk"
        assert first_chunk["choices"][0]["delta"]["role"] == "assistant"
        assert first_chunk["choices"][0]["finish_reason"] is None

        # Final chunk has finish_reason and usage
        final_chunk = json.loads(lines[1].replace("data: ", ""))
        assert final_chunk["choices"][0]["finish_reason"] == "stop"
        assert "usage" in final_chunk

        # Last line is [DONE]
        assert lines[2].strip() == "data: [DONE]"

    def test_missing_messages_error(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1-8B"},
        )
        assert response.status_code == 400

        data = response.json()
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["param"] == "messages"

    def test_invalid_model_error(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 404

        data = response.json()
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["code"] == "model_not_found"
        assert data["error"]["param"] == "model"

    def test_n_greater_than_one_error(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [{"role": "user", "content": "Hello"}],
                "n": 2,
            },
        )
        assert response.status_code == 400

        data = response.json()
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["param"] == "n"

    def test_logit_bias_error(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [{"role": "user", "content": "Hello"}],
                "logit_bias": {"123": 0.5},
            },
        )
        assert response.status_code == 400

        data = response.json()
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["param"] == "logit_bias"

    def test_system_message_handling(self, client, mock_upstream_call):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [
                    {"role": "system", "content": "You are a math expert."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
            },
        )
        assert response.status_code == 200

        # Verify upstream was called with system prompt
        call_args = mock_upstream_call.call_args
        messages, options = call_args[0]
        assert "You are a math expert." in options["systemPrompt"]

    def test_temperature_mapping(self, client, mock_upstream_call):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.5,
            },
        )
        assert response.status_code == 200

        call_args = mock_upstream_call.call_args
        messages, options = call_args[0]
        # temp 0.5 -> topK = min(8, max(1, int(0.5 * 4 + 1))) = 3
        assert options["topK"] <= 8
        assert options["topK"] >= 1

    def test_json_mode(self, client, mock_upstream_call):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [{"role": "user", "content": "Return JSON"}],
                "response_format": {"type": "json_object"},
            },
        )
        assert response.status_code == 200

        call_args = mock_upstream_call.call_args
        messages, options = call_args[0]
        assert "respond with valid JSON only" in options["systemPrompt"]

    def test_tools_injection(self, client, mock_upstream_call, sample_tools):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [{"role": "user", "content": "Get weather"}],
                "tools": sample_tools,
            },
        )
        assert response.status_code == 200

        call_args = mock_upstream_call.call_args
        messages, options = call_args[0]
        assert "Available functions:" in options["systemPrompt"]
        assert "get_weather" in options["systemPrompt"]

    def test_tool_call_parsing(self, client):
        """Test that tool calls in response are parsed correctly."""
        with patch("server.call_chatjimmy", return_value=(
            '{"name": "get_weather", "arguments": {"location": "Tokyo"}}',
            {"prefill_tokens": 10, "decode_tokens": 20, "total_tokens": 30}
        )):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama3.1-8B",
                    "messages": [{"role": "user", "content": "Get weather"}],
                    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {}}}],
                },
            )
        assert response.status_code == 200

        data = response.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        assert data["choices"][0]["message"]["tool_calls"] is not None
        assert len(data["choices"][0]["message"]["tool_calls"]) == 1
        assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert data["choices"][0]["message"]["content"] is None

    def test_max_tokens_parameter(self, client, mock_upstream_call):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            },
        )
        assert response.status_code == 200

    def test_max_completion_tokens_parameter(self, client, mock_upstream_call):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_completion_tokens": 50,
            },
        )
        assert response.status_code == 200


class TestCompletionsEndpoint:
    """Tests for legacy /v1/completions endpoint."""

    def test_basic_completion(self, client, mock_upstream_call):
        response = client.post(
            "/v1/completions",
            json={
                "model": "llama3.1-8B",
                "prompt": "The capital of France is",
                "max_tokens": 50,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "chat.completion"  # Reuses chat completion format


class TestHealthEndpoint:
    """Tests for health check."""

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "upstream" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "ChatJimmy OpenAI-Compatible API"
        assert "endpoints" in data