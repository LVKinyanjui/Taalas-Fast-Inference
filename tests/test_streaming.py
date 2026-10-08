"""Tests for streaming behavior and edge cases."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import json


class TestStreamingEdgeCases:
    """Tests for streaming-specific behavior."""

    def test_streaming_sse_format(self, client, mock_upstream_call):
        """Verify SSE format compliance."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # Check headers
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["connection"] == "keep-alive"

        # Parse all chunks
        chunks = []
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    chunks.append(json.loads(data))

        assert len(chunks) == 2  # First chunk + final chunk

        # First chunk
        assert chunks[0]["object"] == "chat.completion.chunk"
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert chunks[0]["choices"][0]["finish_reason"] is None

        # Final chunk
        assert chunks[1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[1]["choices"][0]["delta"] == {}
        assert "usage" in chunks[1]

    def test_streaming_same_id_across_chunks(self, client, mock_upstream_call):
        """Verify completion ID is consistent across stream chunks."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        ids = []
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() != "[DONE]":
                        chunk = json.loads(data)
                        ids.append(chunk["id"])

        assert len(ids) == 2
        assert ids[0] == ids[1]  # Same ID
        assert ids[0].startswith("chatcmpl-")

    def test_streaming_created_timestamp_consistent(self, client, mock_upstream_call):
        """Verify created timestamp is consistent across chunks."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        timestamps = []
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() != "[DONE]":
                        chunk = json.loads(data)
                        timestamps.append(chunk["created"])

        assert len(timestamps) == 2
        assert timestamps[0] == timestamps[1]

    def test_non_streaming_no_sse(self, client, mock_upstream_call):
        """Verify non-streaming returns JSON, not SSE."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "stream": False,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert data["object"] == "chat.completion"  # Not chat.completion.chunk


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_user_message(self, client, mock_upstream_call):
        """Test handling of empty user message."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [{"role": "user", "content": ""}],
            },
        )
        assert response.status_code == 200

    def test_very_long_conversation(self, client, mock_upstream_call):
        """Test handling of many messages."""
        messages = []
        for i in range(50):
            messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"})

        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1-8B", "messages": messages},
        )
        assert response.status_code == 200

    def test_unicode_content(self, client, mock_upstream_call):
        """Test handling of unicode in messages."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [{"role": "user", "content": "Hello 世界 🌍"}],
            },
        )
        assert response.status_code == 200

    def test_special_characters_in_content(self, client, mock_upstream_call):
        """Test handling of special characters."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [{"role": "user", "content": "Test\n\t\r\"'\\"}],
            },
        )
        assert response.status_code == 200

    def test_tool_message_without_name(self, client, mock_upstream_call):
        """Test tool message without name field."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [
                    {"role": "user", "content": "Call tool"},
                    {"role": "tool", "content": "Result"},  # No name
                ],
            },
        )
        assert response.status_code == 200

    def test_assistant_message_with_tool_calls(self, client, mock_upstream_call):
        """Test assistant message with tool_calls (from previous turn)."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "messages": [
                    {"role": "user", "content": "Get weather"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}
                        }]
                    },
                    {"role": "tool", "tool_call_id": "call_123", "content": "Sunny"},
                ],
            },
        )
        assert response.status_code == 200


class TestParameterValidation:
    """Tests for request parameter validation."""

    def test_temperature_bounds(self, client):
        """Test temperature validation bounds."""
        # Valid: 0.0
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1-8B", "messages": [{"role": "user", "content": "Hi"}], "temperature": 0.0},
        )
        assert response.status_code == 200

        # Valid: 2.0
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1-8B", "messages": [{"role": "user", "content": "Hi"}], "temperature": 2.0},
        )
        assert response.status_code == 200

        # Invalid: -0.1
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1-8B", "messages": [{"role": "user", "content": "Hi"}], "temperature": -0.1},
        )
        assert response.status_code == 422  # Pydantic validation error

        # Invalid: 2.1
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1-8B", "messages": [{"role": "user", "content": "Hi"}], "temperature": 2.1},
        )
        assert response.status_code == 422

    def test_top_p_bounds(self, client):
        """Test top_p validation bounds."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1-8B", "messages": [{"role": "user", "content": "Hi"}], "top_p": 1.5},
        )
        assert response.status_code == 422

    def test_max_tokens_positive(self, client):
        """Test max_tokens must be positive."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1-8B", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 0},
        )
        assert response.status_code == 422

    def test_n_positive(self, client):
        """Test n must be positive."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1-8B", "messages": [{"role": "user", "content": "Hi"}], "n": 0},
        )
        assert response.status_code == 422