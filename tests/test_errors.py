"""Tests for error handling and authentication."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os


class TestAuthentication:
    """Tests for API key authentication."""

    def test_no_auth_required_by_default(self, client):
        """Auth is disabled by default (no CHATJIMMY_API_KEY)."""
        response = client.get("/v1/models")
        assert response.status_code == 200

    def test_auth_required_when_key_set(self):
        """Auth is enforced when CHATJIMMY_API_KEY is set."""
        # Create a new app instance with auth enabled
        import server
        original_key = os.environ.get("CHATJIMMY_API_KEY")
        os.environ["CHATJIMMY_API_KEY"] = "test-secret-key"

        try:
            # Need to reload the module to pick up the new env var
            import importlib
            importlib.reload(server)
            from fastapi.testclient import TestClient
            auth_client = TestClient(server.app)

            # Without auth header
            response = auth_client.get("/v1/models")
            assert response.status_code == 401
            assert response.json()["error"]["type"] == "authentication_error"

            # With wrong key
            response = auth_client.get(
                "/v1/models",
                headers={"Authorization": "Bearer wrong-key"}
            )
            assert response.status_code == 401

            # With correct key
            response = auth_client.get(
                "/v1/models",
                headers={"Authorization": "Bearer test-secret-key"}
            )
            assert response.status_code == 200

            # With malformed header
            response = auth_client.get(
                "/v1/models",
                headers={"Authorization": "Basic test-secret-key"}
            )
            assert response.status_code == 401

        finally:
            if original_key:
                os.environ["CHATJIMMY_API_KEY"] = original_key
            else:
                os.environ.pop("CHATJIMMY_API_KEY", None)
            importlib.reload(server)


class TestErrorHandling:
    """Tests for error responses."""

    def test_upstream_timeout_error(self, client):
        """Test 504 on upstream timeout."""
        import requests
        with patch("server.call_chatjimmy", side_effect=requests.Timeout("Timeout")):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "llama3.1-8B", "messages": [{"role": "user", "content": "Hello"}]},
            )
        assert response.status_code == 504
        data = response.json()
        assert data["error"]["type"] == "server_error"
        assert "timed out" in data["error"]["message"].lower()

    def test_upstream_http_error(self, client):
        """Test 502 on upstream HTTP error."""
        import requests
        mock_response = MagicMock()
        mock_response.status_code = 500
        http_error = requests.HTTPError("Server Error", response=mock_response)

        with patch("server.call_chatjimmy", side_effect=http_error):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "llama3.1-8B", "messages": [{"role": "user", "content": "Hello"}]},
            )
        assert response.status_code == 502
        data = response.json()
        assert data["error"]["type"] == "server_error"

    def test_upstream_generic_error(self, client):
        """Test 500 on generic upstream error."""
        with patch("server.call_chatjimmy", side_effect=Exception("Unexpected error")):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "llama3.1-8B", "messages": [{"role": "user", "content": "Hello"}]},
            )
        assert response.status_code == 500
        data = response.json()
        assert data["error"]["type"] == "server_error"
        assert "Internal error" in data["error"]["message"]

    def test_error_response_format(self, client):
        """Verify OpenAI-compatible error format."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1-8B"},  # Missing messages
        )
        assert response.status_code == 400

        data = response.json()
        assert "error" in data
        error = data["error"]
        assert "message" in error
        assert "type" in error
        assert error["type"] == "invalid_request_error"
        assert "param" in error  # Should include param for validation errors


class TestResponseFormat:
    """Tests for response format compliance."""

    def test_chat_completion_response_structure(self, client, mock_upstream_call):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1-8B", "messages": [{"role": "user", "content": "Hello"}]},
        )
        data = response.json()

        # Required fields per OpenAI spec
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert isinstance(data["created"], int)
        assert data["model"] == "llama3.1-8B"
        assert "choices" in data
        assert isinstance(data["choices"], list)
        assert len(data["choices"]) == 1

        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
        assert choice["finish_reason"] in ["stop", "length", "tool_calls", "content_filter"]

        # Usage object
        assert "usage" in data
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_streaming_chunk_structure(self, client, mock_upstream_call):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1-8B",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        content = response.text
        lines = [l for l in content.strip().split("\n\n") if l.strip()]

        # First chunk
        first = json.loads(lines[0].replace("data: ", ""))
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"]["role"] == "assistant"
        assert first["choices"][0]["finish_reason"] is None

        # Final chunk
        final = json.loads(lines[1].replace("data: ", ""))
        assert final["choices"][0]["finish_reason"] == "stop"
        assert "usage" in final

        # DONE marker
        assert lines[2].strip() == "data: [DONE]"

    def test_model_info_structure(self, client):
        response = client.get("/v1/models/llama3.1-8B")
        data = response.json()

        assert data["id"] == "llama3.1-8B"
        assert data["object"] == "model"
        assert "created" in data
        assert "owned_by" in data