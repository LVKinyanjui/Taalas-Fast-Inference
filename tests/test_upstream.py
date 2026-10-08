"""Tests for upstream ChatJimmy integration."""
import pytest
from unittest.mock import patch, MagicMock
import requests
import json

from server import call_chatjimmy, CHATJIMMY_URL


class TestCallChatJimmy:
    """Tests for the call_chatjimmy function."""

    def test_successful_call_with_stats(self):
        """Test successful upstream call with stats block."""
        mock_response = MagicMock()
        mock_response.text = (
            "The answer is 42."
            "<|stats|>"
            '{"prefill_tokens": 10, "decode_tokens": 5, "total_tokens": 15, "ttft": 0.01}'
            "<|/stats|>"
        )
        mock_response.raise_for_status = MagicMock()

        with patch("server.requests.post", return_value=mock_response) as mock_post:
            text, stats = call_chatjimmy(
                [{"role": "user", "content": "What is 6*7?"}],
                {"selectedModel": "llama3.1-8B", "systemPrompt": "", "topK": 8}
            )

        assert text == "The answer is 42."
        assert stats is not None
        assert stats["prefill_tokens"] == 10
        assert stats["decode_tokens"] == 5
        assert stats["total_tokens"] == 15

        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["chatOptions"]["selectedModel"] == "llama3.1-8B"
        assert call_args[1]["json"]["chatOptions"]["topK"] == 8
        assert call_args[1]["timeout"] == 300

    def test_successful_call_without_stats(self):
        """Test upstream response without stats block."""
        mock_response = MagicMock()
        mock_response.text = "Simple response without stats."
        mock_response.raise_for_status = MagicMock()

        with patch("server.requests.post", return_value=mock_response):
            text, stats = call_chatjimmy(
                [{"role": "user", "content": "Hello"}],
                {"selectedModel": "llama3.1-8B", "systemPrompt": "", "topK": 8}
            )

        assert text == "Simple response without stats."
        assert stats is None

    def test_malformed_stats_block(self):
        """Test handling of malformed stats JSON."""
        mock_response = MagicMock()
        mock_response.text = (
            "Response text"
            "<|stats|>"
            "invalid json {"
            "<|/stats|>"
        )
        mock_response.raise_for_status = MagicMock()

        with patch("server.requests.post", return_value=mock_response):
            text, stats = call_chatjimmy(
                [{"role": "user", "content": "Hello"}],
                {"selectedModel": "llama3.1-8B", "systemPrompt": "", "topK": 8}
            )

        assert text == "Response text"
        assert stats is None  # Should gracefully handle JSON decode error

    def test_missing_stats_end_marker(self):
        """Test handling when stats end marker is missing."""
        mock_response = MagicMock()
        mock_response.text = "Response <|stats|> {\"tokens\": 10}"
        mock_response.raise_for_status = MagicMock()

        with patch("server.requests.post", return_value=mock_response):
            text, stats = call_chatjimmy(
                [{"role": "user", "content": "Hello"}],
                {"selectedModel": "llama3.1-8B", "systemPrompt": "", "topK": 8}
            )

        assert text == "Response "
        assert stats is None

    def test_http_error_raises(self):
        """Test that HTTP errors are raised."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("server.requests.post", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                call_chatjimmy(
                    [{"role": "user", "content": "Hello"}],
                    {"selectedModel": "llama3.1-8B", "systemPrompt": "", "topK": 8}
                )

    def test_timeout_raises(self):
        """Test that timeout errors are raised."""
        with patch("server.requests.post", side_effect=requests.Timeout("Connection timed out")):
            with pytest.raises(requests.Timeout):
                call_chatjimmy(
                    [{"role": "user", "content": "Hello"}],
                    {"selectedModel": "llama3.1-8B", "systemPrompt": "", "topK": 8}
                )

    def test_request_headers(self):
        """Test that correct headers are sent."""
        mock_response = MagicMock()
        mock_response.text = "OK"
        mock_response.raise_for_status = MagicMock()

        with patch("server.requests.post", return_value=mock_response) as mock_post:
            call_chatjimmy(
                [{"role": "user", "content": "Hello"}],
                {"selectedModel": "llama3.1-8B", "systemPrompt": "System prompt", "topK": 8}
            )

        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert headers["User-Agent"] == "Mozilla/5.0"
        assert headers["Content-Type"] == "application/json"
        assert headers["Origin"] == "https://chatjimmy.ai"
        assert headers["Referer"] == "https://chatjimmy.ai/"

    def test_system_prompt_merging(self):
        """Test that system prompts from messages and options are merged."""
        mock_response = MagicMock()
        mock_response.text = "OK"
        mock_response.raise_for_status = MagicMock()

        with patch("server.requests.post", return_value=mock_response) as mock_post:
            call_chatjimmy(
                [
                    {"role": "system", "content": "From messages"},
                    {"role": "user", "content": "Hello"}
                ],
                {"selectedModel": "llama3.1-8B", "systemPrompt": "From options", "topK": 8}
            )

        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        system_prompt = payload["chatOptions"]["systemPrompt"]
        assert "From options" in system_prompt
        assert "From messages" in system_prompt
        # Options prompt should come first
        assert system_prompt.index("From options") < system_prompt.index("From messages")

    def test_attachment_is_null(self):
        """Test that attachment is always null."""
        mock_response = MagicMock()
        mock_response.text = "OK"
        mock_response.raise_for_status = MagicMock()

        with patch("server.requests.post", return_value=mock_response) as mock_post:
            call_chatjimmy(
                [{"role": "user", "content": "Hello"}],
                {"selectedModel": "llama3.1-8B", "systemPrompt": "", "topK": 8}
            )

        call_args = mock_post.call_args
        assert call_args[1]["json"]["attachment"] is None