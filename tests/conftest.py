"""Pytest configuration and shared fixtures."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from server import app, call_chatjimmy, convert_messages_to_chatjimmy, map_openai_params_to_chatjimmy, build_usage_from_stats


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_chatjimmy_response():
    """Mock successful ChatJimmy response with stats."""
    return (
        "The square of 50 is 2500.",
        {
            "created_at": 1790930715.7659063,
            "done": True,
            "done_reason": "stop",
            "total_duration": 1.9955310821533203,
            "logprobs": None,
            "topk": 8,
            "ttft": 0.0010437965393066406,
            "reason": "termination token 128009/<|eot_id|> detected",
            "status": 0,
            "prefill_tokens": 18,
            "prefill_rate": 17244.740063956146,
            "decode_tokens": 26,
            "decode_rate": 14893.7317672767,
            "total_tokens": 44,
            "total_time": 0.0028002262115478516,
            "roundtrip_time": 2131,
        }
    )


@pytest.fixture
def mock_chatjimmy_response_no_stats():
    """Mock ChatJimmy response without stats block."""
    return ("Simple response without stats.", None)


@pytest.fixture
def sample_messages():
    """Sample OpenAI-style messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What is 2+2?"},
    ]


@pytest.fixture
def sample_tools():
    """Sample tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]


# Mock the upstream call for integration tests
@pytest.fixture(autouse=True)
def mock_upstream_call(mock_chatjimmy_response):
    """Automatically mock upstream calls for all tests."""
    with patch("server.call_chatjimmy", return_value=mock_chatjimmy_response) as mock:
        yield mock