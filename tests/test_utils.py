"""Tests for utility functions."""
import pytest
from server import (
    convert_messages_to_chatjimmy,
    map_openai_params_to_chatjimmy,
    build_usage_from_stats,
    call_chatjimmy,
)
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class TestConvertMessagesToChatJimmy:
    """Tests for message conversion."""

    def test_basic_conversion(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        chat_messages, system_prompt = convert_messages_to_chatjimmy(messages)

        assert system_prompt == "System prompt"
        assert len(chat_messages) == 2
        assert chat_messages[0] == {"role": "user", "content": "Hello"}
        assert chat_messages[1] == {"role": "assistant", "content": "Hi there"}

    def test_multiple_system_messages(self):
        messages = [
            {"role": "system", "content": "First"},
            {"role": "system", "content": "Second"},
            {"role": "user", "content": "Hello"},
        ]
        chat_messages, system_prompt = convert_messages_to_chatjimmy(messages)

        assert system_prompt == "First\nSecond"
        assert len(chat_messages) == 1

    def test_tool_message_conversion(self):
        messages = [
            {"role": "user", "content": "Call tool"},
            {"role": "tool", "content": "Tool result", "name": "get_weather"},
        ]
        chat_messages, system_prompt = convert_messages_to_chatjimmy(messages)

        assert system_prompt == ""
        assert len(chat_messages) == 2
        assert chat_messages[1]["role"] == "user"
        assert "Tool result (get_weather): Tool result" in chat_messages[1]["content"]

    def test_function_message_conversion(self):
        messages = [
            {"role": "function", "content": "Function result", "name": "my_func"},
        ]
        chat_messages, system_prompt = convert_messages_to_chatjimmy(messages)

        assert chat_messages[0]["role"] == "user"
        assert "Function result (my_func): Function result" in chat_messages[0]["content"]

    def test_multimodal_content_extraction(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                    {"type": "text", "text": "World"},
                ]
            }
        ]
        chat_messages, system_prompt = convert_messages_to_chatjimmy(messages)

        assert chat_messages[0]["content"] == "Hello\nWorld"

    def test_empty_content_handling(self):
        messages = [
            {"role": "user", "content": None},
            {"role": "assistant", "content": ""},
        ]
        chat_messages, system_prompt = convert_messages_to_chatjimmy(messages)

        assert chat_messages[0]["content"] == ""
        assert chat_messages[1]["content"] == ""


class TestMapOpenAIParamsToChatJimmy:
    """Tests for parameter mapping."""

    def test_default_mapping(self):
        class MockRequest(BaseModel):
            model: str = "llama3.1-8B"
            temperature: Optional[float] = 1.0
            top_p: Optional[float] = 1.0
            response_format: Optional[Dict] = None
            tools: Optional[List] = None

        request = MockRequest()
        options = map_openai_params_to_chatjimmy(request)

        assert options["selectedModel"] == "llama3.1-8B"
        assert options["topK"] == 9  # 1.0 * 8 + 1 = 9
        assert options["systemPrompt"] == ""

    def test_temperature_mapping(self):
        class MockRequest(BaseModel):
            model: str = "llama3.1-8B"
            temperature: Optional[float] = 0.0
            top_p: Optional[float] = 1.0
            response_format: Optional[Dict] = None
            tools: Optional[List] = None

        request = MockRequest()
        options = map_openai_params_to_chatjimmy(request)
        assert options["topK"] == 1  # 0.0 * 8 + 1 = 1

        request.temperature = 2.0
        options = map_openai_params_to_chatjimmy(request)
        assert options["topK"] == 17  # 2.0 * 8 + 1 = 17

    def test_json_mode_injection(self):
        class MockRequest(BaseModel):
            model: str = "llama3.1-8B"
            temperature: Optional[float] = 1.0
            top_p: Optional[float] = 1.0
            response_format: Optional[Dict] = {"type": "json_object"}
            tools: Optional[List] = None

        request = MockRequest()
        options = map_openai_params_to_chatjimmy(request)

        assert "respond with valid JSON only" in options["systemPrompt"]

    def test_tools_injection(self):
        class MockRequest(BaseModel):
            model: str = "llama3.1-8B"
            temperature: Optional[float] = 1.0
            top_p: Optional[float] = 1.0
            response_format: Optional[Dict] = None
            tools: Optional[List] = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ]

        request = MockRequest()
        options = map_openai_params_to_chatjimmy(request)

        assert "Available functions:" in options["systemPrompt"]
        assert "get_weather" in options["systemPrompt"]
        assert "function_name" in options["systemPrompt"]


class TestBuildUsageFromStats:
    """Tests for usage object building."""

    def test_build_from_full_stats(self):
        stats = {
            "prefill_tokens": 10,
            "decode_tokens": 20,
            "total_tokens": 30,
        }
        usage = build_usage_from_stats(stats)

        assert usage is not None
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_build_from_partial_stats(self):
        stats = {"prefill_tokens": 5}
        usage = build_usage_from_stats(stats)

        assert usage.prompt_tokens == 5
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_build_from_none(self):
        usage = build_usage_from_stats(None)
        assert usage is None

    def test_build_from_empty_dict(self):
        usage = build_usage_from_stats({})
        assert usage is not None
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0