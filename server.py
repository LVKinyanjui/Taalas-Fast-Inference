from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Union, Literal
import requests
import asyncio
import json
import time
import uuid
import os
from contextlib import asynccontextmanager

# =============================================================================
# Configuration
# =============================================================================

CHATJIMMY_URL = "https://chatjimmy.ai/api/chat"
DEFAULT_MODEL = "llama3.1-8B"
AVAILABLE_MODELS = ["llama3.1-8B"]  # Extend as upstream adds more

# API Key authentication (set CHATJIMMY_API_KEY env var to enable)
API_KEY = os.getenv("CHATJIMMY_API_KEY")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Upstream request defaults
DEFAULT_TOP_K = 8
DEFAULT_TIMEOUT = 300

# =============================================================================
# Pydantic Models (OpenAI-compatible request/response schemas)
# =============================================================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            # Handle multi-modal content (text + image_url)
            return v
        return str(v)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=128)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = True


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "chatjimmy"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    usage: Optional[ChatCompletionUsage] = None


class ErrorResponse(BaseModel):
    error: Dict[str, Any]


# =============================================================================
# Utility Functions
# =============================================================================

def verify_api_key(authorization: Optional[str] = Depends(api_key_header)) -> None:
    """Verify API key if configured."""
    if API_KEY is None:
        return  # No auth required
    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Missing Authorization header", "type": "authentication_error"}}
        )
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid Authorization header format", "type": "authentication_error"}}
        )
    token = authorization[7:]
    if token != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid API key", "type": "authentication_error"}}
        )


def map_openai_params_to_chatjimmy(request: ChatCompletionRequest) -> Dict[str, Any]:
    """Map OpenAI parameters to ChatJimmy's chatOptions."""
    chat_options = {
        "selectedModel": request.model,
        "systemPrompt": "",
        "topK": DEFAULT_TOP_K,
    }

    # Map temperature -> topK (must be <= 8 for upstream)
    if request.temperature is not None:
        # map temp 0.0-2.0 to topK 1-8
        mapped_topk = int(request.temperature * 4 + 1)
        chat_options["topK"] = max(1, min(8, mapped_topk))

    # Map top_p if provided (ChatJimmy doesn't support top_p directly)
    # Could adjust topK based on top_p, but we'll just log it

    # max_tokens / max_completion_tokens - not directly supported by ChatJimmy
    # We'll handle truncation post-generation if needed

    # stop sequences - not supported by ChatJimmy upstream

    # response_format (JSON mode) - inject into system prompt
    if request.response_format and request.response_format.get("type") == "json_object":
        chat_options["systemPrompt"] += "\n\nYou must respond with valid JSON only."

    # Tools/function calling - inject into system prompt
    if request.tools:
        tool_descriptions = []
        for tool in request.tools:
            if tool.get("type") == "function":
                fn = tool.get("function", {})
                name = fn.get("name", "")
                desc = fn.get("description", "")
                params = fn.get("parameters", {})
                tool_descriptions.append(f"- {name}: {desc}\n  Parameters: {json.dumps(params)}")
        if tool_descriptions:
            chat_options["systemPrompt"] += "\n\nAvailable functions:\n" + "\n".join(tool_descriptions)
            chat_options["systemPrompt"] += "\n\nTo call a function, respond with a JSON object in this format:\n{\"name\": \"function_name\", \"arguments\": {...}}"

    return chat_options


def convert_messages_to_chatjimmy(messages: List[Union[ChatMessage, Dict[str, Any]]]) -> tuple[List[Dict], str]:
    """Convert OpenAI messages to ChatJimmy format. Returns (chat_messages, system_prompt)."""
    system_prompt = ""
    chat_messages = []

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            name = msg.get("name", "")
        else:
            role = msg.role
            content = msg.content
            name = msg.name or ""

        # Handle multi-modal content (extract text)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = "\n".join(text_parts)

        if role == "system":
            system_prompt += (content or "") + "\n"
        elif role == "tool":
            # Tool results - prepend with context
            tool_content = f"Tool result ({name or 'unknown'}): {content}"
            chat_messages.append({"role": "user", "content": tool_content})
        elif role == "function":
            # Legacy function calling
            tool_content = f"Function result ({name}): {content}"
            chat_messages.append({"role": "user", "content": tool_content})
        else:
            chat_messages.append({"role": role, "content": content or ""})

    return chat_messages, system_prompt.strip()


def call_chatjimmy(messages: List[ChatMessage], chat_options: Dict[str, Any]) -> tuple[str, Optional[Dict]]:
    """Call ChatJimmy API and return (text, stats)."""
    chat_messages, system_prompt = convert_messages_to_chatjimmy(messages)

    # Merge system prompt from messages with any from chat_options
    if chat_options.get("systemPrompt"):
        system_prompt = chat_options["systemPrompt"] + "\n" + system_prompt

    payload = {
        "messages": chat_messages,
        "chatOptions": {
            "selectedModel": chat_options.get("selectedModel", DEFAULT_MODEL),
            "systemPrompt": system_prompt,
            "topK": chat_options.get("topK", DEFAULT_TOP_K),
        },
        "attachment": None
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
        "Content-Type": "application/json",
        "Origin": "https://chatjimmy.ai",
        "Referer": "https://chatjimmy.ai/",
    }

    response = requests.post(
        CHATJIMMY_URL,
        headers=headers,
        json=payload,
        timeout=DEFAULT_TIMEOUT,
    )

    response.raise_for_status()
    body = response.text

    # Parse response: split text from <|stats|> block
    stats = None
    if "<|stats|>" in body:
        text, stats_part = body.split("<|stats|>", 1)
        if "<|/stats|>" in stats_part:
            stats_json = stats_part.split("<|/stats|>", 1)[0]
            try:
                stats = json.loads(stats_json)
            except json.JSONDecodeError:
                pass
    else:
        text = body

    return text.strip(), stats


def build_usage_from_stats(stats: Optional[Dict]) -> Optional[ChatCompletionUsage]:
    """Build OpenAI usage object from ChatJimmy stats."""
    if not stats:
        return None
    return ChatCompletionUsage(
        prompt_tokens=stats.get("prefill_tokens", 0),
        completion_tokens=stats.get("decode_tokens", 0),
        total_tokens=stats.get("total_tokens", 0),
    )


def create_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    status_code: int = 400,
    param: Optional[str] = None,
    code: Optional[str] = None
) -> JSONResponse:
    """Create OpenAI-compatible error response."""
    error = {
        "message": message,
        "type": error_type,
    }
    if param:
        error["param"] = param
    if code:
        error["code"] = code
    return JSONResponse(status_code=status_code, content={"error": error})


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting ChatJimmy OpenAI Adapter")
    print(f"Upstream: {CHATJIMMY_URL}")
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"Auth enabled: {API_KEY is not None}")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="ChatJimmy OpenAI-Compatible API",
    description="OpenAI-compatible API adapter for chatjimmy.ai",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/v1/models", response_model=ModelsResponse, dependencies=[Depends(verify_api_key)])
async def list_models():
    """List available models."""
    return ModelsResponse(
        data=[ModelInfo(id=model_id) for model_id in AVAILABLE_MODELS]
    )


@app.get("/v1/models/{model_id}", response_model=ModelInfo, dependencies=[Depends(verify_api_key)])
async def get_model(model_id: str):
    """Get model info."""
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": f"Model '{model_id}' not found", "type": "invalid_request_error", "code": "model_not_found"}}
        )
    return ModelInfo(id=model_id)


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    # Validate model
    if body.model not in AVAILABLE_MODELS:
        return create_error_response(
            f"Model '{body.model}' not found. Available: {AVAILABLE_MODELS}",
            "invalid_request_error",
            404,
            param="model",
            code="model_not_found"
        )

    # Validate messages
    if not body.messages:
        return create_error_response(
            "messages is required",
            "invalid_request_error",
            400,
            param="messages"
        )

    # Check for unsupported features
    if body.n and body.n > 1:
        return create_error_response(
            "n > 1 not supported",
            "invalid_request_error",
            400,
            param="n"
        )

    if body.logit_bias:
        return create_error_response(
            "logit_bias not supported",
            "invalid_request_error",
            400,
            param="logit_bias"
        )

    # Map OpenAI params to ChatJimmy
    chat_options = map_openai_params_to_chatjimmy(body)

    # Generate completion ID and timestamp
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    try:
        # Call upstream
        text, stats = call_chatjimmy(body.messages, chat_options)

        # Handle max_tokens truncation
        max_tokens = body.max_completion_tokens or body.max_tokens
        if max_tokens and stats and stats.get("decode_tokens", 0) > max_tokens:
            # Rough truncation - in practice we'd need to re-request or truncate
            # For now, just note it
            pass

        # Build usage
        usage = build_usage_from_stats(stats)

        # Handle tool calls in response (parse JSON from text)
        tool_calls = None
        finish_reason = "stop"
        if body.tools and text.strip().startswith("{"):
            try:
                parsed = json.loads(text)
                if "name" in parsed and "arguments" in parsed:
                    tool_calls = [{
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": parsed["name"],
                            "arguments": json.dumps(parsed["arguments"])
                        }
                    }]
                    finish_reason = "tool_calls"
                    text = None  # OpenAI returns null content when tool_calls present
            except json.JSONDecodeError:
                pass

        # Build response message
        response_message = ChatMessage(
            role="assistant",
            content=text,
            tool_calls=tool_calls
        )

        if not body.stream:
            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=body.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=response_message,
                    finish_reason=finish_reason
                )],
                usage=usage,
                system_fingerprint=f"fp_{uuid.uuid4().hex[:12]}"
            )

        # Streaming response with simulated incremental chunks for Zed / UI clients
        async def generate():
            # Initial role chunk
            role_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=body.model,
                choices=[ChatCompletionChunkChoice(
                    index=0,
                    delta={"role": "assistant", "content": ""},
                    finish_reason=None
                )]
            )
            yield f"data: {role_chunk.model_dump_json(exclude_none=True)}\n\n"
            await asyncio.sleep(0.01)

            if text:
                # Stream content in small word/character chunks so Zed renders it progressively
                chunk_size = 4  # characters per chunk
                for i in range(0, len(text), chunk_size):
                    piece = text[i:i + chunk_size]
                    content_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=body.model,
                        choices=[ChatCompletionChunkChoice(
                            index=0,
                            delta={"content": piece},
                            finish_reason=None
                        )]
                    )
                    yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"
                    await asyncio.sleep(0.01)

            # Final chunk with finish_reason and usage
            final_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=body.model,
                choices=[ChatCompletionChunkChoice(
                    index=0,
                    delta={},
                    finish_reason=finish_reason
                )],
                usage=usage
            )
            yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    except requests.Timeout as e:
        print(f"[DEBUG] Upstream timeout: {e}")
        return create_error_response(
            "Upstream request timed out",
            "server_error",
            504
        )
    except requests.HTTPError as e:
        print(f"[DEBUG] Upstream HTTP error: {e.response.status_code} - {e.response.text}")
        return create_error_response(
            f"Upstream error: {e.response.status_code} - {e.response.text[:200]}",
            "server_error",
            502
        )
    except Exception as e:
        import traceback
        print(f"[DEBUG] Internal exception: {str(e)}")
        traceback.print_exc()
        return create_error_response(
            f"Internal error: {str(e)}",
            "server_error",
            500
        )


# Legacy completions endpoint (for older clients)
class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(default=16, ge=1)
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=128)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None


@app.post("/v1/completions", dependencies=[Depends(verify_api_key)])
async def completions(request: Request, body: CompletionRequest):
    """Legacy completions endpoint - converts to chat format."""
    # Convert prompt to chat messages
    if isinstance(body.prompt, list):
        prompt_text = "\n".join(body.prompt)
    else:
        prompt_text = body.prompt

    chat_request = ChatCompletionRequest(
        model=body.model,
        messages=[ChatMessage(role="user", content=prompt_text)],
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        stream=body.stream,
        stop=body.stop,
        presence_penalty=body.presence_penalty,
        frequency_penalty=body.frequency_penalty,
        user=body.user,
    )

    # Reuse chat_completions logic
    return await chat_completions(request, chat_request)


# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "upstream": CHATJIMMY_URL}


# Root
@app.get("/")
async def root():
    return {
        "name": "ChatJimmy OpenAI-Compatible API",
        "version": "1.0.0",
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "completions": "/v1/completions",
        },
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
