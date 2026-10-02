from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import requests
import json
import time
import uuid


app = FastAPI()


CHATJIMMY_URL = "https://chatjimmy.ai/api/chat"
MODEL = "llama3.1-8B"


def call_chatjimmy(messages):
    """
    Call ChatJimmy and return:
        text, stats
    """

    # Convert OpenAI-style messages into what ChatJimmy expects.
    system_prompt = ""
    chat_messages = []

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role == "system":
            system_prompt += content + "\n"
        else:
            chat_messages.append({
                "role": role,
                "content": content
            })

    payload = {
        "messages": chat_messages,
        "chatOptions": {
            "selectedModel": MODEL,
            "systemPrompt": system_prompt.strip(),
            "topK": 8
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
        timeout=300,
    )

    response.raise_for_status()

    body = response.text

    # Separate generated text from ChatJimmy stats.
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


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "chatjimmy"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):

    body = await request.json()

    messages = body.get("messages", [])
    model = body.get("model", MODEL)
    stream = body.get("stream", False)

    if not messages:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "messages is required",
                    "type": "invalid_request_error"
                }
            }
        )

    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    text, stats = call_chatjimmy(messages)

    if not stream:

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"
                }
            ]
        }

    # Option 1:
    # The upstream has already buffered the entire response,
    # so send one complete OpenAI-compatible chunk.

    def generate():

        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": None
                }
            ]
        }

        yield f"data: {json.dumps(chunk)}\n\n"

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }

        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )