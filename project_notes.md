# ChatJimmy → OpenAI-Compatible API Adapter — Exploration Notes

## Goal

Convert the existing `https://chatjimmy.ai/api/chat` model endpoint into an **OpenAI-compatible HTTP API**, so standard OpenAI-compatible clients can use it through a base URL such as:

```text
https://<domain>/v1/
```

The intended first endpoints are:

```text
GET  /v1/models
POST /v1/chat/completions
```

The adapter will initially use **buffered upstream responses**. The upstream endpoint advertises `text/event-stream`, but testing showed that the generated response is effectively delivered all at once after generation has completed.

---

## 1. Original ChatJimmy request

The upstream endpoint is:

```text
POST https://chatjimmy.ai/api/chat
```

A working request observed during exploration:

```python
import requests
import json

url = "https://chatjimmy.ai/api/chat"

payload = json.dumps({
  "messages": [
    {
      "role": "user",
      "content": "what is the square of 20"
    }
  ],
  "chatOptions": {
    "selectedModel": "llama3.1-8B",
    "systemPrompt": "",
    "topK": 8
  },
  "attachment": None
})

headers = {
  "User-Agent": "Mozilla/5.0",
  "Accept": "*/*",
  "Accept-Language": "en-US,en;q=0.9",
  "Accept-Encoding": "gzip, deflate, br, zstd",
  "Referer": "https://chatjimmy.ai/",
  "Content-Type": "application/json",
  "Origin": "https://chatjimmy.ai",
  "Connection": "keep-alive",
  "Sec-Fetch-Dest": "empty",
  "Sec-Fetch-Mode": "cors",
  "Sec-Fetch-Site": "same-origin",
  "Priority": "u=0"
}

response = requests.post(
    url,
    headers=headers,
    data=payload
)

print(response.text)
```

The important request fields are:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "what is the square of 20"
    }
  ],
  "chatOptions": {
    "selectedModel": "llama3.1-8B",
    "systemPrompt": "",
    "topK": 8
  },
  "attachment": null
}
```

---

## 2. Observed response format

A typical response looks like:

```text
The square of 20 is 400.
<|stats|>{"created_at":1790929535.8402138,"done":true,"done_reason":"stop","total_duration":9.971221446990967,"logprobs":null,"topk":8,"ttft":0.0011589527130126953,"reason":"termination token 128009/<|eot_id|> detected","status":0,"prefill_tokens":18,"prefill_rate":15531.263526023453,"decode_tokens":10,"decode_rate":16584.83194938711,"total_tokens":28,"total_time":0.001775503158569336,"roundtrip_time":10099}<|/stats|>
```

Another observed response:

```text
The square of 50 is 2500. 

In other words, 50 × 50 = 2500.
<|stats|>{"created_at":1790930715.7659063,"done":true,"done_reason":"stop","total_duration":1.9955310821533203,"logprobs":null,"topk":8,"ttft":0.0010437965393066406,"reason":"termination token 128009/<|eot_id|> detected","status":0,"prefill_tokens":18,"prefill_rate":17244.740063956146,"decode_tokens":26,"decode_rate":14893.7317672767,"total_tokens":44,"total_time":0.0028002262115478516,"roundtrip_time":2131}<|/stats|>
```

The response consists of:

1. Generated model text.
2. A special stats block:
   ```text
   <|stats|>
   {JSON}
   <|/stats|>
   ```

The stats block should not be included in the normal OpenAI `message.content`.

---

## 3. Response Content-Type

The HTTP response reports:

```text
status: 200
content-type: text/event-stream; charset=utf-8
```

This initially suggested that the upstream might provide a streaming response.

We therefore tested it using:

```python
response = requests.post(
    url,
    headers=headers,
    json=payload,
    stream=True
)

for i, line in enumerate(response.iter_lines(decode_unicode=True)):
    print(repr(line))
```

---

## 4. Actual upstream streaming behavior

The observed output was:

```text
status: 200
content-type: text/event-stream; charset=utf-8

[   4.288s] chunk 0: "The sky appears blue to our eyes because of the way light interacts with the atmosphere. Here's a simplified explanation:"
[   4.291s] chunk 1: ''
[   4.291s] chunk 2: '**Scattering of Light**'
[   4.292s] chunk 3: ''
[   4.292s] chunk 4: 'When sunlight enters our atmosphere, it is made up of all the colors of the visible spectrum, including all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). However, our atmosphere scatters the shorter wavelength light more than the longer wavelength light. This is known as the "Tyndall effect" or "Rayleigh scattering".'
[   4.292s] chunk 5: ''
[   4.292s] chunk 6: '**Why Blue Light is Scattered More**'
[   4.292s] chunk 7: ''
[   4.292s] chunk 8: "The reason blue light is scattered more than other colors is because it has a shorter wavelength, around 450-495 nanometers. This is close to the range of wavelengths that our atmosphere's molecular particles (such as nitrogen and oxygen) can absorb and scatter. As a result, blue light is continuously scattered in all directions by these particles, making the sky appear blue to our eyes."
[   4.292s] chunk 9: ''
[   4.292s] chunk 10: "**Why Red Light isn't Scattered**"
[   4.292s] chunk 11: ''
[   4.292s] chunk 12: "Red light, on the other hand, has a much longer wavelength, around 620-750 nanometers. Since red light is not scattered as much as blue light, our atmosphere doesn't scatter as much of this color. When we look at the sky, our eyes see mostly the unscattered light, which is why the sky appears blue rather than red."
[   4.292s] chunk 13: ''
[   4.292s] chunk 14: '**Additional Factors**'
[   4.292s] chunk 15: ''
[   4.292s] chunk 16: 'Other factors like dust, water vapor, and pollutants can also influence the apparent color of the sky, but the basic principle remains the same: blue light is scattered more than other colors, making the sky appear blue.'
[   4.292s] chunk 17: ''
[   4.292s] chunk 18: "That's the basic explanation why the sky is blue!"
[   4.293s] chunk 19: '<|stats|>{"created_at":1790931054.241803,"done":true,"done_reason":"stop","total_duration":1.2595317363739014,"logprobs":null,"topk":8,"ttft":0.0010983943933933,"reason":"termination token 128009/<|eot_id|> detected","status":0,"prefill_tokens":21,"prefill_rate":19118.815715215977,"decode_tokens":333,"decode_rate":14221.309330835336,"total_tokens":354,"total_time":0.024521589279569,"roundtrip_time":1468}<|/stats|>'
```

### Conclusion

Although the endpoint advertises:

```text
text/event-stream
```

it does **not** appear to deliver the generated text incrementally over the network.

The complete generated response arrives at approximately the same time:

```text
4.288s
4.291s
4.292s
4.293s
```

There is no meaningful token-by-token arrival.

Therefore, the adapter should **not pretend that it receives genuine upstream streaming**.

---

## 5. Time-to-first-token / latency observation

There is significant latency before the client receives the generated content.

In the test above, the first response chunk arrived around:

```text
4.288 seconds
```

The ChatJimmy stats for that request reported:

```json
{
  "total_duration": 1.2595317363739014,
  "ttft": 0.0010983943933933,
  "prefill_tokens": 21,
  "prefill_rate": 19118.815715215977,
  "decode_tokens": 333,
  "decode_rate": 14221.309330835336,
  "total_tokens": 354,
  "total_time": 0.024521589279569,
  "roundtrip_time": 1468
}
```

The upstream-reported timing and the wall-clock time observed by the Python client differ substantially.

This is important: **the OpenAI-compatible adapter cannot eliminate the upstream latency**. It can only expose the upstream service in a standard API format.

---

## 6. Parsing the upstream response

The basic parsing strategy is to separate the generated text from the stats marker:

```python
STATS_START = "<|stats|>"
STATS_END = "<|/stats|>"

body = response.text

if STATS_START in body:
    text, stats_part = body.split(STATS_START, 1)

    if STATS_END in stats_part:
        stats_json = stats_part.split(STATS_END, 1)[0]

        try:
            stats = json.loads(stats_json)
        except json.JSONDecodeError:
            stats = None
    else:
        stats = None
else:
    text = body
    stats = None

text = text.strip()
```

For example:

```text
The square of 50 is:

50² = 50 × 50 = 2500
<|stats|>{...}<|/stats|>
```

becomes:

```python
text = """The square of 50 is:

50² = 50 × 50 = 2500"""
```

and `stats` becomes a Python dictionary.

---

# OpenAI-Compatible Adapter Design

## 7. Desired API surface

The initial server should expose:

```text
GET  /v1/models
POST /v1/chat/completions
```

The base URL should therefore be:

```text
https://<domain>/v1/
```

This allows OpenAI-compatible clients to use a base URL such as:

```python
client = OpenAI(
    base_url="https://your-domain/v1",
    api_key="anything"
)
```

The API key can initially be ignored by the adapter.

Authentication can be added later.

---

## 8. Streaming decision

We chose **Option 1: buffered upstream + OpenAI-compatible SSE output**.

For:

```json
{
  "stream": false
}
```

the adapter will:

1. Send the request to ChatJimmy.
2. Wait for the complete upstream response.
3. Strip the `<|stats|>...</|stats|>` section.
4. Return a normal OpenAI-style `chat.completion` JSON response.

For:

```json
{
  "stream": true
}
```

the adapter will:

1. Send the request to ChatJimmy.
2. Wait for the complete upstream response.
3. Strip the stats block.
4. Emit the completed response as an OpenAI-style SSE stream.
5. Emit the final chunk.
6. Emit:
   ```text
   data: [DONE]
   ```

This means `stream=true` will be **protocol-compatible**, but it will not provide genuine token-by-token streaming because the upstream itself is not delivering tokens incrementally.

We explicitly chose this approach rather than artificially splitting the completed text into fake incremental chunks.

---

# 9. Current prototype

A first FastAPI prototype was designed as follows:

```python
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
```

---

# 10. Running the prototype

Dependencies:

```bash
pip install fastapi uvicorn requests
```

Start:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

The API then becomes:

```text
http://localhost:8000/v1/
```

Endpoints:

```text
GET  http://localhost:8000/v1/models
POST http://localhost:8000/v1/chat/completions
```

---

# 11. Direct tests

### Non-streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1-8B",
    "messages": [
      {
        "role": "user",
        "content": "what is the square of 50?"
      }
    ]
  }'
```

Expected general shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1790930000,
  "model": "llama3.1-8B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The square of 50 is 2500..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

### Streaming

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1-8B",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "what is the square of 50?"
      }
    ]
  }'
```

Expected structure:

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}

data: [DONE]
```

Again, the content will arrive after the upstream request has completed.

---

# 12. Current limitations / things not yet implemented

## Model handling

The upstream currently uses:

```json
"selectedModel": "llama3.1-8B"
```

The prototype has this hard-coded:

```python
MODEL = "llama3.1-8B"
```

The next step may be to validate the requested OpenAI `model` against available upstream models or map multiple model names.

## Sampling parameters

ChatJimmy currently exposes:

```json
"topK": 8
```

in the observed request.

We have not yet established how/if it maps OpenAI parameters such as:

```text
temperature
top_p
max_tokens / max_completion_tokens
presence_penalty
frequency_penalty
```

Those should not be assumed to work until tested against the upstream API.

## Usage

The upstream stats contain useful token information:

```text
prefill_tokens
decode_tokens
total_tokens
```

but the prototype currently does not expose an OpenAI-style:

```json
"usage": {
    ...
}
```

field.

This can be added later after deciding exactly how to map the upstream values.

## Error handling

Only basic HTTP error handling currently exists:

```python
response.raise_for_status()
```

A production version should translate upstream errors into OpenAI-style error JSON.

## Authentication

The prototype does not authenticate callers.

The `api_key` supplied by OpenAI-compatible clients can initially be ignored.

Authentication can be added later if the endpoint will be exposed publicly.

## Other OpenAI endpoints

Only these are currently planned/implemented:

```text
GET  /v1/models
POST /v1/chat/completions
```

No embeddings, responses API, image generation, audio, files, etc. are currently involved.

---

# 13. Important architectural conclusion

The core architecture is now understood:

```text
OpenAI-compatible client
        |
        | POST /v1/chat/completions
        v
+---------------------------+
| FastAPI adapter           |
|                           |
| OpenAI request parsing    |
| message conversion        |
| response formatting       |
+-------------+-------------+
              |
              | POST /api/chat
              v
+---------------------------+
| ChatJimmy                 |
|                           |
| llama3.1-8B               |
| topK=8                    |
+-------------+-------------+
              |
              | buffered response
              v
+---------------------------+
| generated text            |
|                           |
| <|stats|>{...}</|stats|> |
+-------------+-------------+
              |
              v
       FastAPI adapter
              |
              +--> strip stats
              |
              +--> OpenAI JSON
              |
              v
       OpenAI-compatible client
```

The key finding is that **ChatJimmy's advertised SSE content type does not currently give us usable incremental generation**. Therefore the adapter should buffer the upstream response and then expose a standard OpenAI-compatible result.

---

# 14. Recommended next incremental steps

1. **Run the FastAPI prototype locally.**
2. Verify:
   ```text
   GET /v1/models
   ```
3. Verify:
   ```text
   POST /v1/chat/completions
   ```
   with `stream=false`.
4. Verify `stream=true` produces valid OpenAI-style SSE.
5. Test with an OpenAI-compatible Python client using:
   ```python
   OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
   ```
6. Improve message conversion, especially system messages.
7. Map useful OpenAI request parameters to ChatJimmy.
8. Add OpenAI-style `usage` from ChatJimmy stats.
9. Improve error handling.
10. Add authentication if required.
11. Deploy behind the intended domain/reverse proxy so clients use:
    ```text
    https://<domain>/v1/
    ```

---

## Important facts to preserve for future work

- Upstream URL: `https://chatjimmy.ai/api/chat`
- Upstream model tested: `llama3.1-8B`
- Upstream request uses `chatOptions.selectedModel`, `chatOptions.systemPrompt`, and `chatOptions.topK`.
- Upstream response HTTP status observed: `200`.
- Upstream `Content-Type`: `text/event-stream; charset=utf-8`.
- Despite that Content-Type, observed chunks arrive essentially simultaneously after generation.
- The upstream response ends with:
  ```text
  <|stats|>{JSON}<|/stats|>
  ```
- Stats should be separated from model text.
- There is significant upstream latency before the first content reaches the client.
- We deliberately chose **buffered upstream + OpenAI-compatible output** rather than fake token streaming.
- Target API base path:
  ```text
  /v1/
  ```
- Initial endpoints:
  ```text
  GET  /v1/models
  POST /v1/chat/completions
  ```
- FastAPI is the current implementation direction.
