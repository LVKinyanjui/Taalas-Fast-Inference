import requests
import json
import time

url = "https://chatjimmy.ai/api/chat"

payload = {
    "messages": [
        {
            "role": "user",
            "content": "write a short explanation of why the sky is blue"
        }
    ],
    "chatOptions": {
        "selectedModel": "llama3.1-8B",
        "systemPrompt": "",
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

start = time.perf_counter()

with requests.post(
    url,
    headers=headers,
    json=payload,
    stream=True
) as response:

    print("status:", response.status_code)
    print("content-type:", response.headers.get("content-type"))
    print()

    for i, line in enumerate(response.iter_lines(decode_unicode=True)):
        elapsed = time.perf_counter() - start

        print(
            f"[{elapsed:8.3f}s] "
            f"chunk {i}: {line!r}"
        )
