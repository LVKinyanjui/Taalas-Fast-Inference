import json
import requests

# 1. Configuration
# Replace with your actual local or remote endpoint (e.g., http://localhost:11434/v1 for Ollama, http://localhost:1234/v1 for LM Studio)
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "sk-..."              # Use 'ollama' or any dummy string if the endpoint doesn't require a key
MODEL_NAME = "llama3.1-8B"                      # Replace with the exact model name hosted on your provider

# 2. Build the target URL and Headers
url = f"{API_BASE_URL}/chat/completions"
headers = {
    "Authorization": f"Base {API_KEY}" if API_KEY else "",
    "Content-Type": "application/json"
}

# 3. Formulate the OpenAI-compatible payload
payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Explain the difference between a REST API and a Webhook in one sentence."
        }
    ],
    "temperature": 0.7,
    "max_tokens": 150
}

try:
    # 4. Make the POST request
    response = requests.post(url, json=payload, headers=headers)

    # Check if the request was successful
    response.raise_for_status()

    # 5. Parse the JSON response
    response_data = response.json()

    # Safely extract the assistant's reply according to the OpenAI spec
    assistant_reply = response_data['choices'][0]['message']['content']
    print("Assistant Response:\n", assistant_reply)

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
