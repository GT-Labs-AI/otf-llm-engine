# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# tests/test_client.py

import json
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"

payload = {
    "messages": [
        {"role": "user", "content": "Tell me a fun fact about cats and programmers."}
    ],
    "stream": True,
    "max_tokens": 150
}

print("📝 Streaming Response from OTF Triton Engine (Zero-Dependency urllib client):\n")

req = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as response:
        for line in response:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: ") and not line_str.endswith("[DONE]"):
                try:
                    data_json = json.loads(line_str[6:])
                    content = data_json["choices"][0]["delta"].get("content", "")
                    print(content, end="", flush=True)
                except json.JSONDecodeError:
                    pass

    print("\n\n✅ Streaming Completed Successfully!")
except Exception as e:
    print(f"\n❌ Client Error: {e}")