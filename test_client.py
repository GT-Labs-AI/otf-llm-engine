# test_client.py
import requests, json

url = "http://localhost:8000/v1/chat/completions"

payload = {
    "messages": [
        {"role": "user", "content": "Расскажи забавный факты про кошек и программистов."}
    ],
    "stream": True,
    "max_tokens": 150
}

response = requests.post(url, json=payload, stream=True)

print("📝 Ответ от OTF Triton 7B Engine (потоковый стриминг):\n")
for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith("data: ") and not line_str.endswith("[DONE]"):
            data_json = json.loads(line_str[6:])
            content = data_json["choices"][0]["delta"].get("content", "")
            print(content, end="", flush=True)

print("\n\n✅ Стриминг завершен!")