import requests

def chat_with_ollama(prompt):
    url = "http://localhost:11434/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama2",
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    reply = response.json()
    return reply['choices'][0]['message']['content']

print("🤖 Ollama Chatbot: Hello! Type 'exit' to quit.")
while True:
    user_input = input("👤 You: ")
    if user_input.lower() == "exit":
        print("🤖 Ollama Chatbot: Goodbye!")
        break
    try:
        bot_response = chat_with_ollama(user_input)
        print(f"🤖 Ollama Chatbot: {bot_response}\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        break
