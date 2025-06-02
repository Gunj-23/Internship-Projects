import datetime

def get_current_time():
    return datetime.datetime.now().strftime("%I:%M %p")

def chatbot_response(user_input):
    user_input = user_input.lower()

    if any(greet in user_input for greet in ["hello", "hi", "hey", "greetings"]):
        return "Hello! How can I assist you today?"

    elif "how are you" in user_input:
        return "I'm an AI chatbot, so I don't have feelings, but I'm here to help you!"

    elif "time" in user_input:
        current_time = get_current_time()
        return f"The current time is {current_time}."

    elif "weather" in user_input:
        return "I'm sorry, I can't provide weather updates right now. Try asking a weather app!"

    elif "help" in user_input or "assist" in user_input:
        return ("I can answer basic questions like greetings, time, or common queries. "
                "Try asking me about the time or just say hello!")

    elif "thank" in user_input:
        return "You're welcome! If you have any more questions, feel free to ask."

    elif "exit" in user_input or "bye" in user_input or "quit" in user_input:
        return "Goodbye! Have a great day ahead."

    else:
        return ("I'm sorry, I don't have an answer for that yet. "
                "You can ask me about the time, greetings, or say 'help'.")

def main():
    print("Professional ChatBot (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("ChatBot: Goodbye! Have a great day ahead.")
            break

        response = chatbot_response(user_input)
        print("ChatBot:", response)

if __name__ == "__main__":
    main()
