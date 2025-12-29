from transformers import pipeline

# Load a small local model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")

print("Offline Chatbot (type 'exit' to stop)")
chat_history = ""

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break

    response = chatbot(user_input, max_new_tokens=50)
    reply = response[0]['generated_text']
    print("Bot:", reply)
