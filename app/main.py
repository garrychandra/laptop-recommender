xfrom app.chatbot import chatbot_reply

print("🤖 Laptop Recommender Chatbot (Bahasa Indonesia)")
print("Tulis 'keluar' untuk berhenti.")
print("--------------------------------------")

while True:
    user = input("Anda: ")
    if user.lower() in ["keluar", "exit", "bye", "dadah", "stop", "quit"]:
        print("Bot: Terima kasih! Semoga membantu 😊")
        break
    reply = chatbot_reply(user)
    print("Bot:", reply)
    if(reply == "Terima kasih! Semoga harimu menyenangkan. 👋"):
        break
