from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

llm = GoogleGenAI(model="gemini-3.5-flash-lite")


def chat():
    chat_history = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful chat assistant. Be concise and accurate",
        )
    ]
    print("llamaIndex chatbot. type exit to qutie\n ")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        chat_history.append(ChatMessage(role=MessageRole.USER, content=user_input))
    
        response = llm.chat(messages=chat_history)
        answer = response.message.content
    
        print(f"Bot: {answer}\n")
    
        chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content = answer))

chat()
    