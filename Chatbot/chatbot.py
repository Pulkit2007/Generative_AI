from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-3.5-flash-lite")

parser = StrOutputParser()

def chat():
    chat_history = [("system", "you are a helpful chatbot. Be concise and accurate")]

    print("Langchain chatbot, type exit to quit")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        chat_history.append(("user", user_input))

        prompt = ChatPromptTemplate.from_messages(chat_history)

        chain = prompt | llm | parser

        response = chain.invoke({})

        print("Bot: ", response)
        chat_history.append(("assistant", response))
        print("-"*50)
        for text in chat_history:
            print(text)


chat()
