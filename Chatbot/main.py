import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(
    page_title="ChatBot",
    page_icon="🤖",
    layout="centered",
)

st.title("Generative AI ChatBot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


user_prompt = st.chat_input("Ask ChatBot")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    response = llm.invoke(
        input=[
            {"role": "system", "content": "You are a helpful assistant"},
            *st.session_state.chat_history,
        ]
    )

    assistant_response = response.content[0]['text']
    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_response}
    )

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
