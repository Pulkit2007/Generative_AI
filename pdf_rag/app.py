import os
import streamlit as st
from utils import document_to_chromadb, answer_query

working_dir = os.path.dirname(os.path.abspath((__file__)))

st.title("Get Answers from your PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"] )

if uploaded_file is not None:
    save_path = os.path.join(working_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_documents = document_to_chromadb(uploaded_file.name)
    st.info("Document uploaded successfully")

user_question = st.text_area('Ask your question about the document')

if st.button("Answer"):

    answer = answer_query(user_question)
    st.markdown("### Document Response")
    st.markdown(answer)