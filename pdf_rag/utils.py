import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain_classic.chains import RetrievalQA


load_dotenv()

working_dir = os.path.dirname(os.path.abspath((__file__)))

embeddings = HuggingFaceEmbeddings()

llm = GoogleGenerativeAI(model = "gemini-3.5-flash-lite")

def document_to_chromadb(filename):
    loader = UnstructuredPDFLoader(f"{working_dir}/{filename}")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 200,
    )

    text = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=text,
        embedding=embeddings,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )   

    return 0

def answer_query(user_query):

    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        chain_type = "stuff",
        llm = llm,
        retriever = retriever
    )

    response = qa_chain.invoke({"query" : user_query})
    answer = response['result']

    return answer