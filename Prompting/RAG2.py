#DOCUMENT RETRIEVAL

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-3.5-flash-lite")

vector_db_path = r"C:\Users\LENOVO\GenAI\Prompting\vector_db"
collection_name = "document_collection"

embedding = HuggingFaceEmbeddings()

vector_store = Chroma(
    collection_name=collection_name,
    embedding_function = embedding,
    persist_directory=vector_db_path
)

retriever = vector_store.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = retriever,
    return_source_documents = True
)

query = "What does the document say about Evolution and Ecosystem?"
response = qa_chain.invoke({"query":query})
print(response["result"])
print("-"*80)
for source in response["source_documents"]:
    print(source.metadata)

