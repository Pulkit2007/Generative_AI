import nltk
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

nltk.download("punkt_tab")

docs_dir_path = r"C:\Users\LENOVO\GenAI\Prompting\docs_dir"
vector_db_path = r"C:\Users\LENOVO\GenAI\Prompting\vector_db"
collection_name = "document_collection"

embedding = HuggingFaceEmbeddings()

loader = DirectoryLoader(
    path = docs_dir_path,
    glob = "./*.pdf",
    loader_cls = PyPDFLoader
)

documents = loader.load()
print(f"Total pages loaded: {len(documents)}")

text_splitter = CharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 500
)

text_chunks = text_splitter.split_documents(documents)

vector_store = Chroma.from_documents(
    documents=text_chunks,
    embedding=embedding,
    persist_directory= vector_db_path,
    collection_name = collection_name
)