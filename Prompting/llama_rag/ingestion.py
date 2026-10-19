import time
import nltk
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# =========================
# TIMING FUNCTION
# =========================

def timed(label, func):
    print(f"\n[START] {label}", flush=True)
    start = time.perf_counter()

    try:
        result = func()
    except Exception:
        elapsed = time.perf_counter() - start
        print(
            f"[FAILED] {label} after {elapsed:.2f} seconds",
            flush=True,
        )
        raise

    elapsed = time.perf_counter() - start
    print(f"[DONE] {label}: {elapsed:.2f} seconds", flush=True)

    return result


total_start = time.perf_counter()


# =========================
# NLTK DOWNLOADS
# =========================

timed(
    "Downloading NLTK punkt_tab",
    lambda: nltk.download("punkt_tab"),
)

timed(
    "Downloading NLTK stopwords",
    lambda: nltk.download("stopwords"),
)


# =========================
# CONFIGURATION
# =========================

docs_db_path = r"C:\Users\LENOVO\GenAI\Prompting\docs_dir"
vector_db_path = r"C:\Users\LENOVO\GenAI\Prompting\vector1_db"
collection_name = "documents_collection"


# =========================
# LOAD EMBEDDING MODEL
# =========================

embed_model = timed(
    "Loading HuggingFace embedding model",
    lambda: HuggingFaceEmbedding(),
)


# =========================
# LOAD DOCUMENTS
# =========================

loader = SimpleDirectoryReader(input_dir=docs_db_path)

documents = timed(
    "Loading documents",
    loader.load_data,
)

print(f"Number of documents loaded: {len(documents)}", flush=True)


# =========================
# SPLIT DOCUMENTS INTO NODES
# =========================

parser = SimpleNodeParser.from_defaults(
    chunk_size=1024,
    chunk_overlap=50,
)

nodes = timed(
    "Splitting documents into nodes",
    lambda: parser.get_nodes_from_documents(documents),
)

print(f"Number of nodes created: {len(nodes)}", flush=True)


# =========================
# CONNECT TO CHROMA
# =========================

db = timed(
    "Opening Chroma persistent database",
    lambda: chromadb.PersistentClient(path=vector_db_path),
)

chroma_collection = timed(
    "Opening or creating Chroma collection",
    lambda: db.get_or_create_collection(name=collection_name),
)

existing_count = timed(
    "Checking existing vector count",
    chroma_collection.count,
)

print(f"Vectors already in collection: {existing_count}", flush=True)


# =========================
# CONFIGURE VECTOR STORE
# =========================

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# =========================
# CREATE VECTOR INDEX
# =========================

index = timed(
    "Embedding nodes and creating vector index",
    lambda: VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    ),
)


# =========================
# VERIFY COMPLETION
# =========================

final_count = timed(
    "Checking final vector count",
    chroma_collection.count,
)

print(f"\nVectors before ingestion: {existing_count}", flush=True)
print(f"Vectors after ingestion: {final_count}", flush=True)

print(
    f"Total execution time: "
    f"{time.perf_counter() - total_start:.2f} seconds",
    flush=True,
)

print("Vector database created successfully!", flush=True)