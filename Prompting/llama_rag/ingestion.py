from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
import nltk

nltk.download("punkt_tab")
nltk.download("stopwords")

# configuration
docs_db_path = r"C:\Users\LENOVO\GenAI\Prompting\docs_dir"
vector_db_path = r"C:\Users\LENOVO\GenAI\Prompting\vector1_db"
collection_name = "documents_collection"

# embedding model
embed_model = HuggingFaceEmbedding()

# settting up the loader
loader = SimpleDirectoryReader(input_dir=docs_db_path)

documents = loader.load_data()
print(len(documents))

db = chromadb.PersistentClient(path=vector_db_path)
chroma_collection = db.get_or_create_collection(name=collection_name)

parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    vector_store=vector_store,
    embed_model=embed_model,
)

print("vector database created ")
