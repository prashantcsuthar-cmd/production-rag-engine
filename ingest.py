import os
import qdrant_client
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Load environment variables from .env file
load_dotenv()


def run_ingestion():
    # 1. Initialize Local Qdrant Storage
    print("🚀 Step 1: Initializing Local Qdrant Database...")
    client = qdrant_client.QdrantClient(path="./qdrant_data")

    # 2. Configure Embedding Model
    print("🧠 Step 2: Configuring Google GenAI Embedding Model...")
    embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")
    Settings.embed_model = embed_model
    Settings.llm = None  # Pure retrieval indexing

    # 3. Setup Vector Store & Storage Context
    print("⚡ Step 3: Configuring Qdrant Vector Store...")
    vector_store = QdrantVectorStore(client=client, collection_name="rag_documents")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Load Local Documents
    print("📄 Step 4: Reading documents from ./data folder...")
    documents = SimpleDirectoryReader("./data").load_data()
    print(f"   └─ Loaded {len(documents)} document file(s).")

    # 5. Generate Embeddings and Ingest
    print("💾 Step 5: Generating Embeddings via API and Ingesting into Qdrant...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    print("\n✅ Ingestion complete! Vectors successfully stored in ./qdrant_data")


if __name__ == "__main__":
    run_ingestion()