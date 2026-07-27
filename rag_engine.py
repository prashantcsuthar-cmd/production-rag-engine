import os
import shutil
import qdrant_client
from dotenv import load_dotenv

from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

load_dotenv()

# --- USE LOCAL BGE EMBEDDINGS (NO RATE LIMITS) ---
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash")

COLLECTION_NAME = "rag_documents"

# Check Qdrant Cloud environment variables vs local disk
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

class RAGEngine:
    def __init__(self):
        # Support Qdrant Cloud if set, otherwise fall back to local disk
        if QDRANT_URL and QDRANT_API_KEY:
            self.client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            QDRANT_PATH = "./qdrant_data"
            os.makedirs(QDRANT_PATH, exist_ok=True)
            self.client = qdrant_client.QdrantClient(path=QDRANT_PATH)

        self.vector_store = QdrantVectorStore(client=self.client, collection_name=COLLECTION_NAME)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Memory buffer to remember last 10 conversational turns
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
        self.chat_engine = None
        self._init_chat_engine()

    def _init_chat_engine(self):
        """Loads index and connects the conversational memory chat engine."""
        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context
            )
            self.chat_engine = CondensePlusContextChatEngine.from_defaults(
                retriever=index.as_retriever(similarity_top_k=3),
                memory=self.memory,
                llm=Settings.llm,
                system_prompt=(
                    "You are an expert enterprise AI assistant. Answer queries strictly using the "
                    "provided context. If the answer cannot be found in the context, state that clearly."
                )
            )
        except Exception:
            # Index might be empty initially
            self.chat_engine = None

    def ingest_file(self, file_path: str):
        """Processes and embeds a single file into Qdrant using local embeddings."""
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context
        )
        self._init_chat_engine()
        return len(documents)

    def query(self, message: str):
        """Queries the engine and returns response along with detailed source metadata."""
        if not self.chat_engine:
            self._init_chat_engine()
            if not self.chat_engine:
                return {
                    "answer": "No documents ingested yet. Please upload a PDF first!",
                    "sources": []
                }

        response = self.chat_engine.chat(message)
        
        # Extract source citations & metadata
        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                sources.append({
                    "file_name": node.metadata.get("file_name", "Unknown File"),
                    "page_label": node.metadata.get("page_label", "N/A"),
                    "score": round(node.score, 3) if node.score else None,
                    "text_snippet": node.node.get_content()[:200] + "..."
                })

        return {
            "answer": response.response,
            "sources": sources
        }