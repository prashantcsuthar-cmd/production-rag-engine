import os
import shutil
import qdrant_client
from dotenv import load_dotenv

from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

load_dotenv()

COLLECTION_NAME = "rag_documents"

class RAGEngine:
    def __init__(self):
        self.client = None
        self.vector_store = None
        self.storage_context = None
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
        self.chat_engine = None
        self._is_initialized = False

    def _ensure_initialized(self):
        """Lazy load Hugging Face models and Qdrant client on first request to avoid deployment timeouts."""
        if self._is_initialized:
            return

        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        print("Lazy-loading local BAAI/bge-small-en-v1.5 embedding model...")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash")

        QDRANT_URL = os.getenv("QDRANT_URL")
        QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

        if QDRANT_URL and QDRANT_API_KEY:
            self.client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            QDRANT_PATH = "./qdrant_data"
            os.makedirs(QDRANT_PATH, exist_ok=True)
            self.client = qdrant_client.QdrantClient(path=QDRANT_PATH)

        self.vector_store = QdrantVectorStore(client=self.client, collection_name=COLLECTION_NAME)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self._is_initialized = True
        self._init_chat_engine()

    def _init_chat_engine(self):
        """Loads index and connects conversational memory."""
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
            self.chat_engine = None

    def ingest_file(self, file_path: str):
        """Processes and embeds a single file into Qdrant using local embeddings."""
        self._ensure_initialized()
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context
        )
        self._init_chat_engine()
        return len(documents)

    def query(self, message: str):
        """Queries the engine and returns response along with detailed source metadata."""
        self._ensure_initialized()
        if not self.chat_engine:
            self._init_chat_engine()
            if not self.chat_engine:
                return {
                    "answer": "No documents ingested yet. Please upload a PDF first!",
                    "sources": []
                }

        response = self.chat_engine.chat(message)
        
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