import os
import shutil
import qdrant_client
from dotenv import load_dotenv

from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

load_dotenv()

# Initialize Models
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")
# Change this line:
Settings.llm = GoogleGenAI(model="gemini-3.6-flash")

COLLECTION_NAME = "rag_documents"
QDRANT_PATH = "./qdrant_data"
DATA_DIR = "./data"


class RAGEngine:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
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
        """Processes and embeds a single file into Qdrant."""
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