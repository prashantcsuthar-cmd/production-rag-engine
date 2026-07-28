import os
from typing import Any, List
import qdrant_client
from dotenv import load_dotenv

from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

from fastembed import TextEmbedding
from pydantic import PrivateAttr

load_dotenv()

COLLECTION_NAME = "rag_documents"


class CustomFastEmbed(BaseEmbedding):
    """Custom FastEmbed wrapper to eliminate llama-index package conflicts & asyncio issues."""
    _model: TextEmbedding = PrivateAttr()

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model = TextEmbedding(model_name=model_name)

    @classmethod
    def class_name(cls) -> str:
        return "CustomFastEmbed"

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = list(self._model.query_embed([query]))
        return embeddings[0].tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = list(self._model.passage_embed([text]))
        return embeddings[0].tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self._model.passage_embed(texts))
        return [e.tolist() for e in embeddings]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


class RAGEngine:
    def __init__(self):
        # Local ONNX fast embedding without external package conflicts or API quotas
        Settings.embed_model = CustomFastEmbed(model_name="BAAI/bge-small-en-v1.5")
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

        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
        self.chat_engine = None
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