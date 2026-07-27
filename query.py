import os
import qdrant_client
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Load environment variables (.env)
load_dotenv()


def main():
    print("⚡ Step 1: Connecting to Local Qdrant Storage...")
    client = qdrant_client.QdrantClient(path="./qdrant_data")

    print("🧠 Step 2: Setting up Gemini Models...")
    # Matching embedding model used during ingestion
    Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")
    # Active Gemini LLM model
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash")

    print("🔍 Step 3: Loading Vector Index...")
    vector_store = QdrantVectorStore(client=client, collection_name="rag_documents")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )

    # Create RAG query engine
    query_engine = index.as_query_engine(similarity_top_k=3)

    print("\n✅ RAG Engine Ready! Type your question below (or 'exit' to quit).\n")
    print("-" * 60)

    while True:
        user_query = input("\n❓ Ask a question: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ["exit", "quit", "q"]:
            print("👋 Bye!")
            break

        print("🔎 Searching Qdrant & Generating Answer...")
        response = query_engine.query(user_query)

        print("\n🤖 Answer:")
        print(response.response)
        print("-" * 60)


if __name__ == "__main__":
    main()