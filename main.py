import tempfile
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag_engine import RAGEngine

app = FastAPI(
    title="Production RAG Engine API",
    description="Enterprise-grade Retrieval-Augmented Generation API powered by Gemini & Qdrant",
    version="1.0.0"
)

rag_system = RAGEngine()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "online", "message": "Production RAG Engine API is running"}


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF, embeds it into Qdrant, and immediately cleans up local disk space."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Create a temporary file location that auto-deletes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        temp_path = tmp.name

    try:
        # Ingest into vector store
        num_docs = rag_system.ingest_file(temp_path)
        return {
            "status": "success",
            "filename": file.filename,
            "message": f"Successfully ingested '{file.filename}' ({num_docs} pages) into Qdrant."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        # Guarantee local file deletion to save storage
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/chat")
def chat(request: ChatRequest):
    """Processes query with context and returns response with citations."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = rag_system.query(request.message)
        return {
            "status": "success",
            "query": request.message,
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")