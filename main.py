import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from config import DOCUMENTS_DIR
from ingest import DocumentIngester
from chroma_db import ChromaDBManager
from rag_pipeline import PromptEngine, RAGPipeline

app = FastAPI(title="AI Document QA & RAG Assistant", version="2.0.0")

templates = Jinja2Templates(directory="templates")

chroma_mgr = ChromaDBManager()
prompt_engine = PromptEngine()
rag_pipeline = RAGPipeline(chroma_mgr, prompt_engine)

class QueryRequest(BaseModel):
    query: str
    strategy: str = "role_based"
    top_k: int = 3

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Renders the Jinja2 frontend template using updated Starlette/FastAPI signature."""
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads, ingests, chunks, and indexes documents into ChromaDB."""
    if not file.filename.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and Markdown files are supported.")
        
    file_path = os.path.join(DOCUMENTS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    chunks = DocumentIngester.get_all_chunks(DOCUMENTS_DIR)
    chroma_mgr.add_documents(chunks)
    
    return {
        "status": "success",
        "filename": file.filename,
        "total_chunks_indexed": len(chunks)
    }

@app.post("/api/ask")
async def ask_question(payload: QueryRequest):
    return rag_pipeline.run_qa(query=payload.query, strategy=payload.strategy, top_k=payload.top_k)

@app.post("/api/compare")
async def compare_prompts(payload: QueryRequest):
    return rag_pipeline.compare_prompt_techniques(query=payload.query)