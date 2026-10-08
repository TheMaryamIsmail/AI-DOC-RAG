import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
CHROMA_STORE_DIR = os.path.join(BASE_DIR, "chroma_store")
CHROMA_PERSIST_DIR = CHROMA_STORE_DIR  # Dono naam support karne ke liye

# Ensure directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(CHROMA_STORE_DIR, exist_ok=True)

# RAG Hyperparameters & API
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
THE_API = os.getenv("THE_API")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")