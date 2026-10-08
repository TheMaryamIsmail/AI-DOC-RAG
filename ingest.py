import os
import logging
from typing import List, Dict, Any
from pypdf import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENTS_DIR

logger = logging.getLogger(__name__)

class DocumentIngester:
    @staticmethod
    def load_document(file_path: str) -> str:
        """Reads content from a single file based on its extension."""
        text = ""
        try:
            if file_path.endswith((".txt", ".md")):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            elif file_path.endswith(".pdf"):
                reader = PdfReader(file_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
        return text

    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Splits text into overlapping chunks to preserve semantic boundaries."""
        if not text:
            return []
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
            
        return chunks

    @classmethod
    def get_all_chunks(cls, directory: str = DOCUMENTS_DIR) -> List[Dict[str, Any]]:
        """Scans the directory and breaks all documents into indexed chunks."""
        processed_chunks = []
        if not os.path.exists(directory):
            return processed_chunks
            
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if not os.path.isfile(file_path):
                continue
                
            raw_text = cls.load_document(file_path)
            if not raw_text.strip():
                continue
                
            chunks = cls.chunk_text(raw_text)
            for idx, chunk in enumerate(chunks):
                processed_chunks.append({
                    "id": f"{filename}_chunk_{idx}",
                    "text": chunk,
                    "metadata": {"source": filename, "chunk_index": idx}
                })
                
        return processed_chunks