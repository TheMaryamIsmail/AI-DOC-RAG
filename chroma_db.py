import os
import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_PERSIST_DIR
from logger import setup_logger

logger = setup_logger("ChromaDBManager")

class ChromaDBManager:
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR, collection_name: str = "document_knowledge_base"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        logger.info(f"Initializing ChromaDB persistent client at '{persist_directory}'")
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, chunks: list[dict]):
        """Embeds and inserts processed text chunks into ChromaDB."""
        if not chunks:
            return
            
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        embeddings = self.embedding_model.encode(texts).tolist()
        
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Successfully stored {len(chunks)} chunks into Chroma collection.")

    def similarity_search(self, query: str, top_k: int = 3) -> list[dict]:
        """Queries the vector database for top_k semantically relevant chunks with score metrics."""
        logger.info(f"Executing similarity search for query: '{query}' (top_k={top_k})")
        
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        formatted_results = []
        if results and results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0] if 'distances' in results and results['distances'] else [0.0] * len(docs)
            
            for doc, meta, dist in zip(docs, metadatas, distances):
                similarity_score = round(max(0.0, 1.0 - float(dist)), 4)
                formatted_results.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": round(float(dist), 4),
                    "similarity_score": similarity_score
                })
                
        return formatted_results