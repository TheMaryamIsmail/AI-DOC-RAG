from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

class EmbeddingManager:
    _instance = None

    def __new__(cls, model_name: str = EMBEDDING_MODEL_NAME):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance.model = SentenceTransformer(model_name)
        return cls._instance

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        return embeddings.tolist()

    def generate_query_embedding(self, query: str) -> list[float]:
        embedding = self.model.encode(query, convert_to_tensor=False)
        return embedding.tolist()