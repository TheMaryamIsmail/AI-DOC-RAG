import os
from config import THE_API, CHROMA_PERSIST_DIR, DOCUMENTS_DIR
from chroma_db import ChromaDBManager
from rag_pipeline import PromptEngine, RAGPipeline
from logger import setup_logger

logger = setup_logger("PipelineValidator")

def validate_environment():
    logger.info("--- 1. Validating Environment & Config ---")
    assert THE_API, "❌ ERROR: THE_API or GROQ_API_KEY is missing in environment variables!"
    logger.info(f"✅ API Key found. Length: {len(THE_API)}")
    
    assert os.path.exists(DOCUMENTS_DIR), f"❌ ERROR: Documents directory missing at {DOCUMENTS_DIR}"
    logger.info(f"✅ Documents directory verified: {DOCUMENTS_DIR}")
    
    assert os.path.exists(CHROMA_PERSIST_DIR), f"❌ ERROR: Chroma store directory missing at {CHROMA_PERSIST_DIR}"
    logger.info(f"✅ Chroma store directory verified: {CHROMA_PERSIST_DIR}")

def validate_chroma_and_embeddings():
    logger.info("\n--- 2. Validating ChromaDB & Embeddings ---")
    try:
        chroma_mgr = ChromaDBManager()
        # Test sample embedding insertion and search
        test_chunk = [{
            "id": "validator_test_chunk_0",
            "text": "FastAPI is a modern, fast web framework for building APIs with Python.",
            "metadata": {"source": "validator_test.txt", "chunk_index": 0}
        }]
        chroma_mgr.add_documents(test_chunk)
        logger.info("✅ Successfully added test document to ChromaDB.")

        results = chroma_mgr.similarity_search("What is FastAPI?", top_k=1)
        assert len(results) > 0, "❌ ERROR: Similarity search returned no results."
        logger.info(f"✅ Similarity search passed! Retrieved text: '{results[0]['text'][:40]}...'")
    except Exception as e:
        logger.error(f"❌ ChromaDB / Embedding validation failed: {str(e)}")
        raise e

def validate_rag_pipeline():
    logger.info("\n--- 3. Validating End-to-End RAG Pipeline & Groq API ---")
    try:
        chroma_mgr = ChromaDBManager()
        prompt_engine = PromptEngine()
        pipeline = RAGPipeline(chroma_mgr, prompt_engine)

        response = pipeline.run_qa(query="What is FastAPI?", strategy="role_based", top_k=1)
        assert response["answer"], "❌ ERROR: RAG pipeline generated an empty answer."
        logger.info(f"✅ RAG Pipeline test passed! Generated Answer:\n{response['answer']}")
    except Exception as e:
        logger.error(f"❌ RAG Pipeline validation failed: {str(e)}")
        raise e

if __name__ == "__main__":
    logger.info("Starting System Health & Component Validator...")
    validate_environment()
    validate_chroma_and_embeddings()
    validate_rag_pipeline()
    logger.info("\n🎉 All pipeline components validated successfully!")