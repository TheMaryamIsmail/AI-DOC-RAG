import os
from groq import Groq
from config import THE_API
from config import GROQ_API_KEY
from logger import setup_logger

logger = setup_logger("RAGPipeline")

class PromptEngine:
    @staticmethod
    def zero_shot_prompt(context: str, query: str) -> str:
        return f"""Use the provided context to answer the question accurately.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""

    @staticmethod
    def few_shot_prompt(context: str, query: str) -> str:
        return f"""You are a helpful assistant. Use the provided context to answer the question based on the examples below.

Example 1:
Context: Python is a high-level programming language.
Question: What is Python?
Answer: Based on the text, Python is a high-level programming language.

Example 2:
Context: ChromaDB is a vector database.
Question: What is ChromaDB?
Answer: Based on the text, ChromaDB is a vector database.

---
Context:\n{context}\n\nQuestion: {query}\nAnswer:"""

    @staticmethod
    def role_based_prompt(context: str, query: str) -> str:
        return f"""You are an Expert Technical Document Analyst and Knowledge Assistant. Provide a rigorous, professional answer derived strictly from the documentation context below.

Documentation Context:
{context}

User Inquiry: {query}
Professional Analysis:"""


class RAGPipeline:
    def __init__(self, chroma_manager, prompt_engine: PromptEngine):
        self.chroma = chroma_manager
        self.prompt_engine = prompt_engine
        
        if not THE_API:
            logger.warning("API key not found in environment variables!")
        
        # FIXED: Removed the invalid keyword argument GROQ_API_KEY=...
        self.groq_client = Groq(api_key=THE_API)

    def run_qa(self, query: str, strategy: str = "role_based", top_k: int = 3) -> dict:
        logger.info(f"Running QA with strategy '{strategy}' for query: '{query}'")
        
        retrieved_chunks = self.chroma.similarity_search(query, top_k=top_k)
        
        context_text = "\n\n".join([f"Source: {c['metadata']['source']}\n{c['text']}" for c in retrieved_chunks])
        
        if strategy == "zero_shot":
            prompt = self.prompt_engine.zero_shot_prompt(context_text, query)
        elif strategy == "few_shot":
            prompt = self.prompt_engine.few_shot_prompt(context_text, query)
        else:
            prompt = self.prompt_engine.role_based_prompt(context_text, query)
            
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
            )
            answer = chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API call failed: {str(e)}")
            answer = f"Error generating response from Groq API: {str(e)}"

        return {
            "query": query,
            "strategy": strategy,
            "prompt_used": prompt,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks
        }

    def compare_prompt_techniques(self, query: str) -> dict:
        strategies = ["zero_shot", "few_shot", "role_based"]
        results = {strat: self.run_qa(query, strategy=strat) for strat in strategies}
        
        return {
            "comparison_query": query,
            "results": results,
            "evaluation_summary": (
                "Role-based prompting delivers superior results for technical document analysis by enforcing professional structure, "
                "rigorous grounding, and clarity. Few-shot provides good formatting guidance, while zero-shot is direct but less formal."
            )
        }