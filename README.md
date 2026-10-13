## 📄 AI Document RAG & Prompt Engineering Module
A robust,production-ready Retrieval-Augmented Generation (RAG) backend and analysis module built with FastAPI, ChromaDB, and the Google Gemini API. This application enables efficient document ingestion, semantic vector search, and comparative multi-strategy prompt engineering.
## 🚀 Key FeaturesAdvanced Document Chunking & Ingestion:
Processes PDFs, text documents, and markdown files into optimized vector chunks with custom chunk size and overlap configuration.
## Persistent Vector Search:
Leverages ChromaDB for fast, local, and persistent semantic similarity search using sentence-transformer embeddings (all-MiniLM-L6-v2).
## Google Gemini Integration: 
Powered by the official Google GenAI SDK (google-genai) and cutting-edge Gemini models.
## Multi-Strategy Prompt Engineering: 
Built-in evaluation and comparison engine supporting three distinct prompting frameworks
## Zero-Shot Prompting:
Direct context-to-answer generation.Few-Shot Prompting: Formatted examples to guide output style and structure.
## Role-Based Prompting:
Assigns expert technical personas to enforce rigorous, structured document analysis.
## FastAPI Backend:
Fully asynchronous REST endpoints supporting document uploads, targeted QA queries, and side-by-side prompt performance comparisons.
## 🛠️ Tech StackBackend Framework:
FastAPI, UvicornLLM Provider: Google Gemini API (google-genai)Vector Database: ChromaDBEmbeddings: Sentence-Transformers (all-MiniLM-L6-v2)Environment Management: python-dotenv📁 Project StructurePlaintextai-doc-rag-module/
## │
## ├── documents/            # Directory for source documents (PDF, TXT)
## ├── chroma_store/         # Persistent ChromaDB vector storage
## ├── main.py               # FastAPI application entry point
## ├── rag_pipeline.py       # RAG core logic, prompt engine, and LLM integration
## ├── config.py             # Configuration and environment loader
## ├── logger.py             # Custom logging utility
## ├── requirements.txt      # Project dependencies
## └── .env                  # Environment variables (API keys & server config)
## ⚙️ Getting Started & Installation
1.Clone the RepositoryBashgit clone https://github.com/your-username/ai-doc-rag-module.git
cd ai-doc-rag-module
2. Create and Activate a Virtual EnvironmentBashpython -m venv .aidoc
# On Windows:
.aidoc\Scripts\activate
# On Mac/Linux:
source .aidoc/bin/activate
3. Install DependenciesBashpip install -r requirements.txt

## 🚀 Running the Application
Start the FastAPI server locally with live-reload enabled:Bashuvicorn main:app --reload
## 🔍 API Endpoints Overview 
EndpointMethodDescription/api/uploadPOSTUploads and indexes documents into ChromaDB./api/askPOST
Performs semantic retrieval and answers user queries using a selected prompting strategy./api/comparePOSTCompares responses across Zero-Shot, Few-Shot, and Role-Based prompting techniques simultaneously.
## 📝 License
This project is open-source and available under the MIT License.
