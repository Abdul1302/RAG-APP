import os
from dotenv import load_dotenv

load_dotenv()

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama3-8b-8192"

# Pinecone
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag-book-index"
PINECONE_DIMENSION  = 384

# Chunking
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 4

# PDF
PDF_PATH = "data/machine-learning-yearning.pdf"