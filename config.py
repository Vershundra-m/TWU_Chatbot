import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Data Source
INFORMATICS_URL = "https://twu.edu/informatics/graduate-program/"

# Database Settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "twu-ai-chatbot"
EMBEDDING_MODEL = "multilingual-e5-large"

# Processing Settings
CHUNK_SIZE = 120 
CHUNK_OVERLAP = 15
TOP_K_RESULTS = 3

# API Settings

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

VERTEX_API_KEY = os.getenv("VERTEX_API_KEY")
