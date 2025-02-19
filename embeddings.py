from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

# Get Hugging Face API Key from environment variables
HF_API_KEY = os.getenv("HF_API_KEY")


# Load the Hugging Face embedding model
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def generate_embedding(text):
    """Generate embeddings using Hugging Face's sentence-transformers."""
    embedding = embedding_model.encode(text, normalize_embeddings=True)
    return embedding  # Already a NumPy array

def embed_chunks(processed_entries):
    """Embed each chunk and add embedding to the data."""
    for entry in processed_entries:
        entry["embedding"] = generate_embedding(entry["text"]).tolist()
    return processed_entries
