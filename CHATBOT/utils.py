from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def format_date(date_str):
    """Convert various date formats to YYYY-MM-DD."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return date_str

def embed_text(text):
    """Generate embeddings for text."""
    return embedding_model.embed_query(text)