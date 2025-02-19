from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import numpy as np

# Load embeddings into FAISS for efficient search
def load_faiss(entries):
    """Create FAISS index for similarity search."""
    embeddings = [np.array(e["embedding"]) for e in entries]
    texts = [e["text"] for e in entries]
    faiss_index = FAISS.from_embeddings(texts, embeddings)
    return faiss_index

def query_faiss(faiss_index, query_text):
    """Retrieve similar entries using FAISS."""
    results = faiss_index.similarity_search(query_text, k=5)
    return [res.page_content for res in results]