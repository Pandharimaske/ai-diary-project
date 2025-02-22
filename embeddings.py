import os
import torch
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.schema import Document

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["diary_database"]
collection = db["diary_entries"]

# Detect device (GPU, MPS, or CPU)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize LangChain embedding model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# MongoDB Atlas Vector Store
vector_store = MongoDBAtlas(
    mongo_uri=MONGO_URI,
    db_name="diary_database",
    collection_name="diary_entries",
    embedding=embedding_model
)

def store_entries(processed_entries):
    """
    Store diary entries with embeddings in MongoDB using LangChain.
    """
    docs = []
    texts = [entry["text"] for entry in processed_entries]
    embeddings = embedding_model.embed_documents(texts)

    for entry, embedding in zip(processed_entries, embeddings):
        doc = Document(
            page_content=entry["text"],
            metadata={
                "date": entry["date"],
                "sentiment": entry["sentiment"],
                "emotions": entry["emotions"],
                "embedding": embedding
            }
        )
        docs.append(doc)

    # Add documents to MongoDB
    vector_store.add_documents(docs)
    print("Data successfully inserted into MongoDB!")

# Store in MongoDB
store_entries(processed_entries)

