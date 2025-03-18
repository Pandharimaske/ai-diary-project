import json
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import uuid

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_diary_db")
collection = chroma_client.get_or_create_collection(name="diary_entries")

def store_embeddings(json_file):
    """Store diary entries in ChromaDB with unique IDs."""
    
    with open(json_file, "r", encoding="utf-8") as f:
        diary_data = json.load(f)

    for entry in diary_data:
        for chunk in entry["entries"]:
            text = chunk["text"]
            embedding = embedding_model.embed_query(text)
            metadata = {
                "date": entry["date"],
                "mood": entry["mood"],
                "dominant_emotion": entry["dominant_emotion"]
            }

            # Generate unique ID for each chunk
            unique_id = str(uuid.uuid4())

            # Store in ChromaDB
            collection.add(
                ids=[unique_id],  # Use UUID instead of hash-based ID
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )

    print("✅ Embeddings stored in ChromaDB!")

# Example usage:
if __name__ == "__main__":
    store_embeddings("/Users/pandhari/ai-diary-project/processed_diary.json")