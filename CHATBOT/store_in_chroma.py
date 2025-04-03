import json
import chromadb
import hashlib
from utils import format_date, embed_text


chroma_client = chromadb.PersistentClient(path="./chroma_diary_db")
collection = chroma_client.get_or_create_collection(name="diary_entries")

def generate_unique_id(text):
   
    return hashlib.md5(text.encode()).hexdigest()

def store_embeddings(json_file):
    
    
    with open(json_file, "r", encoding="utf-8") as f:
        diary_data = json.load(f)

    stored_count = 0

    for entry in diary_data:
        formatted_date = format_date(entry["date"])

        for chunk in entry["entries"]:

            text = f"[{formatted_date}] {chunk['text']}"
            embedding = embed_text(text)

            metadata = {
                "mood": entry["mood"], 
                "dominant_emotion": entry["dominant_emotion"]
            }

            unique_id = generate_unique_id(text)

            
            collection.add(
                ids=[unique_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )

            stored_count += 1
            print(f"Stored chunk for {formatted_date}: {text[:50]}...")

    print(f"Finished storing {stored_count} chunks in ChromaDB!")

if __name__ == "__main__":
    store_embeddings("/Users/pandhari/ai-diary-project/processed_diary.json")