import json
import chromadb
import uuid
from utils import format_date, embed_text
from ner_activity_extraction import extract_named_entities, extract_activities

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_diary_db")
collection = chroma_client.get_or_create_collection(name="diary_entries")

def store_embeddings(json_file):
    """Store diary entries in ChromaDB with unique IDs, including NER and activities."""
    
    with open(json_file, "r", encoding="utf-8") as f:
        diary_data = json.load(f)

    for entry in diary_data:
        formatted_date = format_date(entry["date"])

        for chunk in entry["entries"]:
            text = chunk["text"]
            embedding = embed_text(text)
            named_entities = extract_named_entities(text)
            activities = extract_activities(text)

            metadata = {
                "date": formatted_date,
                "mood": entry["mood"],
                "dominant_emotion": entry["dominant_emotion"],
                "named_entities": ", ".join([f"{ent[0]} ({ent[1]})" for ent in named_entities]) if named_entities else "",  # Convert list to string
                "activities": ", ".join(activities) if activities else ""  # Convert list to string
            }

            unique_id = str(uuid.uuid4())

            collection.add(
                ids=[unique_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )

    print("✅ Embeddings stored in ChromaDB with NER and activities!")

if __name__ == "__main__":
    store_embeddings("/Users/pandhari/ai-diary-project/processed_diary.json")