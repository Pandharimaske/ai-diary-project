import chromadb
from langchain_ollama import OllamaLLM
from utils import embed_text
import os

# Set up environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load LLM (LLaMA 3.2)
llm = OllamaLLM(model="llama3.2")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_diary_db")
collection = chroma_client.get_collection(name="diary_entries")

def retrieve_entries(query, mood_filter=None, date_filter=None, ner_filter=None, activity_filter=None, top_k=3):
    """Retrieve diary entries based on query, mood, date, named entities, and activities."""
    
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    final_results = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        if mood_filter and mood_filter.lower() not in meta["mood"].lower():
            continue
        if date_filter and date_filter != meta["date"]:
            continue
        if ner_filter and not any(ner_filter.lower() in entity[0].lower() for entity in meta.get("named_entities", [])):
            continue
        if activity_filter and not any(activity_filter.lower() in activity.lower() for activity in meta.get("activities", [])):
            continue
        
        final_results.append({"text": doc, "date": meta["date"], "mood": meta["mood"], "named_entities": meta.get("named_entities", []), "activities": meta.get("activities", [])})

    return final_results

if __name__ == "__main__":
    print("💬 AI Diary Chatbot: Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break
        response = retrieve_entries(query)
        print("\n🤖 AI Response:", response, "\n")