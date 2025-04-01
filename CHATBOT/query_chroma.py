import os
import torch
import chromadb
from langchain_ollama import OllamaLLM
from utils import embed_text

# Set up model selection
MODEL_NAME = "llama3.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model & Tokenizer
print(f"🔄 Loading {MODEL_NAME} on {DEVICE}...")
model = OllamaLLM(model="llama3.2")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_diary_db")
collection = chroma_client.get_collection(name="diary_entries")

def retrieve_entries(query, top_k=5):
    """Retrieve diary entries and generate AI response."""
    
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_texts = [
        f"Entry: {doc}\nMood: {meta.get('mood', 'Unknown')}\nDominant Emotion: {meta.get('dominant_emotion', 'Unknown')}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    # Prepare prompt for the model
    prompt = "Based on the diary entries below, answer the query. Consider the mood and dominant emotion for context:\n"
    prompt += "\n\n".join(retrieved_texts)
    prompt += f"\n\nUser Query: {query}\nAI Response:"

    response = model.invoke(prompt)

    return response

if __name__ == "__main__":
    print("💬 AI Diary Chatbot: Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break
        response = retrieve_entries(query)
        print("\n🤖 AI Response:", response, "\n")