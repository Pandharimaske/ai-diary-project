import chromadb
from langchain_ollama import OllamaLLM
import json
from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load LLaMA 3.2 model via Ollama
llm = OllamaLLM(model="mistral")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_diary_db")
collection = chroma_client.get_collection(name="diary_entries")

def retrieve_entries(query, mood_filter=None, date_filter=None, top_k=3):
    """Retrieve diary entries based on a query with optional mood/date filters."""
    
    query_embedding = embedding_model.embed_query(query)
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
        
        final_results.append({"text": doc, "date": meta["date"], "mood": meta["mood"]})

    return final_results

def generate_intelligent_response(query, mood_filter=None, date_filter=None, top_k=3):
    """Retrieve diary entries and generate a simple, effective response."""
    
    retrieved_entries = retrieve_entries(query, mood_filter, date_filter, top_k)

    if not retrieved_entries:
        return "I couldn't find relevant diary entries for your query."

    context = "\n".join([f"Date: {entry['date']}, Mood: {entry['mood']}\nEntry: {entry['text']}" for entry in retrieved_entries])

    prompt = f"""
    You are my personal AI diary assistant. Answer clearly and concisely based on my diary.

    ----
    Diary Entries:
    {context}
    ----
    User: {query}

    Provide a helpful response.
    """

    return llm.invoke(prompt).strip()

if __name__ == "__main__":
    print("💬 AI Diary Chatbot: Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break
        response = generate_intelligent_response(query)
        print("\n🤖 AI Response:", response, "\n")