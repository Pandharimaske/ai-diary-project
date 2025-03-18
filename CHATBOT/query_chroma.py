import chromadb
import subprocess
import json
from langchain_huggingface import HuggingFaceEmbeddings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_diary_db")
collection = chroma_client.get_collection(name="diary_entries")

def retrieve_entries(query, mood_filter=None, date_filter=None, top_k=3):
    """Retrieve similar diary entries based on a query with optional mood/date filters."""
    
    query_embedding = embedding_model.embed_query(query)

    # Base search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    filtered_results = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        if mood_filter and mood_filter.lower() not in meta["mood"].lower():
            continue  # Skip if mood filter doesn't match
        if date_filter and date_filter != meta["date"]:
            continue  # Skip if date filter doesn't match
        
        filtered_results.append({"text": doc, "date": meta["date"], "mood": meta["mood"]})

    return filtered_results

def generate_intelligent_response(query, mood_filter=None, date_filter=None, top_k=3):
    """Retrieve relevant diary entries and generate a detailed response using LLaMA 3.2 via Ollama."""
    
    retrieved_entries = retrieve_entries(query, mood_filter, date_filter, top_k)

    if not retrieved_entries:
        return "I couldn't find relevant diary entries for your query."

    # Format retrieved data into context
    context = "\n".join([f"Date: {entry['date']}, Mood: {entry['mood']}\nEntry: {entry['text']}" for entry in retrieved_entries])

    # Construct prompt
    prompt = f"""
    You are an AI assistant analyzing my diary. Based on the following past entries, answer the question in a meaningful way.
    
    ----
    Past Diary Entries:
    {context}
    ----
    Question: {query}
    
    Please provide a thoughtful response based on my past experiences.
    """

    # Call Ollama LLaMA 3.2 locally
    response = subprocess.run(
        ["ollama", "run", "llama3", prompt], 
        capture_output=True, text=True
    )

    return response.stdout.strip()

# Example usage:
if __name__ == "__main__":
    query = "How have I felt in the past week?"
    response = generate_intelligent_response(query)
    print("🤖 AI Response:", response)