from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from retriever import load_faiss, query_faiss
from database import query_by_date
import datetime


llm = OllamaLLM(model="llama3.2")

def retrieve_entries(query):
    """Retrieve relevant diary entries using RAG pipeline."""
    if "date" in query:
        try:
            date = query.split("date")[-1].strip()
            datetime.datetime.strptime(date, "%Y-%m-%d")  # Validate date format
            return query_by_date(date)
        except:
            return "❌ Invalid date format. Use YYYY-MM-DD."
    
    faiss_index = load_faiss()
    return query_faiss(faiss_index, query)

def chat_with_diary(query):
    """Chatbot interface to query diary entries."""
    relevant_entries = retrieve_entries(query)
    context = " ".join(relevant_entries) if isinstance(relevant_entries, list) else relevant_entries
    
    response = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=context,
        chain_type="stuff"
    ).run(query)
    
    return response