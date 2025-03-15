from CHATBOT.data_ingestion import process_csv
from CHATBOT.vector_db import store_entries
# from chatbot import chat_with_diary

if __name__ == "__main__":
    # Load and process diary entries
    entries = process_csv("/Users/pandhari/ai-diary-project/Data/diary_dataset.csv")
    
    # Generate embeddings + Store in MongoDB
    store_entries(entries)
    # Chatbot Interaction
    while True:
        query = input("📝 Ask something about your diary: ")
        if query.lower() == "exit":
            break
        print("🤖 Diary says:", chat_with_diary(query))
