import os
from langchain.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
from transformers import pipeline

# MongoDB Configuration
MONGO_URI = "your_mongodb_uri"
DB_NAME = "diary_db"
COLLECTION_NAME = "diary_entries"

# LLM Configuration
LLM_MODEL = "meta-llama/Meta-Llama-3-8B"

# Embeddings Model
embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")

# MongoDB Connection
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Sentiment Analysis & NER Models
sentiment_analyzer = pipeline("sentiment-analysis")
ner_tagger = pipeline("ner", aggregation_strategy="simple")