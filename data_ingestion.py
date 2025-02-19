import spacy
import pandas as pd
from transformers import pipeline

# Load NLP models
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion")

# Define individual emotion categories
EMOTION_CATEGORIES = ["happy", "sad", "surprised", "angry", "disgusted", "fearful", "neutral"]

def chunk_text(text):
    """Split text into meaningful sentence chunks."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def analyze_sentiment(text):
    """Perform sentiment analysis and return detected emotion."""
    result = sentiment_analyzer(text)[0]
    return result["label"].lower()  # Extract detected emotion

def extract_ner(text):
    """Extract named entities using spaCy."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def process_diary_entry(entry):
    """Process diary entry: chunk text, analyze sentiment, and extract NER."""
    chunks = chunk_text(entry)
    processed_data = []
    
    # Initialize emotion counters
    emotion_count = {e: 0 for e in EMOTION_CATEGORIES}

    for chunk in chunks:
        emotion = analyze_sentiment(chunk)
        ner_tags = extract_ner(chunk)

        # Update emotion counters
        if emotion in emotion_count:
            emotion_count[emotion] += 1

        processed_data.append({
            "text": chunk,
            "emotion": emotion,
            "ner": ner_tags
        })
    
    return processed_data, emotion_count

def process_csv(file_path):
    """Read CSV, process each entry, and return structured data."""
    df = pd.read_csv(file_path)
    processed_entries = []
    
    for _, row in df.iterrows():
        date = row["Date"]
        entry = row["Entry"]
        processed_chunks, emotion_count = process_diary_entry(entry)
        
        for chunk in processed_chunks:
            processed_entries.append({
                "date": date,
                **chunk
            })
        
        # Store overall emotion counts per entry
        processed_entries.append({
            "date": date,
            "overall_emotion_count": emotion_count
        })
    
    return processed_entries