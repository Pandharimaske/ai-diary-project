import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy
import json

# Load spaCy model for sentence segmentation
nlp = spacy.load("en_core_web_sm")

# Load Emotion Analysis Model
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
emotion_model.eval()

# Load Sentiment Analysis Model
sentiment_model_name = "siebert/sentiment-roberta-large-english"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_model.eval()

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model.to(device)
sentiment_model.to(device)

# Define labels
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
SENTIMENT_LABELS = ["Negative", "Positive"]

def chunk_text(text):
    """Split text into meaningful sentence chunks."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def get_emotion_scores(text):
    """Get emotion distribution for a given text chunk."""
    if not text.strip():
        return {label: 0.0 for label in EMOTION_LABELS}
    
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
    
    return {label: probs[i] for i, label in enumerate(EMOTION_LABELS)}

# def extract_ner(text):
#     """Extract named entities using spaCy."""
#     doc = nlp(text)
#     return [(ent.text, ent.label_) for ent in doc.ents]

def get_sentiment_scores(text):
    """Get sentiment scores."""
    if not isinstance(text, str) or text.strip() == "":
        return {label: 0.0 for label in SENTIMENT_LABELS}

    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
    return {label: probs[i] for i, label in enumerate(SENTIMENT_LABELS)}

def process_diary_entry(entry, date):
    """Process a single diary entry into multiple chunks with emotions & sentiment."""
    chunks = chunk_text(entry)
    processed_data = []
    
    for chunk in chunks:
        emotions = get_emotion_scores(chunk)
        sentiment = get_sentiment_scores(chunk)
        # entities = extract_ner(chunk)
        
        processed_data.append({
            "text": chunk,
            "emotion": emotions,
            "sentiment": sentiment,
            # "named_entities": entities,
            "date": date
        })
    
    return processed_data

def process_csv(file_path):
    """Process diary CSV and return structured data in JSON format."""
    df = pd.read_csv(file_path)
    
    # Ensure necessary columns exist
    if "Date" not in df.columns or "Entry" not in df.columns:
        raise ValueError("CSV file must contain 'Date' and 'Entry' columns")

    processed_entries = []
    
    for _, row in df.iterrows():
        date = row["Date"]
        entry = row["Entry"]
        
        processed_entries.extend(process_diary_entry(entry, date))  # No need to append `date` inside loop
    
    return json.dumps(processed_entries, indent=4)  # Return JSON data