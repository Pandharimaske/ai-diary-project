import pandas as pd
import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

def chunk_text(text, chunk_size=200, chunk_overlap=50):
    """Split text into structured chunks using LangChain."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def get_emotion_scores(text):
    """Get emotion distribution for a given text chunk."""
    if not text.strip():
        return {label: 0.0 for label in EMOTION_LABELS}
    
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
    
    return {label: probs[i] for i, label in enumerate(EMOTION_LABELS)}

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
    """Process a diary entry into chunks, analyze emotions & sentiment, and compute overall mood."""
    chunks = chunk_text(entry)
    processed_data = []
    
    total_emotions = {label: 0.0 for label in EMOTION_LABELS}
    total_sentiments = {label: 0.0 for label in SENTIMENT_LABELS}
    
    for chunk in chunks:
        emotions = get_emotion_scores(chunk)
        sentiment = get_sentiment_scores(chunk)

        # Sum up emotion and sentiment scores across chunks
        for label in EMOTION_LABELS:
            total_emotions[label] += emotions[label]

        for label in SENTIMENT_LABELS:
            total_sentiments[label] += sentiment[label]

        processed_data.append({
            "text": chunk,
            "emotion": emotions,
            "sentiment": sentiment,
            "date": date
        })

    # Compute overall dominant emotion
    dominant_emotion = max(total_emotions, key=total_emotions.get)
    
    # Compute sentiment balance
    mood_score = total_sentiments["Positive"] - total_sentiments["Negative"]  

    # Define mood categories based on dominant emotion & sentiment
    if dominant_emotion in ["joy", "surprise"]:
        overall_mood = "Happy 😀"
    elif dominant_emotion in ["anger", "disgust", "fear"]:
        overall_mood = "Anxious 😨"
    elif dominant_emotion == "sadness":
        overall_mood = "Sad 😢"
    else:
        overall_mood = "Neutral 😐"

    # Adjust based on sentiment
    if mood_score > 0.5:
        overall_mood += " (Positive Outlook)"
    elif mood_score < -0.5:
        overall_mood += " (Negative Outlook)"
    
    return {
        "date": date,
        "mood": overall_mood,
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": total_emotions,
        "overall_sentiment": total_sentiments,
        "entries": processed_data  # All processed chunks
    }

def process_csv(file_path, output_json_path):
    """Process diary CSV, analyze emotions & mood, and save structured data to JSON."""
    df = pd.read_csv(file_path)
    
    # Ensure necessary columns exist
    if "Date" not in df.columns or "Entry" not in df.columns:
        raise ValueError("CSV file must contain 'Date' and 'Entry' columns")

    processed_entries = []
    
    for _, row in df.iterrows():
        date = row["Date"]
        entry = row["Entry"]
        
        processed_entries.append(process_diary_entry(entry, date))

    # Save as JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(processed_entries, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Processed diary entries saved to {output_json_path}")

# Example usage:
if __name__ == "__main__":
    process_csv("/Users/pandhari/ai-diary-project/Data/diary_dataset.csv", "processed_diary.json")