import json
import pandas as pd

# Load the JSON file
with open("/Users/pandhari/ai-diary-project/processed_diary.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# List to store extracted data
rows = []

# Process each day's entry
for day in data:
    date = day["date"]
    mood = day["mood"]
    dominant_emotion = day["dominant_emotion"]
    emotion_distribution = day["emotion_distribution"]
    overall_sentiment = day["overall_sentiment"]

    # Process each text entry within the day's records
    for entry in day["entries"]:
        row = {
            "Date": date,
            "Mood": mood,
            "Dominant Emotion": dominant_emotion,
            "Text Entry": entry["text"],
            # Individual emotion scores
            "Anger": entry["emotion"]["anger"],
            "Disgust": entry["emotion"]["disgust"],
            "Fear": entry["emotion"]["fear"],
            "Joy": entry["emotion"]["joy"],
            "Sadness": entry["emotion"]["sadness"],
            "Surprise": entry["emotion"]["surprise"],
            "Neutral": entry["emotion"]["neutral"],
            # Sentiment Scores
            "Negative Sentiment": entry["sentiment"]["Negative"],
            "Positive Sentiment": entry["sentiment"]["Positive"]
        }
        rows.append(row)

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save to CSV
df.to_csv("diary_entries.csv", index=False, encoding="utf-8")

print("CSV file 'diary_entries.csv' has been successfully created!")