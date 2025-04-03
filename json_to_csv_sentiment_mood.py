import json
import pandas as pd


with open("/Users/pandhari/ai-diary-project/processed_diary.json", "r", encoding="utf-8") as file:
    data = json.load(file)

rows = []

for day in data:
    date = day["date"]
    mood = day["mood"]
    dominant_emotion = day["dominant_emotion"]
    emotion_distribution = day["emotion_distribution"]
    overall_sentiment = day["overall_sentiment"]


    for entry in day["entries"]:
        row = {
            "Date": date,
            "Mood": mood,
            "Dominant Emotion": dominant_emotion,
            "Text Entry": entry["text"],
         
            "Anger": entry["emotion"]["anger"],
            "Disgust": entry["emotion"]["disgust"],
            "Fear": entry["emotion"]["fear"],
            "Joy": entry["emotion"]["joy"],
            "Sadness": entry["emotion"]["sadness"],
            "Surprise": entry["emotion"]["surprise"],
            "Neutral": entry["emotion"]["neutral"],
           
            "Negative Sentiment": entry["sentiment"]["Negative"],
            "Positive Sentiment": entry["sentiment"]["Positive"]
        }
        rows.append(row)


df = pd.DataFrame(rows)


df.to_csv("diary_entries.csv", index=False, encoding="utf-8")

print("CSV file 'diary_entries.csv'created!")