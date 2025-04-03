import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import pandas as pd

def plot_mood_trends(json_file):
    """Visualize mood trends over time with multiple enhanced graphs."""
    
    # Load processed diary data
    with open(json_file, "r", encoding="utf-8") as f:
        diary_data = json.load(f)
    
    # Extract dates and mood scores
    dates = []
    mood_scores = []
    mood_counts = {}

    mood_to_score = {
        "Happy 😀": 2,
        "Neutral 😐": 1,
        "Anxious 😨": -1,
        "Sad 😢": -2
    }

    for entry in diary_data:
        date = entry["date"]
        mood_base = entry["mood"].split(" (")[0]  # Remove extra labels

        dates.append(date)
        mood_scores.append(mood_to_score.get(mood_base, 0))

        # Count occurrences of each mood
        mood_counts[mood_base] = mood_counts.get(mood_base, 0) + 1

    # Convert dates to pandas datetime format
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "Mood Score": mood_scores})
    df.sort_values("Date", inplace=True)

    # Compute a rolling average to smooth out trends
    df["Rolling_Avg"] = df["Mood Score"].rolling(window=3, min_periods=1).mean()

    # --- 1️⃣ Interactive Time-Series Line Chart (Plotly) ---
    fig = px.line(df, x="Date", y="Mood Score", title="Interactive Mood Trends Over Time", markers=True)
    fig.add_scatter(x=df["Date"], y=df["Rolling_Avg"], mode="lines", name="Rolling Avg", line=dict(dash="dash", color="red"))
    fig.show()

    # --- 2️⃣ Seaborn Heatmap: Mood Intensity Over Time ---
    plt.figure(figsize=(12, 5))
    mood_heatmap_data = df.pivot_table(index=df["Date"].dt.date, values="Mood Score", aggfunc="mean")
    sns.heatmap(mood_heatmap_data, cmap="coolwarm", annot=True, linewidths=0.5, linecolor="black")
    plt.title("Mood Intensity Over Time")
    plt.xlabel("Date")
    plt.ylabel("Mood Score")
    plt.xticks(rotation=45)
    plt.show()

    # --- 3️⃣ Bar Chart: Frequency of Moods ---
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(mood_counts.keys()), y=list(mood_counts.values()), palette="viridis")
    plt.xlabel("Mood")
    plt.ylabel("Count")
    plt.title("Mood Distribution Over Time")
    plt.show()

    # --- 4️⃣ Pie Chart: Percentage of Each Mood ---
    plt.figure(figsize=(6, 6))
    plt.pie(mood_counts.values(), labels=mood_counts.keys(), autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    plt.title("Mood Percentage Breakdown")
    plt.show()

# Example usage after processing:
if __name__ == "__main__":
    plot_mood_trends("/Users/pandhari/ai-diary-project/processed_diary.json")