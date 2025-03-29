import matplotlib.pyplot as plt
import pandas as pd
import chromadb

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_diary_db")
collection = chroma_client.get_collection(name="diary_entries")

def visualize_ner_activity_trends():
    """Visualize Named Entity & Activity trends over time."""
    
    results = collection.peek()
    
    entity_counts = {}
    activity_counts = {}

    for meta in results["metadatas"]:
        for entity, entity_type in meta.get("named_entities", []):
            entity_counts[entity] = entity_counts.get(entity, 0) + 1

        for activity in meta.get("activities", []):
            activity_counts[activity] = activity_counts.get(activity, 0) + 1

    # Convert to DataFrame
    df_entities = pd.DataFrame(list(entity_counts.items()), columns=["Entity", "Count"]).sort_values("Count", ascending=False)
    df_activities = pd.DataFrame(list(activity_counts.items()), columns=["Activity", "Count"]).sort_values("Count", ascending=False)

    # Plot Entity Trends
    plt.figure(figsize=(10, 5))
    plt.bar(df_entities["Entity"][:10], df_entities["Count"][:10], color="blue")
    plt.xlabel("Entities")
    plt.ylabel("Count")
    plt.title("Top 10 Named Entities Mentioned Over Time")
    plt.xticks(rotation=45)
    plt.show()

    # Plot Activity Trends
    plt.figure(figsize=(10, 5))
    plt.bar(df_activities["Activity"][:10], df_activities["Count"][:10], color="green")
    plt.xlabel("Activities")
    plt.ylabel("Count")
    plt.title("Top 10 Activities Mentioned Over Time")
    plt.xticks(rotation=45)
    plt.show()