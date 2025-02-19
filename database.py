from pymongo import MongoClient
from dotenv import load_dotenv
import os 
load_dotenv()

client = MongoClient(os.getenv("MONGODB_KEY"))
db = client["diary_db"]
collection = db["entries"]

def store_in_mongo(entries):
    """Store processed diary entries into MongoDB."""
    collection.insert_many(entries)
    print(f"✅ Stored {len(entries)} entries in MongoDB!")

def query_by_date(date):
    """Retrieve diary entries for a specific date."""
    return list(collection.find({"date": date}, {"_id": 0}))

def query_emotion_by_date(date):
    """Retrieve individual emotion statistics for a specific date."""
    result = collection.find_one({"date": date, "overall_emotion_count": {"$exists": True}}, {"_id": 0})
    return result if result else "❌ No emotion data found for this date."