import os
import pandas as pd
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import time

load_dotenv()

def fetch_data():
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("DATABASE_NAME")]
    collection = db[os.getenv("BOOK_COLLECTION")]
    # Fetching slug along with other fields
    cursor = collection.find({}, {"_id": 1, "title": 1, "author": 1, "genre": 1, "description": 1, "slug": 1})
    df = pd.DataFrame(list(cursor))
    df['book_id'] = df['_id'].astype(str)
    return df

def train_and_upload():
    df = fetch_data()
    # Data Cleaning for the "Soup"
    df['genre'] = df['genre'].apply(lambda x: ' '.join(str(i) for i in x) if isinstance(x, list) else str(x))
    df['author'] = df['author'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    df['soup'] = (df['title'] + " " + df['author'] + " " + df['genre'] + " " + df['description']).fillna('')

    print("Loading embedding model...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. INCREASE TIMEOUT: Set timeout to 60 seconds (default is usually 5-10)
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=100 # Increased further for stability
    )

    # 2. MODERNIZED COLLECTION CREATION
    # Avoids the DeprecationWarning
    if not client.collection_exists("books"):
        client.create_collection(
            collection_name="books",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        client.create_payload_index("books", "slug", models.PayloadSchemaType.KEYWORD)
        client.create_payload_index("books", "book_id", models.PayloadSchemaType.KEYWORD)
    
    # --- RESUME LOGIC ---
    # Check how many points exist to skip them
    collection_info = client.get_collection("books")
    points_count = collection_info.points_count
    print(f"Collection already has {points_count} books. Skipping ahead...")

    points = []
    batch_size = 100 
    
    # Start loop from where we left off
    for idx, row in df.iloc[points_count:].iterrows():
        try:
            vector = encoder.encode(row['soup']).tolist()
            points.append(PointStruct(
                id=idx, # Use idx to match your previous upload logic
                vector=vector, 
                payload={
                    "book_id": row['book_id'], 
                    "title": row['title'], 
                    "slug": row.get('slug', ''),
                    "author": row['author']
                }
            ))
            
            if len(points) >= batch_size:
                client.upsert(collection_name="books", points=points)
                print(f"Uploaded {idx + 1} / {len(df)} books...")
                points = []
                
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            print("Waiting 5 seconds before retrying...")
            time.sleep(5)
            continue
    
    if points: 
        client.upsert(collection_name="books", points=points)

    print(f"✅ Uploaded {len(df)} books with slugs to Qdrant Cloud.")

if __name__ == "__main__":
    train_and_upload()