import os
import pandas as pd
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

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
        timeout=60 
    )

    # 2. MODERNIZED COLLECTION CREATION
    # Avoids the DeprecationWarning
    if not client.collection_exists("books"):
        client.create_collection(
            collection_name="books",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    
        # ADD THESE LINES HERE:
        client.create_payload_index("books", "slug", models.PayloadSchemaType.KEYWORD)
        client.create_payload_index("books", "book_id", models.PayloadSchemaType.KEYWORD)
    else:
        print("Collection 'books' already exists. Skipping creation.")

    # 3. REDUCED BATCH SIZE: Send 100 points per request to prevent timeouts
    points = []
    batch_size = 100 
    
    print(f"Starting upload for {len(df)} books...")
    for idx, row in df.iterrows():
        vector = encoder.encode(row['soup']).tolist()
        points.append(PointStruct(
            id=idx, 
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
    
    if points: 
        client.upsert(collection_name="books", points=points)

    print(f"✅ Uploaded {len(df)} books with slugs to Qdrant Cloud.")

if __name__ == "__main__":
    train_and_upload()