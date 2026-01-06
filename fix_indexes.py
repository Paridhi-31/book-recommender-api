import os
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()

def create_payload_indexes():
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    
    print("Creating index for 'slug'...")
    client.create_payload_index(
        collection_name="books",
        field_name="slug",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    print("Creating index for 'book_id'...")
    client.create_payload_index(
        collection_name="books",
        field_name="book_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    
    print("✅ Indexes created successfully!")

if __name__ == "__main__":
    create_payload_indexes()