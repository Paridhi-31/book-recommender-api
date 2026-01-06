import os
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()

def get_qdrant_client():
    return QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

def get_recommendations(search_key, search_value, client, top_n=10):
    # 1. Find the target book vector
    # Using the 'scroll' method to find the specific book by its slug or ID
    scroll_result, _ = client.scroll(
        collection_name="books",
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key=search_key, match=models.MatchValue(value=search_value))]
        ),
        with_vectors=True,
        limit=1
    )
    
    if not scroll_result:
        return []
    
    target_point = scroll_result[0]

    # 2. Perform the search using the 'search' method explicitly
    # If client.search fails, we use the points.search sub-module
    try:
        search_result = client.search(
            collection_name="books",
            query_vector=target_point.vector,
            limit=top_n + 1,
            with_payload=True
        )
    except AttributeError:
        # Fallback for specific older/newer hybrid SDK versions
        search_result = client.query_points(
            collection_name="books",
            query=target_point.vector,
            limit=top_n + 1
        ).points

    # 3. Format and return results
    return [
        {
            "book_id": hit.payload.get("book_id"), 
            "title": hit.payload.get("title"), 
            "slug": hit.payload.get("slug"),
            "score": round(hit.score, 4)
        }
        for hit in search_result 
        if hit.payload.get("book_id") != target_point.payload.get("book_id")
    ]