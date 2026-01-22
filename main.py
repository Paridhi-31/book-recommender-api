from fastapi import FastAPI, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
from recommender_inference import get_qdrant_client, get_recommendations
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
from cachetools import TTLCache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache: Stores 1000 books for 24 hours
recommendation_cache = TTLCache(maxsize=1000, ttl=86400)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize Qdrant
    try:
        app.state.qdrant = get_qdrant_client()
        logger.info("✅ Qdrant client initialized.")
        
        # 2. Start Lazy Warmup in the background
        # This prevents the first user from hitting a 504 timeout
        async def startup_warmup():
            # Add your most popular/important slugs here
            popular_slugs = ["the-great-gatsby", "1984", "atomic-habits"] 
            logger.info(f"🚀 Background warmup started for {len(popular_slugs)} books...")
            
            for slug in popular_slugs:
                cache_key = f"slug_{slug}_10"
                if cache_key not in recommendation_cache:
                    try:
                        # Pre-calculate and store in cache
                        recs = get_recommendations("slug", slug, app.state.qdrant, 10)
                        recommendation_cache[cache_key] = recs if recs is not None else []
                        logger.info(f"✨ Cached: {slug}")
                    except Exception as e:
                        logger.warning(f"⚠️ Warmup failed for {slug}: {e}")
            logger.info("🏁 Warmup complete.")

        asyncio.create_task(startup_warmup())
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Qdrant: {e}")
    
    yield
    logger.info("Shutting down: Releasing resources.")


# --- FastAPI App Definition ---
app = FastAPI(
    lifespan=lifespan, 
    title="Hybrid Book Recommender API",
    description="V1.0: Serves Content-Based Filter (CBF) and is ready for future Collaborative Filtering (CF)."
)

# --- CORS Configuration ---
origins = [
    "http://localhost:5173",  # Your React frontend's address
    "http://localhost:3000",  # Your Express/Node server (if using)
    "https://www.divalorebooks.com",
    "https://divalorebooks.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all standard HTTP methods
    allow_headers=["*"],
)

# --- Endpoints ---

@app.get("/recommend/slug/{slug}")
async def recommend_by_slug(slug: str, top_n: int = 10):
    cache_key = f"slug_{slug}_{top_n}"
    
    # Check Cache
    if cache_key in recommendation_cache:
        return {"source": "slug", "recommendations": recommendation_cache[cache_key], "cached": True}

    try:
        # Calculate if not in cache
        recs = get_recommendations("slug", slug, app.state.qdrant, top_n)
        final_recs = recs if recs is not None else []
        recommendation_cache[cache_key] = final_recs
        return {"source": "slug", "recommendations": final_recs, "cached": False}
    except Exception as e:
        logger.error(f"Error for {slug}: {e}")
        return {"source": "slug", "recommendations": [], "error": str(e)}

@app.get("/recommend/id/{book_id}")
async def recommend_by_id(book_id: str, top_n: int = 10):
    cache_key = f"id_{book_id}_{top_n}"
    
    if cache_key in recommendation_cache:
        return {"source": "id", "recommendations": recommendation_cache[cache_key], "cached": True}

    try:
        recs = get_recommendations("book_id", book_id, app.state.qdrant, top_n)
        final_recs = recs if recs is not None else []
        recommendation_cache[cache_key] = final_recs
        return {"source": "id", "recommendations": final_recs, "cached": False}
    except Exception as e:
        logger.error(f"Error for {book_id}: {e}")
        return {"source": "id", "recommendations": [], "error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "staying alive"}

@app.get("/")
def home():
    return {"status": "online", "modes": ["slug", "id"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)