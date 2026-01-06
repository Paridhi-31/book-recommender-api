from fastapi import FastAPI
from contextlib import asynccontextmanager
from recommender_inference import get_qdrant_client, get_recommendations
from fastapi.middleware.cors import CORSMiddleware
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.qdrant = get_qdrant_client()
    yield
    print("Shutting down: Releasing resources.")


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
    "https://www.divalorebooks.com/",
    "https://divalorebooks.com/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all standard HTTP methods
    allow_headers=["*"],
)

# --- Endpoints ---

# Route 1: Slug-based (SEO friendly)
@app.get("/recommend/slug/{slug}")
async def recommend_by_slug(slug: str, top_n: int = 10):
    recs = get_recommendations("slug", slug, app.state.qdrant, top_n)
    return {"source": "slug", "recommendations": recs}

# Route 2: ID-based (Internal usage)
@app.get("/recommend/id/{book_id}")
async def recommend_by_id(book_id: str, top_n: int = 10):
    recs = get_recommendations("book_id", book_id, app.state.qdrant, top_n)
    return {"source": "id", "recommendations": recs}

@app.get("/")
def home():
    return {"status": "online", "modes": ["slug", "id"]}

if __name__ == "__main__":
    import uvicorn
    # This will run the app on http://127.0.0.1:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)