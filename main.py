# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from recommender_inference import load_all_artifacts, get_hybrid_recommendations, get_book_id_from_title 

# --- Model Paths ---
CBF_PATH = "cbf_similarity_matrix.pkl"
CF_PATH = "cf_model_placeholder.pkl"

# --- Pydantic Schema for Response ---
class RecommendationItem(BaseModel):
    book_id: str
    title: str
    score: float
    model: str # Indicates if the result came from 'CBF', 'CF', or 'Hybrid'

class RecommendationList(BaseModel):
    recommendations: List[RecommendationItem]
    
# --- Lifespan Function for Model Loading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads models into app state on startup."""
    print("Starting up: Loading model artifacts...")
    # This loads the Scikit-learn matrix and the PyTorch placeholder
    app.state.cbf_model, app.state.cf_model = load_all_artifacts(CBF_PATH, CF_PATH)
    
    if app.state.cbf_model is None:
        print("FATAL: CBF model failed to load. Ensure 'recommender_trainer.py' was run.")
    
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
    "https://www.parablebooks.work.gd",
    "https://parablebooks.work.gd",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all standard HTTP methods
    allow_headers=["*"],
)

# --- Endpoints ---

@app.get("/recommend/book/{book_id}", response_model=RecommendationList)
async def recommend_by_book(book_id: str, top_n: int = 10):
    """
    Retrieves recommendations based on a single book's content (Item-to-Item Similarity).
    A dummy user_id is used to prepare the API for the future hybrid function call.
    """
    # Use a dummy user_id as a placeholder for the user who is viewing the book
    user_id = "book_viewer" 

    if app.state.cbf_model is None:
         return {"recommendations": []}

    recs = get_hybrid_recommendations(
        user_id=user_id,
        book_id=book_id,
        cbf_model=app.state.cbf_model,
        cf_model=app.state.cf_model,
        top_n=top_n
    )
    
    return {"recommendations": recs}

@app.get("/recommend/title/{title}", response_model=RecommendationList)
async def recommend_by_title(title: str, top_n: int = 10):
    """
    Retrieves recommendations based on a single book's title.
    """
    user_id = "book_viewer" 

    if app.state.cbf_model is None:
        return {"recommendations": []}
    
    # 1. Look up the book_id using the title
    book_id = get_book_id_from_title(title, app.state.cbf_model)

    if not book_id:
        # 2. Raise a 404 error if title is not found
        raise HTTPException(
            status_code=404, 
            detail=f"Book with title '{title}' not found in the model index. Note: This requires an exact title match."
        )

    # 3. Call the core recommendation logic using the found book_id
    recs = get_hybrid_recommendations(
        user_id=user_id,
        book_id=book_id,
        cbf_model=app.state.cbf_model,
        cf_model=app.state.cf_model,
        top_n=top_n
    )
    
    return {"recommendations": recs}