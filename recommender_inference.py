# recommender_inference.py
import pickle
import os
import numpy as np

# --- 1. Model Loading ---
def load_all_artifacts(cbf_path, cf_path):
    """Loads all model artifacts (CBF matrix and CF placeholder) into memory."""
    
    cbf_model = None
    if os.path.exists(cbf_path):
        with open(cbf_path, 'rb') as f:
            cbf_model = pickle.load(f)
        print("✅ CBF Model loaded.")
    
    cf_model = None
    if os.path.exists(cf_path):
        with open(cf_path, 'rb') as f:
            cf_model = pickle.load(f)
        print("✅ CF Model Placeholder loaded.")
    
    return cbf_model, cf_model

# Public function to look up Book ID from Title
def get_book_id_from_title(title: str, cbf_model):
    """Looks up the book ID (ObjectId string) given a book title."""
    title_to_id = cbf_model.get('title_to_id', {})
    # Use exact match for simplicity. You might want to add fuzzy matching later.
    return title_to_id.get(title)

# --- 2. Core Prediction Functions ---

def get_cbf_recommendations(book_id, cbf_model, top_n=10):
    """Fetches recommendations using the Content-Based Filter (CBF)."""
    
    cosine_sim = cbf_model.get('similarity_matrix')
    indices = cbf_model.get('indices')
    book_ids = cbf_model.get('book_ids')
    id_to_title = cbf_model.get('id_to_title')

    if book_id not in indices:
        return []
    
    # Get the row index for the given book_id
    idx = indices[book_id]
    
    # Get the similarity scores and sort them
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N indices (skip index 0, which is the book itself)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    top_scores = [i[1] for i in sim_scores[1:top_n+1]]
    
    recommendations = [
        {"book_id": book_ids[i],
         "title": id_to_title.get(book_ids[i], "Title Not Found"), 
         "score": round(score, 4), 
         "model": "CBF"
        } 
        for i, score in zip(top_indices, top_scores)
    ]
    
    return recommendations

def get_cf_recommendations(user_id, cf_model, top_n=10):
    """Placeholder for future Collaborative Filtering (PyTorch/DL) predictions."""
    # When CF is ready, this function will contain the logic to call the PyTorch model.
    print(f"CF Model Status for User {user_id}: {cf_model.get('status', 'Unknown')}")
    
    # Currently returns an empty list
    return []

# --- 3. Hybrid Function ---

def get_hybrid_recommendations(user_id, book_id, cbf_model, cf_model, top_n=10):
    """
    Combines results from CBF and CF models.
    For now, it relies solely on CBF.
    """
    
    # 1. Get content-based recommendations (CBF is the only one working now)
    cbf_recs = get_cbf_recommendations(book_id, cbf_model, top_n=top_n)

    # 2. Get collaborative filtering recommendations (Will be empty until you add user data)
    # cf_recs = get_cf_recommendations(user_id, cf_model, top_n=top_n) 
    
    # 3. Simple combination (returns CBF only for now)
    return cbf_recs