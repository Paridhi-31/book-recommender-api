# recommender_trainer.py
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# --- Configuration (UPDATE THESE VALUES!) ---
MONGO_URI = "mongodb+srv://parableapp:281hVHwpBqQ4PQVf@cluster0.v1hbkya.mongodb.net/book_db?retryWrites=true&w=majority" 
DATABASE_NAME = "book_db"         
BOOK_COLLECTION = "books"

# Model Artifact Paths
CBF_MODEL_PATH = "cbf_similarity_matrix.pkl"
CF_MODEL_PATH = "cf_model_placeholder.pkl" 

# --- A. Data Extraction ---

def fetch_book_data_for_cbf():
    """Connects to MongoDB and fetches book metadata."""
    print("Connecting to MongoDB...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster') 
        db = client[DATABASE_NAME]
        collection = db[BOOK_COLLECTION]
        
        # Fetch only the fields needed for the 'soup'
        cursor = collection.find(
            {}, 
            {"_id": 1, "title": 1, "author": 1, "genre": 1, "description": 1}
        )
        df = pd.DataFrame(list(cursor))
        
        # Rename _id to book_id and convert to string
        if '_id' in df.columns:
            df.rename(columns={'_id': 'book_id'}, inplace=True)
            df['book_id'] = df['book_id'].astype(str)
        
        print(f"✅ Successfully loaded {len(df)} books into DataFrame.")
        return df
    except Exception as e:
        print(f"❌ Error fetching book data: {e}")
        return pd.DataFrame()

# --- B. CBF Training ---

def train_cbf_model(df):
    """Generates the Cosine Similarity Matrix and saves CBF artifacts."""
    if df.empty:
        print("Cannot train CBF: DataFrame is empty.")
        return

    print("Starting CBF model training (TF-IDF and Cosine Similarity)...")

    # 1. Data Transformation (The 'Soup' Creation)
    for col in ['description', 'author', 'genre', 'title']:
        if col not in df.columns: df[col] = ''
    
    # Handle 'genre' (if it's a list) and clean 'author'
# 🔑 FIX: Convert every item in the 'genre' list to a string before joining.
    # This prevents the TypeError: sequence item 0: expected str instance, ObjectId found.
    df['genre'] = df['genre'].apply(
        lambda x: ' '.join(str(item) for item in x).lower() 
        if isinstance(x, list) else str(x).lower()
    ) 
    df['author'] = df['author'].apply(
        lambda x: ''.join(str(item) for item in x).lower().replace(' ', '')
        if isinstance(x, list) else str(x).lower().replace(' ', '')
    )   

    # Combine available text fields. No 'tags' column is fine.
    df['soup'] = df['title'] + ' ' + df['author'] + ' ' + df['genre'] + ' ' + df['description']
    
    # 2. TF-IDF and Cosine Similarity
    # min_df=1 is necessary for your small dataset to not ignore any term
    tfidf = TfidfVectorizer(stop_words='english', min_df=1) 
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 3. Create Index Mapping and ID-to-Title Mapping
    indices = pd.Series(df.index, index=df['book_id']).drop_duplicates()
    book_id_list = df['book_id'].tolist()
    id_to_title = pd.Series(df['title'].values, index=df['book_id']).to_dict()
    title_to_id = pd.Series(df['book_id'].values, index=df['title']).to_dict()

    # 4. Save the CBF artifacts (the similarity matrix is the model)
    cbf_artifacts = {
        'similarity_matrix': cosine_sim,
        'indices': indices,
        'book_ids': book_id_list,
        'id_to_title': id_to_title,
        'title_to_id': title_to_id,
        'model_type': 'CBF'
    }
    
    with open(CBF_MODEL_PATH, 'wb') as file:
        pickle.dump(cbf_artifacts, file)
        
    print(f"✅ CBF Model artifacts saved to {CBF_MODEL_PATH}")

# --- C. CF Placeholder (Future-Proofing) ---

def create_cf_placeholder():
    """Saves a placeholder file ready to be replaced by a PyTorch model later."""
    cf_artifacts = {
        'model_type': 'CF',
        'status': 'Placeholder',
        'message': 'This file will be replaced by a trained PyTorch model when user interaction data (ratings/likes) are available.'
    }
    with open(CF_MODEL_PATH, 'wb') as file:
        pickle.dump(cf_artifacts, file)
    print(f"✅ CF Model Placeholder created at {CF_MODEL_PATH}")


if __name__ == '__main__':
    # Execute the training process
    book_df = fetch_book_data_for_cbf()
    train_cbf_model(book_df)
    create_cf_placeholder()
    print("\n--- Training Pipeline Complete ---")