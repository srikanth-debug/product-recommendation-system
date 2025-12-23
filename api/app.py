from fastapi import FastAPI
import pickle
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "als_model.pkl")
MATRIX_PATH = os.path.join(BASE_DIR, "models", "user_item_matrix.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(MATRIX_PATH, "rb") as f:
    user_item_matrix = pickle.load(f)

@app.get("/recommend/{user_id}")
def recommend_products(user_id: int):
    recs = model.recommend(
        user_id,
        user_item_matrix[user_id],
        N=5
    )
    product_ids = [int(r[0]) for r in recs]
    return {"user_id": user_id, "recommendations": product_ids}
