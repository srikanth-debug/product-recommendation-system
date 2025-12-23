import pickle

def load_model():
    with open("models/als_model.pkl", "rb") as f:
        return pickle.load(f)
    
def recommend(model, user_id, n-5):
    recommendations = model.recommend(user_id, None, N=n)
    return recommendations