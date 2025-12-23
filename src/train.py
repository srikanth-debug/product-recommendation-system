import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
import pickle
import os
import mlflow

from data_ingestion import load_data
from feature_engineering import pre_process

def train():
    df=load_data()
    df=pre_process(df)
    print(df.head())

    user_item_matrix=coo_matrix(
        (df["rating"],(df["user_id"],df["product_id"]))
    ).tocsr()

    model = AlternatingLeastSquares(
        factors=50,
        iterations=20
    )
    model.fit(user_item_matrix)

    os.makedirs("models", exist_ok=True)

    with open("models/als_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/user_item_matrix.pkl", "wb") as f:
        pickle.dump(user_item_matrix, f)    

    print("Model trained & saved")
    
mlflow.start_run()
mlflow.log_param("factors", 50)
mlflow.log_param("iterations", 20)
mlflow.log_artifact("models/als_model.pkl")
mlflow.end_run()
if __name__ == "__main__":
    train()




