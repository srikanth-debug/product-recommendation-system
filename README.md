# 🛒 Product Recommendation System (End-to-End ML Project)

An **end-to-end machine learning product recommendation system** built using **Collaborative Filtering (ALS)**, exposed via a **FastAPI REST API**, and **Dockerized** for production deployment.

This project demonstrates **real-world ML engineering**, not just model training 🚀

---

## 🚀 Features

- End-to-end ML pipeline (data → model → API → Docker)
- Collaborative Filtering using **Alternating Least Squares (ALS)**
- Efficient **sparse matrix (CSR)** handling
- Model & interaction matrix persistence
- REST API using **FastAPI**
- Dockerized deployment
- Production-style project structure

---

## 🧠 Problem Statement

E-commerce platforms need to recommend relevant products to users based on historical interactions such as ratings, clicks, or purchases.

**Objective:**  
Build a recommendation system that suggests **Top-N products** to a user using historical **user–product interaction data**.

---

## 🏗️ System Architecture

Data (CSV)
↓
Feature Engineering
↓
User–Item Interaction Matrix (CSR)
↓
ALS Model Training
↓
Model + Matrix Persistence
↓
FastAPI Inference API
↓
Dockerized Deployment


---

## 📂 Project Structure

product-recommendation-system/
├── api/
│ └── app.py # FastAPI inference service
├── src/
│ ├── train.py # Model training pipeline
│ ├── data_ingestion.py # Load interaction data
│ ├── feature_engineering.py
│ ├── evaluate.py # Model evaluation
│ └── Inference.py # Offline inference logic
├── data/
│ └── interactions.csv # Sample interaction data
├── models/ # (Optional) Saved model artifacts
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md

---

## ⚙️ Tech Stack

- **Python 3.10 / 3.11**
- **FastAPI**
- **Implicit (ALS)**
- **Pandas / NumPy**
- **SciPy (Sparse Matrices)**
- **Docker**

---

## 🔬 Model Details

- **Algorithm:** Alternating Least Squares (ALS)
- **Type:** Collaborative Filtering
- **Input:** User–Item interaction matrix
- **Output:** Top-N product recommendations per user
- **Matrix Format:** CSR (Compressed Sparse Row)

---

## ▶️ Run Locally (Without Docker)

### 1️⃣ Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Train the model
python src/train.py
4️⃣ Start the API
uvicorn api.app:app --reload
5️⃣ Open Swagger UI
http://127.0.0.1:8000/docs
🐳 Docker Deployment
Build Docker image
docker build --no-cache -t product-recommender .
Run container
docker run -p 8000:8000 product-recommender
API available at:
http://localhost:8000/docs
🔁 Example API Call
GET /recommend/1
Example Response
{
  "user_id": 1,
  "recommendations": [101, 104, 112, 98, 87]
}
🧠 Key Engineering Learnings
ALS requires user–item interaction matrix at inference time
Importance of saving multiple ML artifacts, not just the model
Efficient handling of large sparse matrices
Environment consistency between training and inference
Debugging real-world Docker & deployment issues
📌 Future Improvements
Cold-start user handling
Hybrid recommendation (content + CF)
MLflow model registry
Batch recommendation jobs
Monitoring & logging
GPU-accelerated ALS
👤 Author
Srikanth Parsa
Aspiring Data Scientist / Machine Learning Engineer
⭐ Why This Project Matters
This is not a toy project.
It demonstrates:

Real ML pipelines
Production deployment
API-based inference
Docker & system-level debugging
Strong project for ML Engineer / Data Scientist interviews.

---


---

