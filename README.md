Credit Card Fraud Detection System

🚀 Overview

An end-to-end Machine Learning system for detecting fraudulent credit card transactions using multiple classification models, advanced imbalance handling techniques, ensemble learning, hyperparameter optimization, and production deployment.

This project goes beyond traditional fraud detection notebooks by implementing a complete ML workflow, including model experimentation, evaluation, API development, Docker containerization, and cloud deployment.

Key Highlights

* End-to-End ML Pipeline
* Fraud Detection on Highly Imbalanced Data
* Random Forest, XGBoost, and Neural Network (MLP)
* SMOTE Oversampling
* Optuna Hyperparameter Optimization
* Stacking Ensemble Learning
* Feature Selection Analysis
* Threshold Optimization
* FastAPI REST API
* Docker Containerization
* Cloud Deployment

⸻

📊 Problem Statement

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a tiny fraction of all transactions.

Challenges include:

* Extreme class imbalance
* High cost of false negatives
* Need for high precision and recall
* Real-world deployment requirements

The objective is to accurately identify fraudulent transactions while minimizing missed fraud cases.

⸻

📂 Dataset

Dataset: Credit Card Fraud Detection Dataset

* Total Transactions: 284,807
* Fraudulent Transactions: 492
* Fraud Rate: 0.172%

Features:

* V1 – V28 (PCA-transformed features)
* Time
* Amount
* Class (Target)

Target:

* 0 → Legitimate Transaction
* 1 → Fraudulent Transaction

🏗️ Project Architecture

Credit-Card-Fraud-Detection/
│
├── api/
│   └── main.py
│
├── artifacts/
│   ├── model.pkl
│   └── scaler.pkl
│
├── data/
│
├── notebook/
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── save_model.py
│   └── predict.py
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── README.md

⚙️ Workflow

Data Collection
        ↓
Data Cleaning
        ↓
Duplicate Removal
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
Model Training
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Error Analysis
        ↓
Model Saving
        ↓
FastAPI API
        ↓
Docker Container
        ↓
Cloud Deployment

🛠️ Technologies Used

Machine Learning

* Scikit-Learn
* XGBoost
* Optuna
* Imbalanced-Learn

Backend

* FastAPI
* Uvicorn

Deployment

* Docker
* Railway

Data Processing

* NumPy
* Pandas
* Joblib

⸻

🤖 Models Implemented

1. Random Forest

Used as the initial baseline model.

Results

<img width="2940" height="1912" alt="image" src="https://github.com/user-attachments/assets/580422da-db4b-4c9e-b2b3-900c14ea5d21" />

2. Random Forest + SMOTE

SMOTE was applied to address class imbalance.







Observation

* Recall improved
* PR-AUC improved
* More fraudulent transactions detected
* Slight increase in false positives

3. XGBoost (Baseline)

Gradient boosting model optimized for tabular data.

Results

Metric         Score

Precision      0.9737
Recall         0.7789
F1 score       0.8655
ROC-AUC        0.9728
PR-AUC0.       0.8251


4. Optuna-Tuned XGBoost

Bayesian optimization using Optuna to find optimal hyperparameters.

Results




Observation

* Slight improvement in Recall
* Slight improvement in PR-AUC
* Increased false positives
* Baseline XGBoost remained the stronger overall model
  

⸻

5. Multi-Layer Perceptron (Neural Network)

A feed-forward neural network was implemented to compare traditional machine learning models
with deep learning approaches on highly imbalanced tabular data.

Purpose

* Evaluate neural network performance on fraud detection
* Compare against tree-based models
* Use as a component in the final ensemble architecture

Observation

* Successfully learned fraud patterns
* Competitive performance
* Added model diversity for stacking

⸻

6. Stacking Ensemble

Base Models

* Random Forest
* XGBoost

Meta Learner

* Multi-Layer Perceptron (Neural Network)
  
Results




Conclusion

Using all features provided better performance.

⸻

🎯 Threshold Analysis

Threshold tuning was performed to reduce false negatives.

Findings

Most missed fraud cases received extremely low predicted probabilities.

Lowering the threshold significantly increased false positives while providing limited recovery of missed fraud transactions.

Final Decision

Default threshold of 0.5 was retained.

⸻

📈 Why PR-AUC?

Because fraud detection is highly imbalanced, accuracy can be misleading.

This project prioritizes:

1. PR-AUC
2. Recall
3. F1 Score
4. Precision
5. ROC-AUC

PR-AUC provides a more reliable measure of model performance for rare-event detection.

⸻

🧠 Error Analysis

Fraudulent transactions missed by the model were investigated manually.

Findings:

* Most false negatives received extremely low fraud probabilities.
* Threshold adjustments alone could not recover them efficiently.
* Additional feature engineering would likely be required for further improvements.

This analysis helped identify practical limitations of the dataset and model.

🌐 FastAPI Inference Service

The trained model is served through a FastAPI REST API.

Endpoints

Health Check
GET /

Fraud Prediction
POST /predict

Input:
{
  "Time": 1000,
  "Amount": 250,
  "V1": 0.1,
  "...": "...",
  "V28": 0.5
}

Response:
{
  "prediction": 0,
  "probability": 0.00024
}

🐳 Docker Deployment

The application is containerized using Docker.

Build
docker build -t fraud-api .

Run
docker run -p 8000:8000 fraud-api

☁️ Deployment

The application is deployed on Railway using Docker.

Features:

* Containerized deployment
* Production-ready API
* Scalable architecture
* Accessible through a public endpoint

⸻

📌 Key Learning Outcomes

* Handling highly imbalanced datasets
* Precision-Recall tradeoffs
* SMOTE oversampling
* Ensemble Learning
* Neural Networks for tabular data
* Hyperparameter Optimization with Optuna
* Feature Selection
* Error Analysis
* Model Serialization
* REST API Development
* Docker Containerization
* Cloud Deployment

⸻

🎯 Future Improvements

* SHAP Explainability
* MLflow Experiment Tracking
* Model Monitoring
* Data Drift Detection
* CI/CD Pipeline
* Automated Retraining
* Real-Time Fraud Streaming

⸻

👨‍💻 Author

Ansh Yadav

B.Tech CSE (AI & ML)

Aspiring Machine Learning Engineer

GitHub: https://github.com/anshyadavv1














  
