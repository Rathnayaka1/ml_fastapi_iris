# 🌸 Iris Classification with FastAPI

This project demonstrates how to deploy a **Machine Learning model** as a **web API** using **FastAPI**.  
The model predicts the species of an Iris flower based on its measurements (sepal length, sepal width, petal length, petal width).  

---

## 📌 Problem Description

The **Iris dataset** is a well-known dataset in machine learning with 150 flower samples.  
Each flower is classified into one of three species:

- Setosa 🌱
- Versicolor 🌿
- Virginica 🌸

The task is to build a model that accurately classifies the species given the four input features.

---

## 🧠 Model Choice

We used a **RandomForestClassifier** from scikit-learn because:

- It works well with small datasets
- Provides good accuracy and interpretability
- Reduces overfitting using bagging

The model was trained on 80% of the dataset and evaluated on 20% test data.  
The model was serialized using `joblib` and loaded in the FastAPI app for inference.

---

## ⚡ Technology Stack

- **Python 3**
- **FastAPI** (Web Framework)
- **scikit-learn** (ML Model)
- **Uvicorn** (ASGI Server)
- **pydantic** (Input validation)
- **joblib** (Model serialization)

---

## 🚀 API Endpoints

### 1. Health Check 

**GET /**  

Returns API status.  

```json
{
  "status": "healthy",
  "message": "Iris Classifier API is running"
}
