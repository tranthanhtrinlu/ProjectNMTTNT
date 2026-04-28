# 🎯 AI Image Real/Fake Classifier (Java + Python Microservices)

🚀 A full-stack AI-powered web application that detects whether an image is **REAL or FAKE** using Deep Learning models.

This project demonstrates how to integrate:
- **Spring Boot (Java)** for web & backend
- **FastAPI (Python)** for AI inference
- **Deep Learning (PyTorch)** for image classification

---

## 📌 Demo Overview

👉 User uploads an image → system predicts:
- REAL ✅
- FAKE ❌

With:
- Confidence score
- Model name
- Visualization UI

---

## 🧠 Architecture
[User]
↓ upload image
[Spring Boot + JSP UI]
↓
[Controller]
↓
[Service (RestTemplate)]
↓ HTTP
[FastAPI AI Service]
↓
[Deep Learning Model (EfficientNet / ViT)]
↓
JSON response
↓
Spring Boot render result

---

## 🛠️ Tech Stack

### 🔹 Backend (Java)
- Spring Boot 3
- Spring MVC
- JSP (View)
- RestTemplate (API call)
- Multipart File Upload

### 🔹 AI Service (Python)
- FastAPI
- PyTorch
- timm (EfficientNet, ViT)
- Transformers

### 🔹 Others
- Kaggle Dataset (CIFAKE)
- Maven
- REST API

---

## 📂 Project Structure

ProjectJavaAI/
├── controller/ # Handle HTTP requests
├── service/ # Call AI service
├── model/ # Response mapping
├── config/ # RestTemplate config
├── webapp/views/ # JSP UI
└── application.properties

ProjectPythonAI/
├── api/main.py # FastAPI endpoint
├── src/ # Model & prediction logic
├── config.py # Model config
├── notebooks/ # Training notebooks
├── models/ # Saved models (.pth)
└── requirements.txt

---

## ⚙️ How It Works

### 1. Upload Image
User uploads an image via JSP UI.

### 2. Java Backend
- Receives image
- Sends HTTP request to AI service

### 3. Python AI Service
- Loads trained model
- Predicts image (REAL / FAKE)
- Returns JSON result

### 4. Display Result
- Java renders prediction on UI
- Shows probabilities and label

---

## 🔥 API Example

### Request
POST /predict
Content-Type: multipart/form-data
file: image.jpg

### Response
```json
{
  "model": "efficientnet_b0",
  "label": "REAL",
  "probabilities": {
    "FAKE": 0.12,
    "REAL": 0.88
  }
}
🧪 Run Project Locally
🔹 Step 1: Run AI Service (Python)
cd ProjectPythonAI
pip install -r requirements.txt
uvicorn api.main:app --reload

👉 AI service runs at:
http://localhost:8000
🔹 Step 2: Run Spring Boot (Java)
cd ProjectJavaAI
./mvnw spring-boot:run
👉 Web app:
http://localhost:8080
📊 Dataset
CIFAKE Dataset (Kaggle)
Used for training:
REAL images
AI-generated FAKE images
🤖 Models Used
EfficientNet-B0
Vision Transformer (ViT)
💡 Key Features

✅ Upload and classify images
✅ Real-time prediction via API
✅ Probability visualization
✅ Clean UI with preview
✅ Microservice architecture (Java ↔ Python)

🧨 Highlights (For Recruiters)
Designed microservice architecture (Java + Python AI)
Integrated Spring Boot with FastAPI via REST API
Built end-to-end AI inference pipeline
Applied Deep Learning models (EfficientNet, ViT)
Implemented file upload + real-time prediction UI
🚀 Future Improvements
Deploy on cloud (GCP / AWS)
Replace JSP with React frontend
Use WebClient instead of RestTemplate
Add authentication
Optimize model inference (GPU / batching)
👤 Author

Tran Thanh Tri
📍 Ho Chi Minh City
