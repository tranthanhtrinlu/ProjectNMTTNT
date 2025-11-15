# ai-service/api/main.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.predict import predict_image_bytes

app = FastAPI(title="AI Image Real/Fake Classifier")

# Cho phép CORS (để Spring Boot / front-end gọi được)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Dev thì để *, sau này có thể siết lại
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là hình ảnh")

    image_bytes = await file.read()
    result = predict_image_bytes(image_bytes)

    return {
        "model": result["model_name"],
        "label": result["predicted_label"],
        "probabilities": result["probabilities"],
    }
