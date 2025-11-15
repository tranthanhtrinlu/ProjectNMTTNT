# ai-service/src/predict.py
import io
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
from PIL import Image
from torchvision import transforms
import timm

from config import (
    IMAGE_SIZE,
    DEVICE,
    CLASSES,
    EFFICIENTNET_MODEL_PATH,
)

# Tiền xử lý ảnh
_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

_model = None


def load_model():
    global _model
    if _model is None:
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASSES))
        model.load_state_dict(torch.load(EFFICIENTNET_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model


def predict_image_bytes(image_bytes: bytes):
    model = load_model()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    pred_idx = int(probs.argmax())
    pred_label = CLASSES[pred_idx]

    return {
        "model_name": "efficientnet_b0",
        "predicted_label": pred_label,
        "probabilities": prob_dict,
    }
