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
    DROPOUT_RATE,
    DROP_PATH_RATE,
)

# Tiền xử lý ảnh - PHẢI GIỐNG TRAINING (ImageNet normalization)
_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None


def load_model():
    """Load model với dropout rate giống training"""
    global _model
    if _model is None:
        model = timm.create_model(
            "efficientnet_b0", 
            pretrained=False, 
            num_classes=len(CLASSES),
            drop_rate=DROPOUT_RATE,
            drop_path_rate=DROP_PATH_RATE
        )
        model.load_state_dict(torch.load(EFFICIENTNET_MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
        _model = model
        print(f"✅ Model loaded from: {EFFICIENTNET_MODEL_PATH}")
    return _model


def predict_image_bytes(image_bytes: bytes, threshold: float = 0.35, return_confidence: bool = True):
    """
    Dự đoán ảnh từ bytes với threshold đã điều chỉnh.
    
    Args:
        image_bytes: Dữ liệu ảnh dạng bytes
        threshold: Ngưỡng để phân loại REAL (0.0 - 1.0)
                   - threshold=0.35 (MẶC ĐỊNH): Giảm false positive FAKE
                   - threshold=0.5: Cân bằng
                   - threshold=0.6-0.7: Nghiêm ngặt với REAL
        return_confidence: Có trả về confidence score hay không
    
    Returns:
        Dictionary chứa kết quả dự đoán
    """
    model = load_model()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {
            "error": f"Cannot open image: {str(e)}",
            "predicted_label": None,
        }
    
    tensor = _preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    prob_fake = float(probs[0])
    prob_real = float(probs[1])
    
    # LOGIC: Dựa trên probability margin
    margin = abs(prob_fake - prob_real)
    
    # Nếu margin nhỏ (< 0.2), ưu tiên dự đoán REAL để giảm false positive
    if margin < 0.2:
        if prob_real >= 0.3:  # Threshold thấp hơn khi không chắc chắn
            pred_label = "REAL"
            confidence = prob_real
        else:
            pred_label = "FAKE"
            confidence = prob_fake
    else:
        # Margin lớn, dùng threshold bình thường
        if prob_real >= threshold:
            pred_label = "REAL"
            confidence = prob_real
        else:
            pred_label = "FAKE"
            confidence = prob_fake
    
    result = {
        "model_name": "efficientnet_b0",
        "predicted_label": pred_label,
        "probabilities": {
            "FAKE": prob_fake,
            "REAL": prob_real,
        },
        "threshold_used": threshold,
        "probability_margin": margin,
    }
    
    if return_confidence:
        result["confidence"] = confidence
        result["confidence_level"] = _get_confidence_level(confidence, margin)
    
    return result


def _get_confidence_level(confidence: float, margin: float) -> str:
    """
    Đánh giá mức độ tự tin với margin.
    
    Args:
        confidence: Xác suất của class được dự đoán
        margin: Khoảng cách giữa 2 probabilities
    
    Returns:
        String mô tả mức độ tự tin
    """
    # Nếu margin nhỏ, độ tin cậy thấp dù confidence cao
    if margin < 0.1:
        return "Very Low (Uncertain)"
    elif margin < 0.2:
        return "Low (Unclear)"
    
    # Margin lớn, dựa vào confidence
    if confidence >= 0.95:
        return "Very High"
    elif confidence >= 0.85:
        return "High"
    elif confidence >= 0.70:
        return "Medium"
    elif confidence >= 0.55:
        return "Low"
    else:
        return "Very Low"


def predict_image_file(image_path: str, threshold: float = 0.35):
    """
    Dự đoán từ file path.
    
    Args:
        image_path: Đường dẫn đến file ảnh
        threshold: Ngưỡng phân loại (mặc định 0.35)
    
    Returns:
        Dictionary chứa kết quả dự đoán
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return predict_image_bytes(image_bytes, threshold=threshold)


def batch_predict(image_bytes_list: list, threshold: float = 0.35):
    """
    Dự đoán nhiều ảnh cùng lúc.
    
    Args:
        image_bytes_list: List các image bytes
        threshold: Ngưỡng phân loại (mặc định 0.35)
    
    Returns:
        List các kết quả dự đoán
    """
    model = load_model()
    results = []
    
    for image_bytes in image_bytes_list:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = _preprocess(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            prob_fake = float(probs[0])
            prob_real = float(probs[1])
            margin = abs(prob_fake - prob_real)
            
            # Áp dụng logic tương tự
            if margin < 0.2:
                pred_label = "REAL" if prob_real >= 0.3 else "FAKE"
            else:
                pred_label = "REAL" if prob_real >= threshold else "FAKE"
            
            confidence = prob_real if pred_label == "REAL" else prob_fake
            
            results.append({
                "predicted_label": pred_label,
                "probabilities": {"FAKE": prob_fake, "REAL": prob_real},
                "confidence": confidence,
                "confidence_level": _get_confidence_level(confidence, margin),
                "probability_margin": margin,
            })
        except Exception as e:
            results.append({
                "error": str(e),
                "predicted_label": None,
            })
    
    return results


def find_optimal_threshold(test_images_dir: str, true_labels: dict):
    """
    Tìm threshold tối ưu dựa trên test images thực tế.
    
    Args:
        test_images_dir: Thư mục chứa ảnh test
        true_labels: Dict {filename: "FAKE" or "REAL"}
    
    Returns:
        Best threshold và accuracy
    """
    from pathlib import Path
    
    model = load_model()
    
    test_dir = Path(test_images_dir)
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No images found in test directory")
        return None
    
    # Collect all predictions
    predictions = {}
    print(f"\n📊 Collecting predictions from {len(image_files)} images...")
    
    for img_path in image_files:
        filename = img_path.name
        if filename not in true_labels:
            continue
        
        result = predict_image_file(str(img_path), threshold=0.5)
        if "error" not in result:
            predictions[filename] = result["probabilities"]["REAL"]
    
    if not predictions:
        print("❌ No valid predictions")
        return None
    
    # Try different thresholds
    best_threshold = 0.5
    best_accuracy = 0.0
    best_f1 = 0.0
    
    print(f"\n🔍 Testing different thresholds on {len(predictions)} images...")
    print(f"{'Threshold':<12} {'Accuracy':<10} {'F1-Score':<10} {'FAKE Acc':<12} {'REAL Acc':<12}")
    print("="*60)
    
    for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        correct = 0
        fake_correct = 0
        fake_total = 0
        real_correct = 0
        real_total = 0
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for filename, prob_real in predictions.items():
            true_label = true_labels[filename]
            pred_label = "REAL" if prob_real >= threshold else "FAKE"
            
            if true_label == "FAKE":
                fake_total += 1
                if pred_label == "FAKE":
                    fake_correct += 1
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                real_total += 1
                if pred_label == "REAL":
                    real_correct += 1
                else:
                    false_positives += 1
            
            if pred_label == true_label:
                correct += 1
        
        accuracy = correct / len(predictions)
        fake_acc = fake_correct / fake_total if fake_total > 0 else 0
        real_acc = real_correct / real_total if real_total > 0 else 0
        
        # Calculate F1-Score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:<12.2f} {accuracy:<10.2%} {f1_score:<10.4f} {fake_acc:<12.2%} {real_acc:<12.2%}")
        
        # Ưu tiên F1-score cao hơn là accuracy
        if f1_score > best_f1:
            best_f1 = f1_score
            best_accuracy = accuracy
            best_threshold = threshold
    
    print("="*60)
    print(f"✅ Best threshold: {best_threshold:.2f}")
    print(f"   Accuracy: {best_accuracy:.2%}")
    print(f"   F1-Score: {best_f1:.4f}")
    
    return best_threshold, best_accuracy, best_f1


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [threshold]")
        print("Example: python predict.py test.jpg 0.35")
        print("\nNote: Default threshold is 0.35 to reduce false FAKE predictions")
        sys.exit(1)
    
    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
    
    print(f"\n{'='*70}")
    print(f"🔍 Testing prediction with threshold={threshold}")
    print(f"{'='*70}\n")
    
    result = predict_image_file(image_path, threshold=threshold)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"📊 Result:")
        print(f"  Predicted: {result['predicted_label']}")
        print(f"  Confidence: {result['confidence']:.2%} ({result['confidence_level']})")
        print(f"\n  Probabilities:")
        print(f"    FAKE: {result['probabilities']['FAKE']:.2%}")
        print(f"    REAL: {result['probabilities']['REAL']:.2%}")
        print(f"    Margin: {result['probability_margin']:.2%}")
        print(f"\n  Threshold used: {result['threshold_used']}")
        
        # Recommendation
        margin = result['probability_margin']
        if margin < 0.2:
            print(f"\n  ⚠️  Low margin ({margin:.2%}) - Prediction uncertain")
            print(f"     Consider manual review for this image")
    
    print(f"\n{'='*70}")