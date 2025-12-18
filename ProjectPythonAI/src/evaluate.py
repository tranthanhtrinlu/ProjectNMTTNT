from src.utils import compute_metrics
from config import (
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    DEVICE,
    CLASSES,
    EFFICIENTNET_MODEL_PATH,
)
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import sys
from pathlib import Path

# Thêm ROOT_DIR vào sys.path để import được config, src.utils,...
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


def get_test_loader():
    """
    Tạo DataLoader cho tập test CIFAKE.
    """
    test_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=test_tf)
    print("Test classes:", test_dataset.classes)
    print("Số lượng mẫu test:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )
    return test_loader


def load_efficientnet_model():
    """
    Load EfficientNet-B0 đã train (file .pth) để đánh giá.
    """
    model = timm.create_model(
        "efficientnet_b0", pretrained=False, num_classes=len(CLASSES))
    state_dict = torch.load(EFFICIENTNET_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"✅ Loaded model from: {EFFICIENTNET_MODEL_PATH}")
    return model


def evaluate_model(model, data_loader):
    """
    Chạy model trên tập test, tính Accuracy, F1, AUC-ROC, Confusion Matrix.
    """

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            # Lấy xác suất cho lớp REAL (index = 1, nếu CLASSES = ["FAKE", "REAL"])
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_probs)
    return metrics


def main():
    print("=== ĐÁNH GIÁ MODEL EfficientNet-B0 TRÊN TẬP TEST CIFAKE ===")

    # 1. Tạo DataLoader cho test
    test_loader = get_test_loader()

    # 2. Load model đã train
    model = load_efficientnet_model()

    # 3. Đánh giá
    metrics = evaluate_model(model, test_loader)

    print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    print(f"Accuracy    : {metrics['accuracy']:.4f}")
    print(f"F1-score    : {metrics['f1_score']:.4f}")
    print(f"AUC-ROC     : {metrics['auc_roc']:.4f}")
    print("Confusion matrix (dạng list 2D):")
    print(metrics["confusion_matrix"])
    print("\nClassification report:")
    print(metrics["classification_report"])
    print("\n=== HOÀN THÀNH EVALUATE ===")


if __name__ == "__main__":
    main()
