# ai-service/src/train_efficientnet.py
import sys
from pathlib import Path

# Thêm đường dẫn gốc để import được config, utils
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

from config import (
    TRAIN_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LR,
    DEVICE,
    EFFICIENTNET_MODEL_PATH,
    CLASSES,
)
from src.utils import compute_metrics


def get_dataloaders():
    # Augmentation + normalize
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_tf)
    test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=test_tf)

    print("Train classes:", train_dataset.classes)
    print("Test  classes:", test_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, test_loader


def build_model():
    # EfficientNet-B0 từ timm
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(CLASSES))
    return model.to(DEVICE)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Lấy xác suất class REAL (index=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / len(data_loader.dataset)
    metrics = compute_metrics(all_labels, all_probs)

    return avg_loss, metrics


def train():
    train_loader, test_loader = get_dataloaders()
    model = build_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_f1 = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Đánh giá trên tập test
        val_loss, metrics = evaluate(model, test_loader, criterion)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(
            f"Val loss: {val_loss:.4f}, "
            f"Acc: {metrics['accuracy']:.4f}, "
            f"F1: {metrics['f1_score']:.4f}, "
            f"AUC: {metrics['auc_roc']:.4f}"
        )
        print("Classification report:\n", metrics["classification_report"])

        # Lưu model tốt nhất theo F1
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            torch.save(model.state_dict(), EFFICIENTNET_MODEL_PATH)
            print(f"✅ Lưu model tốt nhất tại {EFFICIENTNET_MODEL_PATH}, F1 = {best_f1:.4f}")

    print("Huấn luyện xong!")


if __name__ == "__main__":
    train()
