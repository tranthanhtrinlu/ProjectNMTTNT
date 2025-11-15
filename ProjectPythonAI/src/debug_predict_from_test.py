# ai-service/src/debug_predict_from_test.py
from pathlib import Path
from torchvision import datasets
from config import TEST_DIR
from src.predict import predict_image_bytes

test_ds = datasets.ImageFolder(root=str(TEST_DIR))

print("classes:", test_ds.classes)

# Lấy 5 ảnh đầu tiên trong test
for i in range(5):
    img_path, label = test_ds.samples[i]
    true_label_name = test_ds.classes[label]

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    result = predict_image_bytes(img_bytes)
    print(f"\nFile: {Path(img_path).name}")
    print("True :", true_label_name)
    print("Pred :", result["predicted_label"])
    print("Probs:", result["probabilities"])
