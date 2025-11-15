from torchvision import datasets
from config import TRAIN_DIR

ds = datasets.ImageFolder(root=str(TRAIN_DIR))
print("ImageFolder classes:", ds.classes)
