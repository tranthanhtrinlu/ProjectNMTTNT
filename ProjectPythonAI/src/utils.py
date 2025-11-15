# ai-service/src/utils.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from config import CLASSES


def compute_metrics(y_true, y_prob):
    """
    y_true: list[int]  (0=FAKE, 1=REAL)
    y_prob: list[float]  xác suất dự đoán là class=1 (REAL)
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)

    return {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "auc_roc": float(auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
