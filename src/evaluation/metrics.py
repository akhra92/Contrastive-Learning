"""
Evaluation metrics for multi-label chest X-ray classification.

Standard metrics for NIH Chest X-ray14 (following the original NIH paper):
  - Per-class AUC-ROC (primary metric)
  - Per-class Average Precision (AP / PR-AUC)
  - Per-class F1 score at a fixed threshold (0.5 by default)
  - Macro-averaged versions of all above
"""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from src.data.dataset import ALL_CLASSES


def tune_thresholds(
    y_true: np.ndarray,
    y_pred_logits: np.ndarray,
    default: float = 0.5,
) -> np.ndarray:
    """
    Find the per-class decision threshold that maximises F1.

    Because training uses BCE with `pos_weight`, the model's logits are not
    calibrated around 0.5, so a fixed 0.5 threshold gives poor F1 even when
    ranking (AUC/AP) is good. Tune this on a held-out split (e.g. the validation
    set) and apply the resulting thresholds to the test set — tuning on the test
    set itself would leak information.

    Returns:
        (C,) array of per-class thresholds (probability space).
    """
    from sklearn.metrics import precision_recall_curve

    if isinstance(y_pred_logits, torch.Tensor):
        y_pred_logits = y_pred_logits.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    y_pred_prob = _sigmoid(y_pred_logits)
    thresholds = np.full(y_true.shape[1], default, dtype=np.float32)

    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        prec, rec, thr = precision_recall_curve(y_true[:, i], y_pred_prob[:, i])
        if len(thr) == 0:
            continue
        # prec/rec have length len(thr)+1; the last point has no threshold.
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
        thresholds[i] = thr[int(np.argmax(f1))]

    return thresholds


def evaluate_multilabel(
    y_true: np.ndarray,
    y_pred_logits: np.ndarray,
    threshold=0.5,
) -> dict:
    """
    Compute per-class and macro-averaged metrics for multi-label classification.

    Args:
        y_true        : (N, C) binary ground-truth labels.
        y_pred_logits : (N, C) raw logits (not probabilities).
        threshold     : decision threshold for F1. Either a scalar applied to all
                        classes, or a (C,) array of per-class thresholds (e.g.
                        from `tune_thresholds`). AUC-ROC and AP are threshold-free
                        and unaffected.

    Returns:
        dict with per-class {auc_roc, avg_precision, f1, threshold} and
        'macro_auc_roc', 'macro_ap', 'macro_f1'.
    """
    if isinstance(y_pred_logits, torch.Tensor):
        y_pred_logits = y_pred_logits.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    y_pred_prob = _sigmoid(y_pred_logits)
    thr_vec = (
        np.full(y_true.shape[1], threshold, dtype=np.float32)
        if np.isscalar(threshold)
        else np.asarray(threshold, dtype=np.float32)
    )
    y_pred_bin = (y_pred_prob >= thr_vec).astype(int)

    per_class = {}
    auc_list, ap_list, f1_list = [], [], []

    for i, cls_name in enumerate(ALL_CLASSES):
        n_pos = int(y_true[:, i].sum())
        if n_pos == 0 or n_pos == len(y_true):
            # Skip degenerate classes (no positive or all positive)
            continue
        auc = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
        ap = average_precision_score(y_true[:, i], y_pred_prob[:, i])
        f1 = f1_score(y_true[:, i], y_pred_bin[:, i], zero_division=0)

        per_class[cls_name] = {
            "auc_roc": auc,
            "avg_precision": ap,
            "f1": f1,
            "threshold": float(thr_vec[i]),
        }
        auc_list.append(auc)
        ap_list.append(ap)
        f1_list.append(f1)

    per_class["macro_auc_roc"] = float(np.mean(auc_list)) if auc_list else 0.0
    per_class["macro_ap"] = float(np.mean(ap_list)) if ap_list else 0.0
    per_class["macro_f1"] = float(np.mean(f1_list)) if f1_list else 0.0

    return per_class


def print_metrics(metrics: dict):
    """Pretty-print per-class and aggregate metrics."""
    print("\n" + "=" * 70)
    print(f"{'Class':<25} {'AUC-ROC':>8} {'Avg-Prec':>10} {'F1':>8} {'Thresh':>8}")
    print("-" * 70)
    for cls_name in ALL_CLASSES:
        if cls_name in metrics:
            m = metrics[cls_name]
            print(
                f"{cls_name:<25} {m['auc_roc']:>8.4f} {m['avg_precision']:>10.4f}"
                f" {m['f1']:>8.4f} {m.get('threshold', 0.5):>8.3f}"
            )
    print("=" * 70)
    print(f"{'Macro AUC-ROC':<25} {metrics.get('macro_auc_roc', 0):>8.4f}")
    print(f"{'Macro Avg-Precision':<25} {metrics.get('macro_ap', 0):>8.4f}")
    print(f"{'Macro F1':<25} {metrics.get('macro_f1', 0):>8.4f}")
    print("=" * 70 + "\n")


def collect_predictions(model, loader, device) -> tuple:
    """
    Run the model over `loader` and collect ground-truth labels and logits.

    Returns:
        y_true        : (N, C) numpy float32 array
        y_pred_logits : (N, C) numpy float32 array
    """
    model.eval()
    all_labels, all_logits = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_labels, axis=0), np.concatenate(all_logits, axis=0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
