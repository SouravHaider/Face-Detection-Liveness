"""Evaluation metrics.

Detection:  IoU, mean IoU, precision/recall at an IoU threshold.
Liveness/anti-spoofing:  APCER, BPCER, ACER (the ISO/IEC 30107-3 standard
presentation-attack-detection metrics).
"""
import numpy as np


def iou(box_a, box_b):
    """Intersection-over-Union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mean_iou(pred_boxes, true_boxes):
    vals = [iou(p, t) for p, t in zip(pred_boxes, true_boxes)]
    return float(np.mean(vals)) if vals else 0.0


def detection_precision_recall(pred_boxes, true_boxes, thr=0.5):
    """Precision/recall treating IoU >= thr as a true positive."""
    tp = sum(1 for p, t in zip(pred_boxes, true_boxes) if iou(p, t) >= thr)
    n = len(true_boxes)
    precision = tp / len(pred_boxes) if pred_boxes else 0.0
    recall = tp / n if n else 0.0
    return precision, recall


def anti_spoof_metrics(y_true, y_pred):
    """APCER / BPCER / ACER.

    Convention: 1 = bona fide (live), 0 = attack (spoof).
      APCER = proportion of attacks misclassified as live
      BPCER = proportion of live faces misclassified as attack
      ACER  = (APCER + BPCER) / 2
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    attacks = y_true == 0
    bonafide = y_true == 1
    apcer = float(np.mean(y_pred[attacks] == 1)) if attacks.any() else 0.0
    bpcer = float(np.mean(y_pred[bonafide] == 0)) if bonafide.any() else 0.0
    return {"APCER": apcer, "BPCER": bpcer, "ACER": (apcer + bpcer) / 2}


def confusion(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
