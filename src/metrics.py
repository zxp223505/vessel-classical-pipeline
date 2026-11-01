"""评价指标：Dice、Se、Sp、Acc、AUC（均在视场掩膜内计算）。"""

from __future__ import annotations

import numpy as np


def confusion(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    """返回视场内的 (TP, FP, FN, TN)。"""
    m = mask > 0
    p = (pred[m] > 0).astype(np.uint8)
    g = (gt[m] > 0).astype(np.uint8)
    tp = float(np.sum((p == 1) & (g == 1)))
    fp = float(np.sum((p == 1) & (g == 0)))
    fn = float(np.sum((p == 0) & (g == 1)))
    tn = float(np.sum((p == 0) & (g == 0)))
    return tp, fp, fn, tn


def dice(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Dice 系数；预测与真值都为空记为 1.0。"""
    tp, fp, fn, _ = confusion(pred, gt, mask)
    if tp + fp + fn == 0:
        return 1.0
    return float(2 * tp / (2 * tp + fp + fn))


def sensitivity(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """敏感度（召回）。"""
    tp, _, fn, _ = confusion(pred, gt, mask)
    return float(tp / (tp + fn + 1e-9))


def specificity(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """特异度。"""
    _, fp, _, tn = confusion(pred, gt, mask)
    return float(tn / (tn + fp + 1e-9))


def accuracy(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """像素准确率。"""
    tp, fp, fn, tn = confusion(pred, gt, mask)
    return float((tp + tn) / (tp + fp + fn + tn + 1e-9))


def auc_score(score: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """基于秩的 AUC，并列值取平均秩。"""
    m = mask > 0
    s = score[m].astype(np.float64)
    y = (gt[m] > 0).astype(np.uint8)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if uniq.size < s.size:
        sums = np.zeros(uniq.size, dtype=np.float64)
        np.add.at(sums, inv, ranks)
        avg = sums / cnt
        ranks = avg[inv]
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def all_metrics(pred: np.ndarray, score: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    """一次返回全部指标与预测/真值的前景占比。"""
    return {
        "dice": dice(pred, gt, mask),
        "se": sensitivity(pred, gt, mask),
        "sp": specificity(pred, gt, mask),
        "acc": accuracy(pred, gt, mask),
        "auc": auc_score(score, gt, mask),
        "pred_ratio": float(pred[mask > 0].mean()) if mask.any() else 0.0,
        "gt_ratio": float(gt[mask > 0].mean()) if mask.any() else 0.0,
    }
