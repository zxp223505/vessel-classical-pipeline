"""分割：全局 Otsu、局部自适应阈值、滞后阈值。"""

from __future__ import annotations

import cv2
import numpy as np


def _to_u8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)


def otsu_global(img01: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """全局 Otsu 阈值，取暗类为前景。"""
    u8 = _to_u8(img01)
    vals = u8[mask > 0]
    if vals.size == 0:
        return np.zeros_like(u8, np.uint8)
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    total = hist.sum()
    prob = hist / max(total, 1)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = -1.0
    best = float(sigma_b.max())
    if best <= 0:
        return np.zeros_like(u8, np.uint8)
    cand = np.flatnonzero(sigma_b >= best - 1e-12)
    t = int(round(cand.mean()))
    t = max(1, min(255, t))
    return ((u8 < t) & (mask > 0)).astype(np.uint8)


def adaptive_local(img01: np.ndarray, mask: np.ndarray, block: int = 31, c: float = 6.0) -> np.ndarray:
    """局部自适应阈值分割。

    block: 邻域窗口边长，内部会强制为奇数；c: 与邻域均值的差值偏移。
    """
    u8 = _to_u8(img01)
    block = int(block) | 1
    binv = cv2.adaptiveThreshold(
        u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, float(c)
    )
    return ((binv > 0) & (mask > 0)).astype(np.uint8)


def hysteresis(score: np.ndarray, mask: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """滞后阈值：以 t_high 为种子、t_low 为生长下限，保留与种子连通的弱响应。"""
    weak = ((score >= t_low) & (mask > 0)).astype(np.uint8)
    strong = ((score >= t_high) & (mask > 0)).astype(np.uint8)
    if weak.sum() == 0:
        return weak
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    if n <= 1:
        return np.zeros_like(weak)
    keep = np.zeros(n, dtype=bool)
    seeds = np.unique(lab[strong > 0])
    keep[seeds] = True
    keep[0] = False
    return keep[lab].astype(np.uint8)


def quantile_thresholds(score: np.ndarray, mask: np.ndarray, q_low: float, q_high: float) -> tuple[float, float]:
    """把分位数换算成绝对阈值。"""
    vals = score[mask > 0]
    if vals.size == 0:
        return 1.0, 1.0
    return float(np.quantile(vals, q_low)), float(np.quantile(vals, q_high))
