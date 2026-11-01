"""分割模块（对应课件第 4 章：阈值分割、区域生长思想）。

提供三种分割器：
    otsu_global        : 全局 Otsu（课件 4.2.1 最大类间方差阈值）
    adaptive_local     : 局部自适应阈值（课件 4.2.1 依赖坐标的阈值）
    hysteresis         : 双阈值 + 连通域生长的滞后阈值（本项目主分割器）
"""
from __future__ import annotations

import cv2
import numpy as np


def _to_u8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)


def otsu_global(img01: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """全局 Otsu：在视场内找最大类间方差阈值，血管为暗像素（取 < t）。"""
    u8 = _to_u8(img01)
    vals = u8[mask > 0]
    if vals.size == 0:
        return np.zeros_like(u8, np.uint8)
    # 只用视场内像素估计阈值，避免四角纯黑像素污染直方图
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
    # 关键细节：两峰之间那一段的类间方差完全相等（直方图为空），argmax 会取到平台左端，
    # 于是阈值落在暗峰上、把目标全部排除。取平台中点才是教科书的“峰谷之间”。
    best = float(sigma_b.max())
    if best <= 0:
        return np.zeros_like(u8, np.uint8)
    cand = np.flatnonzero(sigma_b >= best - 1e-12)
    t = int(round(cand.mean()))
    t = max(1, min(255, t))
    return ((u8 < t) & (mask > 0)).astype(np.uint8)


def adaptive_local(img01: np.ndarray, mask: np.ndarray, block: int = 31, c: float = 6.0) -> np.ndarray:
    """局部自适应阈值：每个像素与邻域高斯加权均值比较，血管更暗则判为前景。

    对应课件"依赖坐标/局部性质的阈值选取"，对光照不均比全局阈值稳健。
    """
    u8 = _to_u8(img01)
    block = int(block) | 1  # 必须为奇数
    binv = cv2.adaptiveThreshold(
        u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, float(c)
    )
    return ((binv > 0) & (mask > 0)).astype(np.uint8)


def hysteresis(score: np.ndarray, mask: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """滞后阈值（双阈值 + 连通域生长）。

    低于 t_low 的像素直接丢弃；高于 t_high 的像素作为"种子"；
    与种子连通的弱响应像素（≥ t_low）被保留。这样既保住细血管的连续性，
    又不会把整片弱响应噪声连成一片。
    """
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
    """把"分位数"参数换算成绝对阈值（对亮度/尺度变化更稳健）。"""
    vals = score[mask > 0]
    if vals.size == 0:
        return 1.0, 1.0
    return float(np.quantile(vals, q_low)), float(np.quantile(vals, q_high))
