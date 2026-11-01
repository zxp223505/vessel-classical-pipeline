"""后处理：面积/长宽比过滤、填孔、膨胀、视场裁剪。"""

from __future__ import annotations

import cv2
import numpy as np


def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """去掉面积小于 min_area 的连通域。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if n <= 1:
        return binary
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(min_area)
    return keep[lab].astype(np.uint8)


def remove_blobs(binary: np.ndarray, min_elongation: float = 2.0) -> np.ndarray:
    """去掉非细长且填充率高的块状区域。

    min_elongation: 长宽比的保留下限，小于它且填充率 >0.6 的区域被移除。
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if n <= 1:
        return binary
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        w = max(stats[i, cv2.CC_STAT_WIDTH], 1)
        h = max(stats[i, cv2.CC_STAT_HEIGHT], 1)
        area = stats[i, cv2.CC_STAT_AREA]
        elong = max(w, h) / min(w, h)
        filled = area / float(w * h)
        keep[i] = (elong >= min_elongation) or (filled <= 0.6)
    return keep[lab].astype(np.uint8)


def fill_small_holes(binary: np.ndarray, max_hole: int = 40) -> np.ndarray:
    """填充面积不超过 max_hole、且不触碰图像边界的孔洞。"""
    b = binary.astype(np.uint8)
    inv = (1 - b).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = b.copy()
    h, w = b.shape
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if x == 0 or y == 0 or x + ww >= w or y + hh >= h:
            continue
        if area <= max_hole:
            out[lab == i] = 1
    return out


def clip_to_fov(binary: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """按视场掩膜裁剪预测。"""
    return (binary.astype(np.uint8) * (fov > 0)).astype(np.uint8)


def dilate(binary: np.ndarray, r: int = 1) -> np.ndarray:
    """椭圆核膨胀，r 为半径（像素）。"""
    if r <= 0:
        return binary.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(binary.astype(np.uint8), k)


def postprocess(binary: np.ndarray, fov: np.ndarray, cfg: dict) -> np.ndarray:
    """按 cfg 依次执行后处理。

    cfg 可用键：min_area、min_elongation、fill_holes、max_hole、dilate。
    """
    out = binary.astype(np.uint8)
    if cfg.get("min_area", 0):
        out = remove_small_components(out, int(cfg["min_area"]))
    if cfg.get("min_elongation", 0):
        out = remove_blobs(out, float(cfg["min_elongation"]))
    if cfg.get("fill_holes", False):
        out = fill_small_holes(out, int(cfg.get("max_hole", 40)))
    out = clip_to_fov(out, fov)
    if cfg.get("dilate", 0):
        out = dilate(out, int(cfg["dilate"]))
        out = clip_to_fov(out, fov)
    return out
