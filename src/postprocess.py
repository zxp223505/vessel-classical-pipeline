"""后处理模块（对应课件第 3、4 章：形态学、连通域分析、轮廓处理）。

后处理只做"去伪"和"补漏"两类事，全部按连通域操作，便于在报告里逐条描述：
    1) 面积过滤   ：血管是细长结构，孤立的小斑点多为噪声/病灶
    2) 长宽比过滤 ：剔除块状伪响应（如视盘边缘、渗出）
    3) 孔洞填充   ：血管内部偶尔出现断点，填掉被前景包围的小空洞
    4) 视场裁剪   ：边界处的黑色过渡带常被误判，用内缩的 FOV 掩膜裁掉
"""
from __future__ import annotations

import cv2
import numpy as np


def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if n <= 1:
        return binary
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(min_area)
    return keep[lab].astype(np.uint8)


def remove_blobs(binary: np.ndarray, min_elongation: float = 2.0) -> np.ndarray:
    """按长宽比剔除块状区域：血管细长，宽高比接近 1 的大块多半不是血管。"""
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
        # 细长（或细线状）且填充率低的结构才保留
        keep[i] = (elong >= min_elongation) or (filled <= 0.6)
    return keep[lab].astype(np.uint8)


def fill_small_holes(binary: np.ndarray, max_hole: int = 40) -> np.ndarray:
    """填掉被前景完全包围且面积小于 max_hole 的小孔。"""
    b = binary.astype(np.uint8)
    inv = (1 - b).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = b.copy()
    h, w = b.shape
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        # 触碰边界的连通域是背景，不是孔洞
        if x == 0 or y == 0 or x + ww >= w or y + hh >= h:
            continue
        if area <= max_hole:
            out[lab == i] = 1
    return out


def clip_to_fov(binary: np.ndarray, fov: np.ndarray) -> np.ndarray:
    return (binary.astype(np.uint8) * (fov > 0)).astype(np.uint8)


def dilate(binary: np.ndarray, r: int = 1) -> np.ndarray:
    """膨胀补偿：Frangi/自适应阈值给出的血管偏细（细血管常只剩 1px），
    膨胀 1 圈可把"略细+位置正确"的预测补回真实宽度。"""
    if r <= 0:
        return binary.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(binary.astype(np.uint8), k)


def postprocess(binary: np.ndarray, fov: np.ndarray, cfg: dict) -> np.ndarray:
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
