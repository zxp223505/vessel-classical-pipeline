"""图像增强：灰度/绿通道、归一化、CLAHE、顶帽、边界填充。"""

from __future__ import annotations

import cv2
import numpy as np


def bgr_to_gray(bgr: np.ndarray) -> np.ndarray:
    """三通道等权平均。"""
    return bgr.astype(np.float32).mean(axis=2)


def green_channel(bgr: np.ndarray) -> np.ndarray:
    """取绿色通道。"""
    return bgr[:, :, 1].astype(np.float32)


def normalize01(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """按掩膜内 1%/99% 分位拉伸到 [0,1]。"""
    x = img.astype(np.float32)
    vals = x[mask > 0] if mask is not None and mask.any() else x.ravel()
    lo, hi = np.percentile(vals, 1.0), np.percentile(vals, 99.0)
    if hi - lo < 1e-6:
        return np.clip(x, 0, 1)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def border_fill(img: np.ndarray, fov: np.ndarray, band: int = 8) -> np.ndarray:
    """用边界内侧一圈的中值填充视场外像素。

    band: 取中值所用的内圈宽度（像素）。默认配置未启用（消融显示略降）。
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    inner = cv2.erode(fov.astype(np.uint8), k).astype(bool)
    ring = (fov > 0) & (~inner)
    out = img.copy()
    if ring.any():
        out[fov == 0] = float(np.median(img[ring]))
    return out


def clahe(img01: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """限制对比度的自适应直方图均衡。"""
    u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(grid), int(grid)))
    return op.apply(u8).astype(np.float32) / 255.0


def top_hat(img01: np.ndarray, ksize: int = 51) -> np.ndarray:
    """形态学顶帽（保留小于结构元的亮细节）。"""
    u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, k).astype(np.float32) / 255.0


def prepare(bgr: np.ndarray, mask: np.ndarray, cfg: dict) -> np.ndarray:
    """按 cfg 生成 [0,1] 的增强图。

    cfg 键（除 channel 外均可省略）：
        channel  'gray' | 'green'
        clahe    是否做 CLAHE；clip 截断系数；grid 分块数
        top_hat  结构元尺寸，>0 时先做顶帽
        border   是否做边界填充（默认 True）
    """
    img = green_channel(bgr) if cfg.get("channel", "gray") == "green" else bgr_to_gray(bgr)
    img = normalize01(img, mask)
    if cfg.get("border", True):
        img = border_fill(img, mask)
    if cfg.get("top_hat", 0):
        img = top_hat(img, int(cfg["top_hat"]))
        img = normalize01(img, mask)
    if cfg.get("clahe", False):
        img = clahe(img, float(cfg.get("clip", 2.0)), int(cfg.get("grid", 8)))
        img = normalize01(img, mask)
    return img.astype(np.float32)
