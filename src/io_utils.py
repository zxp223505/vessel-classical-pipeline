"""图像读写与视场掩膜（FOV mask）。

数据约定（与课程数据一致）：
    train/image/*.jpg  彩色眼底图（BGR，584x565）
    train/label/*.jpg  二值血管标注，血管=255，背景=0
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class Sample:
    idx: str
    bgr: np.ndarray        # 原始彩色图（resize 后）
    label: Optional[np.ndarray]  # 0/1 uint8，测试集为 None
    fov: np.ndarray        # 0/1 uint8 视场掩膜


def imread_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imread_label(path: str) -> np.ndarray:
    lab = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if lab is None:
        raise FileNotFoundError(path)
    return (lab > 127).astype(np.uint8)


def _largest_component(m: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m.astype(np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


def fov_mask(gray: np.ndarray, thresh: int = 8, erode: int = 8, rel: float = 0.35) -> np.ndarray:
    """估计圆形视网膜视场掩膜。

    先用绝对阈值取粗掩膜，再以掩膜内部中位亮度乘以 rel 作相对阈值细化，最后内缩 erode 像素。
    thresh: 绝对阈值下限；rel: 相对阈值系数；erode: 边缘内缩像素。
    """
    m = (gray > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    cv2.floodFill(ff, None, (0, 0), 2)
    holes = ((ff != 2).astype(np.uint8))[1:-1, 1:-1]
    m = np.clip(m + holes, 0, 1).astype(np.uint8)
    m = _largest_component(m)

    inner = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    if inner.any():
        med = float(np.median(gray[inner > 0]))
        t = max(float(thresh), float(rel) * med)
        m2 = ((gray > t).astype(np.uint8) * m)
        m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        m = _largest_component(m2)

    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode * 2 + 1, erode * 2 + 1))
        m = cv2.erode(m, k)
    return m


def resize_pair(bgr: np.ndarray, label: Optional[np.ndarray], size: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    bgr_r = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    if label is None:
        return bgr_r, None
    lab_r = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
    return bgr_r, lab_r


def load_dataset(data_dir: str, split: str, size: int = 512, ids: Optional[list[str]] = None) -> list[Sample]:
    """读取一个 split；split in {'train', 'test'}。"""
    img_dir = os.path.join(data_dir, split, "image")
    lab_dir = os.path.join(data_dir, split, "label")
    if ids is None:
        ids = sorted(
            (f.rsplit(".", 1)[0] for f in os.listdir(img_dir)),
            key=lambda s: int(s) if s.isdigit() else s,
        )
    out: list[Sample] = []
    for i in ids:
        bgr = imread_bgr(os.path.join(img_dir, i + ".jpg"))
        lab_path = os.path.join(lab_dir, i + ".jpg")
        label = imread_label(lab_path) if os.path.exists(lab_path) else None
        bgr, label = resize_pair(bgr, label, size)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        out.append(Sample(idx=i, bgr=bgr, label=label, fov=fov_mask(gray)))
    return out
