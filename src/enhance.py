"""图像增强模块（对应课件第 3 章：灰度映射、直方图修正、形态学背景校正）。

每个函数只做一件事，便于在消融实验里单独开关：
    C0 : bgr_to_gray
    C1 : green_channel + border_fill + clahe
"""
from __future__ import annotations

import cv2
import numpy as np


def bgr_to_gray(bgr: np.ndarray) -> np.ndarray:
    """三通道等权平均（课件 3.1 灰度映射的最简形式）。"""
    return bgr.astype(np.float32).mean(axis=2)


def green_channel(bgr: np.ndarray) -> np.ndarray:
    """取绿色通道。

    眼底图中血红蛋白对绿光吸收最强，血管与背景的对比度在 G 通道最高，
    是视网膜血管分割最常用的第一步。
    """
    return bgr[:, :, 1].astype(np.float32)


def normalize01(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """按掩膜内 1%/99% 分位拉伸到 [0,1]，抑制极值影响。"""
    x = img.astype(np.float32)
    vals = x[mask > 0] if mask is not None and mask.any() else x.ravel()
    lo, hi = np.percentile(vals, 1.0), np.percentile(vals, 99.0)
    if hi - lo < 1e-6:
        return np.clip(x, 0, 1)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def border_fill(img: np.ndarray, fov: np.ndarray, band: int = 8) -> np.ndarray:
    """把视场外像素替换成边界内侧一圈的中值。

    这一步解决一个具体问题：视场边缘是"视网膜—纯黑"的强阶跃，任何基于
    导数/局部均值的算子都会在这里产生一条高响应圆环，进而把阈值拉偏。
    把外圈填成边界亮度后，阶跃消失，伪圆环随之消失。
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    inner = cv2.erode(fov.astype(np.uint8), k).astype(bool)
    ring = (fov > 0) & (~inner)
    out = img.copy()
    if ring.any():
        out[fov == 0] = float(np.median(img[ring]))
    return out


def clahe(img01: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """限制对比度的自适应直方图均衡（CLAHE）。

    与全局直方图均衡相比，CLAHE 只在局部窗口内均衡，能拉高血管与背景的
    局部对比度；clip_limit 限制每个直方图 bin 的高度，抑制噪声过增强。
    """
    u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(grid), int(grid)))
    return op.apply(u8).astype(np.float32) / 255.0


def top_hat(img01: np.ndarray, ksize: int = 51) -> np.ndarray:
    """形态学顶帽：保留小于结构元的亮细节（本项目中作为对比分支的预处理）。"""
    u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, k).astype(np.float32) / 255.0


def prepare(bgr: np.ndarray, mask: np.ndarray, cfg: dict) -> np.ndarray:
    """生成送进分割器的增强图（float32，[0,1]）。

    cfg 键：
        channel : 'gray' | 'green'
        clahe   : bool   是否 CLAHE
        clip    : float  CLAHE 截断系数
        top_hat : int    若 >0 则先做顶帽（对比分支用）
        border  : bool   是否做边界填充（默认 True）
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
