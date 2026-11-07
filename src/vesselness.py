"""多尺度 Frangi 血管增强（Hessian 滤波）。"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

CACHE_VERSION = 1


def _hessian_eigenvalues(img: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """返回按 |λ| 从大到小排列的两个 Hessian 特征值 (l2, l1)。"""
    hxx = ndi.gaussian_filter(img, sigma, order=(2, 0), mode="nearest") * (sigma ** 2)
    hyy = ndi.gaussian_filter(img, sigma, order=(0, 2), mode="nearest") * (sigma ** 2)
    hxy = ndi.gaussian_filter(img, sigma, order=(1, 1), mode="nearest") * (sigma ** 2)

    tmp = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4.0 * hxy ** 2, 0.0))
    mu1 = 0.5 * (hxx + hyy + tmp)
    mu2 = 0.5 * (hxx + hyy - tmp)
    swap = np.abs(mu1) > np.abs(mu2)
    l1 = np.where(swap, mu2, mu1)
    l2 = np.where(swap, mu1, mu2)
    return l1, l2


def vesselness(
    img01: np.ndarray,
    mask: np.ndarray,
    sigmas: list[float],
    beta: float = 0.5,
    c_ratio: float = 0.5,
    per_scale_norm: bool = True,
    stat_erode: int = 12,
) -> np.ndarray:
    """返回 [0,1] 的血管响应图，img01 中血管为暗结构。

    beta: Rb 项敏感度；c_ratio: 结构强度归一化系数（乘该尺度 S 的 99.5% 分位）；
    per_scale_norm: 是否逐尺度归一化后融合；stat_erode: 统计量所用掩膜的内缩像素。
    """
    inv = 1.0 - img01
    m = mask > 0
    sm = ndi.binary_erosion(m, iterations=int(stat_erode)) if stat_erode else m
    if not sm.any():
        sm = m
    resp = np.zeros_like(inv, dtype=np.float32)
    for s in sigmas:
        l1, l2 = _hessian_eigenvalues(inv, float(s))
        valid = l2 < 0
        rb = np.zeros_like(l1)
        np.divide(np.abs(l1), np.abs(l2) + 1e-12, out=rb, where=np.abs(l2) > 1e-12)
        s_norm = np.sqrt(l1 ** 2 + l2 ** 2)
        c = float(c_ratio * np.percentile(s_norm[sm], 99.5)) + 1e-12
        v = np.exp(-(rb ** 2) / (2.0 * beta ** 2)) * (1.0 - np.exp(-(s_norm ** 2) / (2.0 * c ** 2)))
        v = np.where(valid, v, 0.0).astype(np.float32)
        if per_scale_norm:
            v = np.clip(v / (np.percentile(v[sm], 99.5) + 1e-12), 0.0, 1.0)
        resp = np.maximum(resp, v)
    return (np.clip(resp, 0.0, 1.0) * m).astype(np.float32)


def sigma_range(fov: np.ndarray, s_min: float = 1.0, s_max_ratio: float = 0.006, n: int = 6) -> list[float]:
    """按视场直径生成尺度列表，s_max = s_max_ratio × 直径。"""
    ys, xs = np.nonzero(fov)
    if ys.size == 0:
        return [round(float(x), 2) for x in np.linspace(s_min, 3.0, n)]
    d = float(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    s_max = max(s_min + 0.5, s_max_ratio * d)
    return [round(float(x), 2) for x in np.linspace(s_min, s_max, n)]
