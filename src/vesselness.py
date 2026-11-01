"""多尺度 Frangi 血管增强（Hessian 滤波）。

原理（课件 3.6 图像锐化 / 4.3 边缘检测中的二阶微分思想）：
血管在垂直于走向的方向上是"暗谷"，该方向上二阶导数取较大正值。对图像求
Hessian H = [[Ixx, Ixy], [Ixy, Iyy]]，其特征值满足 |λ1| ≤ |λ2| 时，管状结构
表现为 λ2 幅值大且符号稳定；用 Rb、S 两个度量组合出血管相似度：

    V = exp(-Rb²/2β²) · (1 - exp(-S²/2c²)),   Rb = |λ1|/|λ2|,  S = √(λ1²+λ2²)

本项目相对"教科书版 Frangi"的两处改动：
    1) 尺度自适应：σ 上限按视场直径等比缩放，换分辨率/换相机不用改代码；
    2) 逐尺度归一化融合：每个尺度先各自归一化再取 max，避免大尺度响应
       把细血管淹没（同时把统计量限制在内缩的视场内，避开边缘）。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _hessian_eigenvalues(img: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """用高斯导数计算 Hessian 的两个特征值（按 |λ| 从大到小返回 l2, l1）。"""
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
    """返回 [0,1] 的血管响应图；img01 中血管为"暗"结构。"""
    inv = 1.0 - img01  # 暗血管 → 亮脊，套用标准 Frangi 的"亮结构"约定
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
    """按视场直径推算尺度列表（改动点 1：尺度自适应）。

    s_max = s_max_ratio × 视场直径。512×512 下视场直径约 490px，
    默认 0.006 → σ_max ≈ 3px，覆盖 1–3px 宽的血管；分辨率变化时自动缩放。
    """
    ys, xs = np.nonzero(fov)
    if ys.size == 0:
        return [round(float(x), 2) for x in np.linspace(s_min, 3.0, n)]
    # 视场直径：取包围盒长宽中的较大者（用对角线会高估 1.41 倍）
    d = float(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    s_max = max(s_min + 0.5, s_max_ratio * d)
    return [round(float(x), 2) for x in np.linspace(s_min, s_max, n)]
