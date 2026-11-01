"""四个对照方案 C0–C3 的统一入口（C0 基线 → C3 加后处理）。"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np

from . import enhance, postprocess as pp, segment, vesselness

ENH_C1 = {"channel": "green", "border": False, "clahe": True, "clip": 2.0, "grid": 8}
ENH_C2 = {"channel": "green", "border": False, "top_hat": 51, "clahe": True, "clip": 2.0, "grid": 8}

CONFIGS: dict[str, dict] = {
    "C0": {
        "label": "C0 基线：灰度 + 全局 Otsu",
        "enhance": {"channel": "gray", "border": True, "clahe": False},
        "method": "otsu",
        "default": {},
    },
    "C1": {
        "label": "C1 +预处理：绿通道 + CLAHE + 局部自适应阈值",
        "enhance": ENH_C1,
        "method": "adaptive",
        "default": {"block": 21, "c": 12.0},
    },
    "C2": {
        "label": "C2 +血管增强：Frangi(自适应σ,逐尺度归一化) + 滞后阈值",
        "enhance": ENH_C2,
        "method": "frangi",
        "default": {"q_low": 0.90, "q_high": 0.99, "s_min": 1.0, "s_max_ratio": 0.006},
    },
    "C3": {
        "label": "C3 +后处理：在 C2 上加面积/填孔 + 膨胀补偿",
        "enhance": ENH_C2,
        "method": "frangi",
        "default": {"q_low": 0.90, "q_high": 0.99, "s_min": 1.0, "s_max_ratio": 0.006},
        "post": {"min_area": 40, "fill_holes": True, "max_hole": 40, "dilate": 1},
    },
}


def _frangi_cache_path(cache_dir: str, idx: str, enh: dict, sigmas: list[float], fov: np.ndarray) -> str:
    """由影像编号、预处理配置、实际 σ 列表与视场掩膜生成缓存文件名。"""
    mask_sig = hashlib.md5(np.ascontiguousarray(fov).tobytes()).hexdigest()[:8]
    raw = f"{idx}|{sorted(enh.items())}|{[round(s, 4) for s in sigmas]}|{mask_sig}"
    key = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"frangi_{idx}_{key}.npy")


def _frangi(sample, enh_cfg: dict, params: dict, cache_dir: str | None) -> np.ndarray:
    sig = vesselness.sigma_range(
        sample.fov, float(params.get("s_min", 1.0)), float(params.get("s_max_ratio", 0.006)), n=6
    )
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = _frangi_cache_path(cache_dir, sample.idx, enh_cfg, sig, sample.fov)
        if os.path.exists(path):
            return np.load(path)
    img = enhance.prepare(sample.bgr, sample.fov, enh_cfg)
    resp = vesselness.vesselness(img, sample.fov, sig)
    if cache_dir:
        np.save(path, resp)
    return resp


def run(sample, key: str, params: dict | None = None, cache_dir: str | None = None) -> dict:
    """跑一个方案，返回 {'binary', 'score', 'seconds'}。"""
    cfg = CONFIGS[key]
    p = {**cfg.get("default", {}), **(params or {})}
    t0 = time.perf_counter()

    if cfg["method"] == "otsu":
        img = enhance.prepare(sample.bgr, sample.fov, cfg["enhance"])
        binary = segment.otsu_global(img, sample.fov)
        score = 1.0 - img
    elif cfg["method"] == "adaptive":
        img = enhance.prepare(sample.bgr, sample.fov, cfg["enhance"])
        binary = segment.adaptive_local(img, sample.fov, int(p["block"]), float(p["c"]))
        score = 1.0 - img
    elif cfg["method"] == "frangi":
        resp = _frangi(sample, cfg["enhance"], p, cache_dir)
        t_low, t_high = segment.quantile_thresholds(resp, sample.fov, float(p["q_low"]), float(p["q_high"]))
        binary = segment.hysteresis(resp, sample.fov, t_low, t_high)
        score = resp
    else:
        raise ValueError(cfg["method"])

    if cfg.get("post"):
        post_cfg = {**cfg["post"], **{k: v for k, v in p.items() if k in
                                      ("min_area", "min_elongation", "fill_holes", "max_hole", "dilate")}}
        binary = pp.postprocess(binary, sample.fov, post_cfg)

    return {
        "binary": binary.astype(np.uint8),
        "score": score.astype(np.float32),
        "seconds": time.perf_counter() - t0,
    }
