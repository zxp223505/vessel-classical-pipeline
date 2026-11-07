"""核心函数自检（不依赖 pytest，直接运行）。

覆盖：Dice/AUC 定义、Otsu 阈值、滞后阈值连通性、FOV 掩膜、Frangi 尺度自适应。
运行：python -m tests.test_core
"""
from __future__ import annotations

import sys

import os
import shutil

import cv2
import numpy as np

from src import dl_reference, experiment, io_utils, metrics, segment, vesselness


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise AssertionError(name)


def test_dice() -> None:
    gt = np.zeros((50, 50), np.uint8)
    gt[10:20, 10:20] = 1
    mask = np.ones_like(gt)
    check("Dice: 完全一致 = 1", abs(metrics.dice(gt, gt, mask) - 1.0) < 1e-9)
    other = np.zeros_like(gt)
    other[30:40, 30:40] = 1
    check("Dice: 完全不重叠 = 0", abs(metrics.dice(other, gt, mask)) < 1e-9)
    empty = np.zeros_like(gt)
    check("Dice: 双方都为空 = 1", abs(metrics.dice(empty, empty, mask) - 1.0) < 1e-9)


def test_auc() -> None:
    rng = np.random.default_rng(0)
    score = rng.random(4000).astype(np.float32)
    gt = (rng.random(4000) < 0.3).astype(np.uint8)
    mask = np.ones(4000, np.uint8)
    fast = metrics.auc_score(score, gt, mask)
    # 暴力配对定义
    pos = score[gt == 1]
    neg = score[gt == 0]
    brute = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    check("AUC: 秩统计 == 暴力配对", abs(fast - brute) < 1e-9, f"{fast:.6f} vs {brute:.6f}")


def test_otsu() -> None:
    img = np.full((256, 256), 0.78, np.float32)
    img[80:180, 60:200] = 0.20            # 暗目标，约占 20%
    gt = np.zeros((256, 256), np.uint8)
    gt[80:180, 60:200] = 1
    mask = np.ones_like(gt)
    pred = segment.otsu_global(img, mask)
    check("Otsu: 双峰图上 Dice > 0.95", metrics.dice(pred, gt, mask) > 0.95,
          f"Dice={metrics.dice(pred, gt, mask):.4f}")


def test_hysteresis() -> None:
    score = np.zeros((60, 60), np.float32)
    score[10:20, 10:20] = 0.9             # 有种子
    score[10:20, 30:45] = 0.9             # 有种子
    score[40:50, 40:50] = 0.55            # 只有弱响应，无种子
    mask = np.ones((60, 60), np.uint8)
    out = segment.hysteresis(score, mask, t_low=0.5, t_high=0.8)
    check("滞后阈值: 保留含种子的连通域", out[15, 15] == 1 and out[15, 35] == 1)
    check("滞后阈值: 丢弃无种子的弱区域", out[45, 45] == 0)


def test_fov() -> None:
    img = np.zeros((300, 300), np.uint8)
    cv2.circle(img, (150, 150), 120, 180, -1)
    m = io_utils.fov_mask(img, thresh=20, erode=0)
    area_ratio = m.sum() / (np.pi * 120 ** 2)
    check("FOV: 面积接近圆形理论值", 0.97 < area_ratio < 1.03, f"ratio={area_ratio:.3f}")
    check("FOV: 单一连通域", cv2.connectedComponents(m)[0] == 2)
    m8 = io_utils.fov_mask(img, thresh=20, erode=8)
    m16 = io_utils.fov_mask(img, thresh=20, erode=16)
    check("FOV: 内缩越多面积越小", m8.sum() < m.sum() and m16.sum() < m8.sum(),
          f"{m.sum()} > {m8.sum()} > {m16.sum()}")


def test_sigma_scaling() -> None:
    small = np.zeros((400, 400), np.uint8)
    cv2.circle(small, (200, 200), 150, 1, -1)
    big = np.zeros((800, 800), np.uint8)
    cv2.circle(big, (400, 400), 300, 1, -1)
    s_small = vesselness.sigma_range(small)[-1]
    s_big = vesselness.sigma_range(big)[-1]
    check("Frangi 尺度自适应: σ 随视场直径等比缩放", 1.9 < s_big / s_small < 2.1,
          f"{s_small} -> {s_big}")


def test_fov_transition_band() -> None:
    """回归：视网膜外的渐隐带不能被算进视场（曾导致 Dice 虚高）。"""
    img = np.zeros((240, 240), np.uint8)
    cv2.circle(img, (120, 120), 90, 120, -1)
    ring = np.zeros_like(img)
    cv2.circle(ring, (120, 120), 100, 1, -1)
    cv2.circle(ring, (120, 120), 88, 0, -1)
    img[ring > 0] = 15
    m = io_utils.fov_mask(img, erode=0)
    ys, xs = np.nonzero(m)
    r_max = float(np.sqrt((ys - 120) ** 2 + (xs - 120) ** 2).max())
    check("FOV: 相对阈值排除视网膜外渐隐带", r_max <= 93, f"r_max={r_max:.1f}")


def test_fold_assignment() -> None:
    ids = [str(i) for i in range(1, 21)]
    folds = {k: [x for x in ids if experiment.fold_of(x) == k] for k in range(5)}
    flat = [x for v in folds.values() for x in v]
    check("交叉验证: 5 折各 4 张且覆盖全部图", all(len(v) == 4 for v in folds.values()) and len(flat) == 20)


def test_dl_resolution_filter() -> None:
    """回归：不同分辨率的 OOF 混在一起会导致 np.stack 崩溃。"""
    # 用仓库内的临时目录（沙箱可能不允许写系统 TEMP）
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_dl")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    rng = np.random.default_rng(0)
    for run, size in (("ok", 8), ("mixed", 4)):
        d = os.path.join(root, run, "oof")
        os.makedirs(d)
        np.save(os.path.join(d, "1.npy"), rng.random((size, size)).astype(np.float16))
        gt = np.zeros((size, size), np.uint8)
        gt[2:5, 2:5] = 1
        np.save(os.path.join(d, "1_gt.npy"), gt)
        np.save(os.path.join(d, "1_fov.npy"), np.ones((size, size), np.uint8))
    try:
        rows, summary = dl_reference.collect(root, margin=0, size=8)
        check("DL 对照: 只采用与评测网格同尺寸的 OOF",
              summary["n"] == 1 and summary["skipped"] == 1,
              f"n={summary['n']} skipped={summary['skipped']}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    test_dice()
    test_auc()
    test_otsu()
    test_hysteresis()
    test_fov()
    test_sigma_scaling()
    test_fov_transition_band()
    test_fold_assignment()
    test_dl_resolution_filter()
    print("\n全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
