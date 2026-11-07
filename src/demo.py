"""单图演示：跑完 C0–C3、打印逐级指标并存一张对比图。

用法：python -m src.demo --data-dir <数据目录> --image 5 --out results
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import enhance, io_utils, metrics, pipeline, vesselness  # noqa: E402


def overlay(img01: np.ndarray, gt: np.ndarray | None, pred: np.ndarray | None) -> np.ndarray:
    """在增强图上叠加标注（绿）与预测（红）。"""
    base = (np.clip(img01 * 255, 0, 255).astype(np.uint8)[:, :, None].repeat(3, axis=2) * 0.75).astype(np.uint8)
    if gt is not None:
        base[gt > 0] = (60, 200, 60)
    if pred is not None:
        base[pred > 0] = (255, 60, 60)
    return base


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("VESSEL_DATA", "data/segmentation"))
    ap.add_argument("--image", default="5", help="图像编号（train/image 下的文件名）")
    ap.add_argument("--out", default="results")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args(argv)

    sample = io_utils.load_dataset(args.data_dir, "train", size=args.size, ids=[args.image])[0]
    if sample.label is None:
        print("该图没有标注，无法计算指标")
        return 1

    folds_path = os.path.join(args.out, "folds.json")
    folds = json.load(open(folds_path, encoding="utf-8")) if os.path.exists(folds_path) else {}
    if not folds:
        print("提示：未找到 results/folds.json，将使用默认超参；指标可能与报告中的 OOF 结果不同")
    fold = (int(sample.idx) - 1) % 5

    def params_for(key: str) -> dict:
        for rec in folds.get(key, []):
            if rec["fold"] == fold:
                return rec["params"]
        return {}

    cache = os.path.join(args.out, "cache")
    results = {}
    print(f"图像 #{sample.idx}（第 {fold} 折）")
    for key in ("C0", "C1", "C2", "C3"):
        out = pipeline.run(sample, key, params_for(key), cache_dir=cache)
        m = metrics.all_metrics(out["binary"], out["score"], sample.label, sample.fov)
        results[key] = out
        print(f"  {key}: Dice={m['dice']:.4f}  Se={m['se']:.4f}  Sp={m['sp']:.4f}  AUC={m['auc']:.4f}"
              f"  ({out['seconds'] * 1000:.0f} ms)")

    enh_cfg = pipeline.CONFIGS["C2"]["enhance"]
    img = enhance.prepare(sample.bgr, sample.fov, enh_cfg)
    sig = vesselness.sigma_range(sample.fov, 1.0, 0.006, 6)
    resp = vesselness.vesselness(img, sample.fov, sig)

    panels = [
        (sample.bgr[:, :, ::-1], "input (BGR->RGB)"),
        (np.clip(img * 255, 0, 255).astype(np.uint8), f"enhanced: G+CLAHE\nsigma={sig[0]}..{sig[-1]}"),
        (np.clip(resp * 255, 0, 255).astype(np.uint8), "Frangi vesselness"),
        (overlay(img, sample.label, None), "ground truth (green)"),
        (overlay(img, None, results["C0"]["binary"]), "C0: gray + global Otsu"),
        (overlay(img, None, results["C1"]["binary"]), "C1: + CLAHE + local adaptive"),
        (overlay(img, None, results["C2"]["binary"]), "C2: + Frangi + hysteresis"),
        (overlay(img, None, results["C3"]["binary"]), "C3: + post-processing"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (im, title) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out_path = os.path.join(args.out, f"demo_{sample.idx}.png")
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"图像已保存：{out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
