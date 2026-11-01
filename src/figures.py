"""出图：代表图定性对比、失效案例分解、灰度直方图。"""

from __future__ import annotations

import json
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import enhance, io_utils, metrics, pipeline  # noqa: E402

plt.rcParams["font.size"] = 9


def gray3(x: np.ndarray) -> np.ndarray:
    u8 = np.clip(x * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)


def overlay(img01: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = gray3(img01)
    base = (base * 0.75).astype(np.uint8)
    base[gt > 0] = (60, 200, 60)
    base[pred > 0] = (255, 60, 60)
    return base


def params_for(folds: dict, key: str, fold: int) -> dict:
    for rec in folds.get(key, []):
        if rec["fold"] == fold:
            return rec["params"]
    return {}


def main(data_dir: str, out_dir: str = "results", size: int = 512, cases=("1", "12", "5")) -> None:
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    folds_path = os.path.join(out_dir, "folds.json")
    folds = json.load(open(folds_path, encoding="utf-8")) if os.path.exists(folds_path) else {}

    samples = {s.idx: s for s in io_utils.load_dataset(data_dir, "train", size=size)}

    rows = []
    for idx in cases:
        s = samples[idx]
        fold = (int(idx) - 1) % 5
        res = {}
        for key in ("C1", "C2", "C3"):
            res[key] = pipeline.run(s, key, params_for(folds, key, fold), cache_dir=os.path.join(out_dir, "cache"))
        img = enhance.prepare(s.bgr, s.fov, pipeline.CONFIGS["C1"]["enhance"])
        px = [s.bgr[:, :, ::-1], gray3(img), overlay(img, s.label, np.zeros_like(s.label))]
        for key in ("C1", "C2", "C3"):
            b = res[key]["binary"]
            px.append(overlay(img, np.zeros_like(s.label), b))
        rows.append((idx, [res["C1"], res["C2"], res["C3"]], px))

    ncol = 6
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.0 * ncol, 3.0 * len(rows)))
    titles = ["Image", "Enhanced (G+CLAHE)", "GT", "C1 adaptive", "C2 Frangi+hyst", "C3 +post"]
    for r, (idx, res, px) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r, c] if len(rows) > 1 else axes[c]
            ax.imshow(px[c])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if c >= 3:
                m = res[c - 3]
                d = metrics.dice(m["binary"], samples[idx].label, samples[idx].fov)
                ax.set_xlabel(f"Dice={d:.3f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"#{idx}", fontsize=10)
    fig.tight_layout()
    p1 = os.path.join(fig_dir, "fig_cases.png")
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    print("saved", p1)

    dice_all = {}
    for idx, s in samples.items():
        fold = (int(idx) - 1) % 5
        out = pipeline.run(s, "C3", params_for(folds, "C3", fold), cache_dir=os.path.join(out_dir, "cache"))
        dice_all[idx] = metrics.dice(out["binary"], s.label, s.fov)
    worst = min(dice_all, key=dice_all.get)
    best = max(dice_all, key=dice_all.get)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, idx in zip(axes[:2], (best, worst)):
        s = samples[idx]
        fold = (int(idx) - 1) % 5
        out = pipeline.run(s, "C3", params_for(folds, "C3", fold), cache_dir=os.path.join(out_dir, "cache"))
        tp = (out["binary"] > 0) & (s.label > 0)
        fp = (out["binary"] > 0) & (s.label == 0)
        fn = (out["binary"] == 0) & (s.label > 0)
        err = np.zeros((*s.label.shape, 3), np.uint8)
        err[tp] = (90, 90, 90)
        err[fp] = (255, 60, 60)
        err[fn] = (60, 120, 255)
        ax.imshow(err)
        ax.set_title(f"#{idx}  Dice={dice_all[idx]:.3f}\nred=FP  blue=FN  grey=TP", fontsize=9)
        ax.axis("off")
    axes[2].hist([dice_all[k] for k in samples], bins=10, color="#4C72B0")
    axes[2].set_title("C3 Dice per image (20 images)", fontsize=10)
    axes[2].set_xlabel("Dice")
    axes[2].set_ylabel("count")
    axes[3].axis("off")
    txt = "\n".join([f"#{k}: {dice_all[k]:.3f}" for k in sorted(dice_all, key=lambda x: int(x))])
    axes[3].text(0, 1, txt, va="top", fontsize=7.5, family="monospace")
    axes[3].set_title("all images", fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(fig_dir, "fig_failure.png")
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    print("saved", p2)

    s = samples[cases[0]]
    gray = s.bgr.astype(np.float32).mean(axis=2)
    inner = cv2.erode(s.fov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).astype(bool)
    band = inner & (gray > 15)
    vals = gray[band]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ax[0].hist(vals, bins=64, color="#4C72B0")
    ax[0].set_title("gray histogram of retinal pixels: one wide mode", fontsize=10)
    ax[0].set_xlabel("gray level")
    ax[0].set_ylabel("count")
    ax[1].imshow(gray, cmap="gray")
    ax[1].set_title("illumination gradient dominates the histogram", fontsize=10)
    ax[1].axis("off")
    fig.tight_layout()
    p3 = os.path.join(fig_dir, "fig_hist.png")
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    print("saved", p3)

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("VESSEL_DATA", "data/segmentation"))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    main(a.data_dir, a.out)
