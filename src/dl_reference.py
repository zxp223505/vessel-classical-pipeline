"""深度学习对照基线：读取 retina/ 工程的 OOF 预测，按同一口径算指标。

说明（报告第 5 节会引用）：
    retina/ 是同一数据的 ResNet34+U-Net 方案，训练时做了 5 折交叉验证，
    每个 checkpoint 目录的 oof/ 里保存的是"该折验证图"的概率图（512×512 float16），
    因此这些预测对每张图而言都来自"没见过它"的模型，可以直接与本项目的 OOF 指标比较。

    同一张图可能被多个 run 覆盖（例如 f0 与 f0_thick 都验证了同一折），
    这里按图号去重、对概率图取平均（等价于多模型集成），从而不重复计权。
    这是"集成上界"，与本项目单模型的对比要在报告里说明清楚。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

from . import io_utils, metrics


def collect(oof_root: str, margin: int = 8, data_dir: str | None = None) -> tuple[list[dict], dict]:
    """data_dir 给定时，用本项目同一套 fov_mask 重算评价区域（保证与 C0–C3 完全同口径）。"""
    """返回 (逐图指标, 汇总)。按图号去重，多 run 的概率图取平均。"""
    buckets: dict[str, dict] = {}
    if not os.path.isdir(oof_root):
        return [], {}
    for run in sorted(os.listdir(oof_root)):
        d = os.path.join(oof_root, run, "oof")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npy") or "_" in f:
                continue
            idx = f[:-4]
            gt_p = os.path.join(d, f"{idx}_gt.npy")
            fov_p = os.path.join(d, f"{idx}_fov.npy")
            if not (os.path.exists(gt_p) and os.path.exists(fov_p)):
                continue
            prob = np.load(os.path.join(d, f)).astype(np.float32)
            b = buckets.setdefault(idx, {"probs": [], "gt": np.load(gt_p), "fov": np.load(fov_p), "runs": []})
            b["probs"].append(prob)
            b["runs"].append(run)

    rows = []
    for idx in sorted(buckets, key=lambda s: int(s) if s.isdigit() else s):
        b = buckets[idx]
        prob = np.mean(np.stack(b["probs"], axis=0), axis=0)
        pred = (prob >= 0.5).astype(np.uint8)
        fov = b["fov"]
        if data_dir:  # 用同一套视场掩膜重算，保证与 C0–C3 完全同口径
            img_path = os.path.join(data_dir, "train", "image", f"{idx}.jpg")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (prob.shape[1], prob.shape[0]), interpolation=cv2.INTER_AREA)
                fov = io_utils.fov_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), erode=margin)
        elif margin > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            fov = cv2.erode(fov.astype(np.uint8), k)
        m = metrics.all_metrics(pred, prob, b["gt"], fov)
        m.update({"image": idx, "n_runs": len(b["runs"]), "runs": "|".join(b["runs"])})
        rows.append(m)

    summary = {"n": len(rows)}
    for k in ("dice", "se", "sp", "acc", "auc"):
        summary[k] = float(np.mean([r[k] for r in rows]))
    summary["runs_total"] = sum(r["n_runs"] for r in rows)
    return rows, summary


def main(oof_root: str, out: str, margin: int = 8, data_dir: str | None = None) -> None:
    rows, summary = collect(oof_root, margin=margin, data_dir=data_dir)
    if not rows:
        print("未找到 OOF 预测，跳过深度学习对照")
        return
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "dl_reference.csv")
    fields = ["image", "n_runs", "runs", "dice", "se", "sp", "acc", "auc", "pred_ratio", "gt_ratio"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 6) if isinstance(r[k], float) else r[k]) for k in fields})
    print(f"{summary['n']} 张图（{summary['runs_total']} 条 OOF 输出，同图取平均）：", end=" ")
    for m in ("dice", "se", "sp", "acc", "auc"):
        print(f"{m}={summary[m]:.4f}", end="  ")
    print(f"\n写入 {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-root", default="..\\retina\\runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--margin", type=int, default=8, help="视场过渡带内缩像素")
    ap.add_argument("--data-dir", default=os.environ.get("VESSEL_DATA", "data/segmentation"),
                    help="给定时用同一套掩膜重算评价区域")
    a = ap.parse_args()
    sys.exit(main(a.oof_root, a.out, a.margin, a.data_dir))
