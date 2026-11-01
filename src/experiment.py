"""实验主程序：5 折交叉验证 + 每折独立调参 + 指标汇总。

用法：python -m src.experiment --data-dir <数据目录> --out results
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np

from . import io_utils, metrics, pipeline

N_FOLDS = 5
GRID = {
    "C0": [{}],
    "C1": [{"block": int(b), "c": float(c)}
           for b, c in itertools.product((15, 21, 31), (8, 10, 12, 14, 16))],
    "C2": [{"q_low": float(ql), "q_high": float(qh)}
           for ql, qh in itertools.product((0.80, 0.85, 0.90, 0.93), (0.97, 0.98, 0.99, 0.995))],
    "C3": [{"q_low": float(ql), "q_high": float(qh), "min_area": ma, "dilate": dl}
           for (ql, qh), ma, dl in itertools.product(((0.85, 0.97), (0.90, 0.99)), (20, 40), (0, 1))],
}


def fold_of(idx: str) -> int:
    return (int(idx) - 1) % N_FOLDS


def evaluate(samples, key: str, params: dict, cache_dir: str) -> list[dict]:
    rows = []
    for s in samples:
        out = pipeline.run(s, key, params, cache_dir)
        m = metrics.all_metrics(out["binary"], out["score"], s.label, s.fov)
        m.update({"image": s.idx, "method": key, "seconds": out["seconds"]})
        rows.append(m)
    return rows


def mean_dice(rows: list[dict]) -> float:
    return float(np.mean([r["dice"] for r in rows]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("VESSEL_DATA", "data/segmentation"),
                    help="数据目录（默认读环境变量 VESSEL_DATA）")
    ap.add_argument("--out", default="results")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    cache_dir = os.path.join(args.out, "cache")

    samples = io_utils.load_dataset(args.data_dir, "train", size=args.size)
    print(f"载入 {len(samples)} 张带标注图像，分辨率 {args.size}")

    all_rows: list[dict] = []
    chosen: dict[str, list[dict]] = {}

    for key in ("C0", "C1", "C2", "C3"):
        print(f"\n=== {pipeline.CONFIGS[key]['label']}")
        chosen[key] = []
        for f in range(args.folds):
            test = [s for s in samples if fold_of(s.idx) == f]
            train = [s for s in samples if fold_of(s.idx) != f]
            t0 = time.perf_counter()
            best, best_params = -1.0, None
            for params in GRID[key]:
                rows = evaluate(train, key, params, cache_dir)
                d = mean_dice(rows)
                if d > best:
                    best, best_params = d, params
            rows = evaluate(test, key, best_params, cache_dir)
            for r in rows:
                r["fold"] = f
                r["params"] = json.dumps(best_params, ensure_ascii=False)
            all_rows.extend(rows)
            chosen[key].append({"fold": f, "params": best_params, "train_dice": round(best, 4),
                                "test_dice": round(mean_dice(rows), 4),
                                "seconds": round(time.perf_counter() - t0, 1)})
            print(f"  fold {f}: 选中 {best_params}  训练均值 {best:.4f} → 本折 {mean_dice(rows):.4f}"
                  f"  ({time.perf_counter()-t0:.0f}s)")

    per_image = os.path.join(args.out, "metrics_per_image.csv")
    fields = ["method", "fold", "image", "dice", "se", "sp", "acc", "auc", "pred_ratio", "gt_ratio",
              "seconds", "params"]
    with open(per_image, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (round(r[k], 6) if isinstance(r.get(k), float) else r.get(k)) for k in fields})

    summary = []
    for key in ("C0", "C1", "C2", "C3"):
        rs = [r for r in all_rows if r["method"] == key]
        row = {"method": key, "label": pipeline.CONFIGS[key]["label"], "n": len(rs)}
        for m in ("dice", "se", "sp", "acc", "auc"):
            vals = np.array([r[m] for r in rs], dtype=float)
            row[m] = round(float(vals.mean()), 4)
            row[m + "_std"] = round(float(vals.std(ddof=1)), 4)
        row["seconds"] = round(float(np.mean([r["seconds"] for r in rs])), 3)
        summary.append(row)

    with open(os.path.join(args.out, "summary.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    with open(os.path.join(args.out, "folds.json"), "w", encoding="utf-8") as fh:
        json.dump(chosen, fh, ensure_ascii=False, indent=2)

    md = ["# 实验结果（5 折交叉验证，每折独立调参）", "",
          "| 方案 | Dice | Se | Sp | Acc | AUC | 单图耗时(s) |", "|---|---|---|---|---|---|---|"]
    for r in summary:
        md.append(f"| {r['method']} | {r['dice']:.4f} ± {r['dice_std']:.4f} | {r['se']:.4f} "
                  f"| {r['sp']:.4f} | {r['acc']:.4f} | {r['auc']:.4f} | {r['seconds']:.3f} |")
    md.append("")
    for r in summary:
        md.append(f"- {r['method']}：{r['label']}")
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    print("\n=== 汇总 ===")
    for r in summary:
        print(f"{r['method']}: Dice={r['dice']:.4f}±{r['dice_std']:.4f}  Se={r['se']:.4f}  "
              f"Sp={r['sp']:.4f}  Acc={r['acc']:.4f}  AUC={r['auc']:.4f}  {r['seconds']:.3f}s")
    print(f"\n结果已写入 {os.path.abspath(args.out)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
