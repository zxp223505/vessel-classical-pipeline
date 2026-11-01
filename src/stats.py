"""统计检验：配对比较、置信区间、可分辨效应量。

输出 results/stats.csv 与 stats.md；用法：python -m src.stats --results results
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

METHODS = ("C0", "C1", "C2", "C3")
PAIRS = (("C0", "C1"), ("C1", "C2"), ("C1", "C3"), ("C2", "C3"))


def _load(results: str) -> dict[str, dict[str, float]]:
    path = os.path.join(results, "metrics_per_image.csv")
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        out.setdefault(r["method"], {})[r["image"]] = float(r["dice"])
    return out


def _stats(a: np.ndarray, b: np.ndarray) -> dict:
    """计算配对差值、置信区间、t 检验与 Wilcoxon 检验。"""
    d = b - a
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    dz = mean / sd if sd > 1e-12 else float("nan")
    try:
        from scipy import stats as st
        t = st.ttest_rel(a, b)
        p_t = float(t.pvalue)
        try:
            p_w = float(st.wilcoxon(a, b).pvalue)
        except ValueError:
            p_w = float("nan")
        tcrit = float(st.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    except Exception:  # 极端情况下没有 scipy
        p_t, p_w = float("nan"), float("nan")
        tcrit = 1.96
    ci_lo, ci_hi = mean - tcrit * se, mean + tcrit * se
    n_need = int(np.ceil((2.802 * sd / 0.01) ** 2)) if sd > 1e-12 else 0
    return {"mean_diff": mean, "sd_diff": sd, "se": se, "dz": dz,
            "p_t": p_t, "p_wilcoxon": p_w, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_for_0.01": n_need, "n": n}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--delta", type=float, default=0.01, help="要分辨的效应量（Dice）")
    args = ap.parse_args(argv)

    data = _load(args.results)
    imgs = sorted(data["C3"], key=lambda s: int(s))

    summary = []
    for m in METHODS:
        x = np.array([data[m][i] for i in imgs])
        sd = float(x.std(ddof=1))
        se = sd / np.sqrt(x.size)
        summary.append({"method": m, "n": int(x.size), "mean": float(x.mean()),
                        "std": sd, "se": se, "ci_lo": float(x.mean() - 1.96 * se),
                        "ci_hi": float(x.mean() + 1.96 * se)})

    with open(os.path.join(args.results, "stats.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "n", "mean", "std", "se", "ci_lo", "ci_hi"])
        w.writeheader()
        for r in summary:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    pairs = []
    for a, b in PAIRS:
        x = np.array([data[a][i] for i in imgs])
        y = np.array([data[b][i] for i in imgs])
        s = _stats(x, y)
        s.update({"a": a, "b": b})
        pairs.append(s)

    with open(os.path.join(args.results, "stats_pairs.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        fields = ["a", "b", "mean_diff", "sd_diff", "se", "dz", "ci_lo", "ci_hi", "p_t", "p_wilcoxon", "n"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in pairs:
            w.writerow({k: (round(r[k], 6) if isinstance(r[k], float) else r[k]) for k in fields})

    md = ["# 统计检验（配对设计，n = %d 张图）" % len(imgs), "",
          "| 方案 | Dice 均值 | 标准差 | 标准误 | 95% 置信区间 |", "|---|---|---|---|---|"]
    for r in summary:
        md.append(f"| {r['method']} | {r['mean']:.4f} | {r['std']:.4f} | {r['se']:.4f} "
                  f"| [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] |")
    md += ["", "## 配对比较（同一批图、同一折，仅换方法）", "",
           "| 比较 | 平均差 | 标准误 | Cohen's dz | 95% CI | 配对 t 的 p | Wilcoxon 的 p |",
           "|---|---|---|---|---|---|---|"]
    for r in pairs:
        md.append(f"| {r['a']} → {r['b']} | {r['mean_diff']:+.4f} | {r['se']:.4f} | {r['dz']:.2f} "
                  f"| [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_t']:.2e} | {r['p_wilcoxon']:.2e} |")
    sd_pair = min(r["sd_diff"] for r in pairs if r["sd_diff"] > 0)
    md += ["", "## 可分辨的效应量", "",
           f"- 配对差值标准差约 {sd_pair:.4f}，标准误约 {sd_pair / np.sqrt(len(imgs)):.4f}；",
           f"- 以 80% 功效、双侧 α=0.05 计，分辨 {args.delta:.3f} 的 Dice 差异约需 "
           f"{int(np.ceil((2.802 * sd_pair / args.delta) ** 2))} 张图；",
           "- 因此本报告只对 ≥0.02 的差异下结论，更小的差异一律表述为“分辨不出”。"]
    with open(os.path.join(args.results, "stats.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    print("方案均值与 95% CI：")
    for r in summary:
        print(f"  {r['method']}: {r['mean']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
    print("配对比较：")
    for r in pairs:
        print(f"  {r['a']}→{r['b']}: {r['mean_diff']:+.4f} (p_t={r['p_t']:.2e})")
    print(f"结果写入 {os.path.join(os.path.abspath(args.results), 'stats.md')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
