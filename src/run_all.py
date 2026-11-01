"""一键复现：交叉验证实验 → 出图 → 统计检验 → 深度学习对照。"""

from __future__ import annotations

import argparse
import os
import sys

from . import dl_reference, experiment, figures, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("VESSEL_DATA", "data/segmentation"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--retina-runs", default=r"..\retina\runs", help="深度学习对照的 OOF 目录，可留空")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    print("[1/4] 交叉验证实验 ...")
    experiment.main(["--data-dir", args.data_dir, "--out", args.out, "--size", str(args.size)])

    print("\n[2/4] 生成对比图 ...")
    figures.main(args.data_dir, args.out, size=args.size)

    print("\n[3/4] 统计检验 ...")
    stats.main(["--results", args.out])

    print("\n[4/4] 深度学习对照 ...")
    if args.retina_runs and os.path.isdir(args.retina_runs):
        dl_reference.main(args.retina_runs, args.out, data_dir=args.data_dir)
    else:
        print("未找到 OOF 目录，跳过")

    print(f"\n完成，结果在 {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
