import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
os.makedirs("results/figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(11, 3.0))
ax.set_xlim(0, 110); ax.set_ylim(0, 32); ax.axis("off")
boxes = [
    (2,  "Input\ncolour fundus\n584x565", "#DCE6F1"),
    (24, "Preprocess\nG channel -> border fill\n-> normalise -> CLAHE", "#E2EFDA"),
    (48, "Vessel enhancement\nmulti-scale Frangi\nadaptive sigma / per-scale norm", "#FCE4D6"),
    (72, "Segmentation\nlocal adaptive threshold\nor hysteresis", "#FFF2CC"),
    (94, "Post-process\narea filter -> fill holes\n-> dilate -> FOV clip", "#EDE7F6"),
]
for x, text, color in boxes:
    ax.add_patch(FancyBboxPatch((x, 8), 14, 16, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=color, ec="#5A5A5A", lw=1.2))
    ax.text(x + 7, 16, text, ha="center", va="center", fontsize=8)
for x, *_ in boxes[:-1]:
    ax.add_patch(FancyArrowPatch((x + 15.2, 16), (x + 21.6, 16), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.3, color="#404040"))
ax.text(55, 3, "Output: binary vessel map 512x512    Metric: Dice / Se / Sp / AUC inside FOV",
        ha="center", fontsize=8.5, color="#333333")
fig.tight_layout()
fig.savefig("results/figures/fig_pipeline.png", dpi=170)
plt.close(fig)
print("saved")
