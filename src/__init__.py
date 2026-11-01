"""视网膜血管分割 —— 经典图像处理流水线（课程大作业）。

模块划分与课件对应关系：
    enhance.py      第 3 章 图像增强（灰度映射、直方图修正、形态学背景校正）
    vesselness.py   3.6 图像锐化 / 4.3 边缘检测（Hessian 二阶微分，多尺度 FRANGI）
    segment.py      4.2 区域分割（Otsu 全局阈值、局部自适应阈值、滞后阈值）
    postprocess.py  3.5 形态学 / 4.3 轮廓与连通域处理
    metrics.py      Dice / Se / Sp / Acc / AUC
    pipeline.py     四个消融配置 C0→C3
"""

__version__ = "1.0.0"
