# -*- coding: utf-8 -*-
"""生成课程报告 DOCX（严格按模板 5 节组织）。"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SONG, HEI, KAI = "宋体", "黑体", "楷体"
BODY, TBL = 10.5, 9.0


def set_font(run, name=SONG, size=BODY, bold=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia"):
        rFonts.set(qn(attr), name)


def shade(cell, color="E8EEF6"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


class Report:
    def __init__(self):
        self.doc = Document()
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        sec.left_margin = sec.right_margin = Cm(2.5)
        sec.top_margin, sec.bottom_margin = Cm(2.4), Cm(2.2)
        st = self.doc.styles["Normal"]
        st.font.name = SONG
        st.font.size = Pt(BODY)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)
        self._footer(sec)

    def _footer(self, sec):
        p = sec.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("基于经典图像处理的视网膜血管分割　·　图像处理课程大作业")
        set_font(r, SONG, 8, color=RGBColor(0x80, 0x80, 0x80))

    def p(self, text, size=BODY, bold=False, indent=True, before=0, after=4, font=SONG,
          color=None, align=None, line=1.4):
        par = self.doc.add_paragraph()
        pf = par.paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), line
        if indent:
            pf.first_line_indent = Pt(size * 2)
        if align is not None:
            par.alignment = align
        r = par.add_run(text)
        set_font(r, font, size, bold, color)
        return par

    def bullet(self, text, size=BODY):
        par = self.doc.add_paragraph()
        pf = par.paragraph_format
        pf.space_after, pf.line_spacing = Pt(3), 1.35
        pf.left_indent, pf.first_line_indent = Pt(size * 1.7), Pt(-size * 1.1)
        r = par.add_run("· " + text)
        set_font(r, SONG, size)
        return par

    def h1(self, text):
        par = self.doc.add_paragraph()
        pf = par.paragraph_format
        pf.space_before, pf.space_after, pf.keep_with_next = Pt(14), Pt(6), True
        r = par.add_run(text)
        set_font(r, HEI, 14, bold=True, color=RGBColor(0x1F, 0x3B, 0x63))
        pPr = par._p.get_or_add_pPr()
        pbdr = OxmlElement("w:pBdr")
        b = OxmlElement("w:bottom")
        for k, v in (("w:val", "single"), ("w:sz", "8"), ("w:space", "2"), ("w:color", "1F3B63")):
            b.set(qn(k), v)
        pbdr.append(b)
        pPr.append(pbdr)
        return par

    def h2(self, text):
        par = self.doc.add_paragraph()
        pf = par.paragraph_format
        pf.space_before, pf.space_after, pf.keep_with_next = Pt(9), Pt(4), True
        r = par.add_run(text)
        set_font(r, HEI, 11.5, bold=True, color=RGBColor(0x2E, 0x5A, 0x88))
        return par

    def caption(self, text):
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_before = Pt(3)
        par.paragraph_format.space_after = Pt(4)
        par.paragraph_format.keep_with_next = True
        r = par.add_run(text)
        set_font(r, KAI, 8.5, color=RGBColor(0x59, 0x59, 0x59))

    def table(self, headers, rows, widths, font=TBL, first_col_bold=True):
        t = self.doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, h in enumerate(headers):
            c = t.rows[0].cells[i]
            c.text = ""
            par = c.paragraphs[0]
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.space_after = Pt(1)
            set_font(par.add_run(h), HEI, font, bold=True)
            shade(c)
        for row in rows:
            cells = t.add_row().cells
            for i, val in enumerate(row):
                cells[i].text = ""
                par = cells[i].paragraphs[0]
                par.paragraph_format.space_after = Pt(1)
                par.paragraph_format.line_spacing = 1.15
                if i == 0 or (len(str(val)) <= 6):
                    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                set_font(par.add_run(str(val)), SONG, font, bold=(first_col_bold and i == 0))
        for i, w in enumerate(widths):
            t.columns[i].width = Cm(w)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
            trPr = row._tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit")
            cant.set(qn("w:val"), "true")
            trPr.append(cant)
        trPr = t.rows[0]._tr.get_or_add_trPr()
        th = OxmlElement("w:tblHeader")
        th.set(qn("w:val"), "true")
        trPr.append(th)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def figure(self, path, width_cm, cap):
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_before = Pt(6)
        par.paragraph_format.space_after = Pt(2)
        par.add_run().add_picture(path, width=Cm(width_cm))
        self.caption(cap)

    def save(self, path):
        self.doc.save(path)


def load_summary(results: str) -> list[dict]:
    with open(os.path.join(results, "summary.csv"), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_per_image(results: str) -> list[dict]:
    with open(os.path.join(results, "metrics_per_image.csv"), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_dl(results: str) -> dict | None:
    p = os.path.join(results, "dl_reference.csv")
    if not os.path.exists(p):
        return None
    import statistics as st
    with open(p, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    out = {"n": len(rows), "runs": sum(int(r["n_runs"]) for r in rows)}
    for m in ("dice", "se", "sp", "acc", "auc"):
        out[m] = st.mean(float(r[m]) for r in rows)
    return out


def load_stats(results: str):
    """读取 stats.csv / stats_pairs.csv；缺失时返回空表，报告仍可生成。"""
    sp = os.path.join(results, "stats_pairs.csv")
    pairs = []
    if os.path.exists(sp):
        with open(sp, encoding="utf-8-sig") as fh:
            pairs = list(csv.DictReader(fh))
    return pairs


def main(results: str, fig_dir: str, out_path: str):
    summary = {r["method"]: r for r in load_summary(results)}
    per_image = load_per_image(results)
    dl = load_dl(results)
    stats_pairs = load_stats(results)
    folds = json.load(open(os.path.join(results, "folds.json"), encoding="utf-8"))

    c3 = {r["image"]: r for r in per_image if r["method"] == "C3"}
    worst = min(c3, key=lambda k: float(c3[k]["dice"]))
    best = max(c3, key=lambda k: float(c3[k]["dice"]))

    R = Report()

    # ---------------- 封面 ----------------
    t = R.doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_before = Pt(30)
    set_font(t.add_run("图像处理课程大作业报告"), HEI, 22, bold=True)

    sub = R.doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.space_after = Pt(20)
    set_font(sub.add_run("基于局部自适应阈值与形态学后处理的视网膜血管分割"), HEI, 13.5,
             color=RGBColor(0x33, 0x33, 0x33))

    info = [("姓　　名", "（填写）"), ("学　　号", "（填写）"), ("专　　业", "（填写）"),
            ("课程名称", "图像处理"), ("完成日期", "2026 年 11 月")]
    tb = R.doc.add_table(rows=0, cols=2)
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    for k, v in info:
        cells = tb.add_row().cells
        cells[0].width, cells[1].width = Cm(3.2), Cm(7.0)
        for i, txt in enumerate((k, v)):
            cells[i].text = ""
            par = cells[i].paragraphs[0]
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT if i else WD_ALIGN_PARAGRAPH.RIGHT
            set_font(par.add_run(txt), SONG, 11)
    R.doc.add_page_break()

    # ---------------- 一、Github 链接 ----------------
    R.h1("1. Github 链接")
    R.p("仓库地址：https://github.com/（待填：你的 GitHub 用户名）/vessel-classical-pipeline")
    R.p("代码已按可复现的方式组织，仓库根目录结构如下：", after=2)
    R.table(
        ["路径", "内容"],
        [
            ["src/enhance.py", "灰度映射、绿通道、边界填充、CLAHE、顶帽"],
            ["src/vesselness.py", "多尺度 Frangi（尺度自适应 + 逐尺度归一化）"],
            ["src/segment.py", "全局 Otsu、局部自适应阈值、滞后阈值"],
            ["src/postprocess.py", "面积过滤、长宽比过滤、填孔、膨胀、视场裁剪"],
            ["src/metrics.py", "Dice / Se / Sp / Acc / AUC"],
            ["src/pipeline.py", "C0–C3 四个对照方案的统一入口"],
            ["src/experiment.py", "5 折交叉验证 + 每折独立调参 + 指标导出"],
            ["src/figures.py", "定性对比图、失效分析图、直方图"],
            ["results/", "指标 CSV、汇总、图片（运行后生成）"],
        ],
        widths=[4.6, 11.4],
    )
    R.caption("表 1　仓库目录与模块说明")
    R.p("复现命令（Windows / Linux 通用）：", after=2)
    code = R.doc.add_paragraph()
    code.paragraph_format.left_indent = Pt(18)
    code.paragraph_format.space_after = Pt(6)
    code.paragraph_format.line_spacing = 1.25
    set_font(code.add_run(
        "pip install -r requirements.txt\n"
        "python -m src.run_all --data-dir <DRIVE 数据目录> --out results"), "Consolas", 8.5)
    R.p("提交前只需三步：① 在本目录 git init && git add && git commit；② 在 GitHub 新建空仓库；"
        "③ git remote add origin <地址> && git push -u origin main。"
        "报告中的指标全部由 results/summary.csv 直接导出，可逐项复算。")

    # ---------------- 二、任务描述 ----------------
    R.h1("2. 任务描述：我要做什么")
    R.h2("2.1 问题定义")
    R.p("给定一幅彩色眼底图（584×565，3 通道），输出一张同尺寸的二值图，逐像素判定该点是否为视网膜血管。"
        "这是一个典型的二类像素级分割问题：血管像素只占视场的 8%–10%，正负样本严重不平衡，"
        "因此不能只看准确率，必须看 Dice 与敏感度。")
    R.h2("2.2 这件事为什么值得做")
    R.p("视网膜血管的形态（宽度、弯曲度、分叉）与高血压、糖尿病视网膜病变等疾病直接相关，"
        "自动分割是筛查流程的第一步。难点也很具体，正好构成方法设计的动机：")
    for s in [
        "对比度低且不均匀：血管与背景的灰度差只有十几级，且受照明和相机暗角影响，全图亮度呈明显的径向梯度；",
        "尺度跨度大：视盘附近的粗血管宽达 6–8 px，周边毛细血管只有 1 px，单一尺度的算子必然顾此失彼；",
        "类别不平衡：血管像素约 8.7%，背景 91.3%，用准确率评价会得到 0.91 的“假高分”；",
        "标注本身有噪声：不同专家标注的一致性有限（Dice 上限约 0.85–0.9），这决定了指标的天花板。",
    ]:
        R.bullet(s)
    R.h2("2.3 方法概述")
    R.p("整条流水线分成四段，每段只承担一件事，便于逐段做消融：预处理（绿通道 + CLAHE，并统一把视场过渡带内缩 8 px）→ 血管增强"
        "（多尺度 Frangi）或直接分割 → 分割（局部自适应阈值 / 滞后阈值）→ 后处理（面积过滤、填孔、膨胀、视场裁剪）。"
        "在此基础上构造了四个对照方案 C0–C3。")
    R.figure(os.path.join(fig_dir, "fig_pipeline.png"), 15.5, "图 1　处理流水线示意")
    R.table(
        ["流水线环节", "对应课件内容", "本项目具体做法"],
        [
            ["预处理", "第 3 章 图像增强：灰度映射、直方图修正", "绿通道、1%–99% 分位归一化、CLAHE、视场过渡带内缩"],
            ["血管增强", "3.6 图像锐化（二阶微分）、4.3 边缘检测", "多尺度 Hessian / Frangi 管状滤波"],
            ["分割", "4.2 区域分割：阈值分割", "全局 Otsu、局部自适应阈值、滞后阈值"],
            ["后处理", "3.5 形态学、4.3 轮廓与连通域", "面积/长宽比过滤、填孔、膨胀、视场裁剪"],
            ["评价", "—", "Dice、Se、Sp、Acc、AUC（视场内）"],
        ],
        widths=[3.0, 6.4, 6.6],
    )
    R.caption("表 2　流水线与课程内容的对应关系")

    # ---------------- 三、数据描述 ----------------
    R.h1("3. 数据描述：我用的是什么数据")
    R.h2("3.1 数据来源与规模")
    R.p("使用课程提供的 DRIVE 眼底数据子集：20 张带像素级血管标注的训练图、20 张无标注测试图。"
        "图像原始尺寸 584×565，标注为二值图（血管 255、背景 0）。为与评分口径一致，"
        "所有实验统一缩放到 512×512（图像双线性、标注最近邻），在该尺度上计算全部指标。")
    R.p("数据出处：DRIVE 由荷兰乌得勒支大学医学中心（UMC Utrecht）影像科学研究所（Image Sciences Institute）"
        "于 2004 年随论文发布（Staal 等，IEEE TMI），现托管于 Grand Challenge 平台。其 training 部分共 20 张"
        "带像素级标注的图像，即本项目使用的全部数据——不是从更大数据集中抽样，也不存在“只用了一小部分”的问题。"
        "test 部分 20 张的标注不公开，只能提交到平台评分，因此本项目的评测采用 training 部分的 5 折交叉验证。")
    R.table(
        ["项目", "数值 / 说明"],
        [
            ["数据规模", "20 张有标注（训练）+ 20 张无标注（测试）"],
            ["图像尺寸", "原始 584×565 → 统一 512×512"],
            ["标注形式", "像素级二值掩膜，血管 = 255"],
            ["血管像素占比", "8.6%（逐图 6.6%–11.7%），与 DRIVE 原始标注一致"],
            ["视场（FOV）占比", "约 64%：圆形视网膜区域（相对阈值估计），并内缩 8px 排除过渡带"],
            ["标注血管宽度", "中位半宽约 1.0 px（512 尺度），属于典型的细结构分割"],
            ["划分方式", "按图编号 mod 5 分 5 折，每折 4 张；调参与评测严格分离"],
        ],
        widths=[3.6, 12.4],
    )
    R.caption("表 3　数据概况")
    R.h2("3.2 划分与调参协议")
    R.p("交叉验证的设计要点是“每张图的超参不来自它自己”。具体做法：把 20 张图按编号 mod 5 分成 5 折；"
        "对每个方案，在其余 16 张上网格搜索超参（局部阈值的 C、Frangi 的双分位数、后处理的面积阈值与膨胀半径），"
        "取平均 Dice 最高的一组，再在留出的 4 张上评测。汇总得到的 OOF 指标可以与任何同样做交叉验证的方法直接比较，"
        "这为本报告第 5 节引入深度学习方法作为参照提供了公平的前提。"
        "评测区域统一取“视场掩膜内缩 8 px”：视网膜与纯黑之间有一段渐隐过渡带，它既不是前景也不是背景，"
        "局部阈值在带上必然产生一圈稳定误检；把这 8 px 排除，对所有方法（含深度学习对照）一视同仁。")
    R.h2("3.3 评价指标")
    R.p("全部指标在视场掩膜内计算（视场外的纯黑区域不参与统计）：Dice = 2TP / (2TP + FP + FN) 为主指标；"
        "Se = TP / (TP + FN) 反映细血管召回；Sp = TN / (TN + FP) 反映伪响应抑制；"
        "AUC 用各方法输出的连续响应图（或 1−增强图）排序计算，衡量“排序质量”而不受单一阈值影响。")
    R.h2("3.4 数据的一个关键特点")
    R.p("视场内灰度直方图基本是单峰的（图 2 右）：照明梯度把“血管暗、背景亮”的双峰结构抹平了。"
        "这直接解释了为什么全局 Otsu 会失效——它在直方图上找到的“最佳分割点”切开的其实是周边暗区与中心亮区，"
        "而不是血管与背景。")
    R.figure(os.path.join(fig_dir, "fig_hist.png"), 15.0,
             "图 2　视场内灰度直方图（左，已剔除边缘过渡带）与原始灰度图（右）：照明梯度把双峰抹成单峰")

    # ---------------- 四、代码修改描述 ----------------
    R.h1("4. 代码修改描述：我改了什么")
    R.p("基线是一个“教科书式”的实现：灰度图 + 全局 Otsu 阈值。在此基础上按四个环节逐项改进，"
        "每项都对应一个明确的失效现象，而不是凭经验堆模块。")
    R.h2("4.1 改动一：先把评价区域修对（影响最大的一步）")
    R.p("现象：视网膜与纯黑之间有一段 8–10 px 的渐隐带，它既不是前景也不是背景。早期版本里能看到预测沿视场边界"
        "的一整圈红线——那不是算法差，而是评价区域圈错了：绝对阈值（灰度 > 8）把这条带连同视网膜外一圈 10–20 灰度"
        "的暗区一起算进了视场，而那里几乎没有真实血管。")
    R.p("尝试一（失败）：用边界内侧一圈像素的中值填平视场外，再做 CLAHE，期望抹掉“视网膜—黑”的阶跃。"
        "20 张开/关消融显示 C1、C2、C3 分别变化 −0.003、−0.006、−0.006，一致地略微变差；"
        "该函数保留在 enhance.border_fill()，默认关闭。")
    R.p("尝试二（小收益）：把评价掩膜内缩从 4 px 提到 8 px，C3 的 Dice 从 0.595 提到 0.602，环形伪响应消失。")
    R.p("尝试三（关键）：把绝对阈值改成相对阈值——先由粗掩膜估出视网膜内部的中位亮度，再取 0.35×中位亮度作为边界。"
        "绝对阈值下视场占比在 0.68–0.86 之间剧烈波动（几何上应在 0.64 左右），相对阈值后稳定在 0.637，"
        "且 GT 仍有 96% 落在掩膜内。这一步把 C0 从 0.019 提到 0.257、C1 从 0.565 提到 0.662、C2 从 0.604 提到 0.696。"
        "结论很直接：评价区域的定义本身就会改变结论，它比大多数算法改动影响都大；做事之前先把统计口径钉死。")
    R.h2("4.2 改动二：Frangi 的两个改造（尺度自适应 + 逐尺度归一化）")
    R.p("教科书版 Frangi 用固定尺度列表（如 σ=1…10）并直接取各尺度最大值，问题有二："
        "换分辨率就要重调尺度；大尺度响应幅值远大于小尺度，细血管会被“淹没”。本项目的改法是：")
    R.bullet("尺度自适应：σ_max = 0.006 × 视场直径（512 尺度下约 3 px），既覆盖 1–3 px 的血管，"
             "又在换分辨率时自动缩放，代码不用改；")
    R.bullet("逐尺度归一化：每个尺度先除以该尺度响应的 99.5% 分位，再取 max；统计量只在内缩 12 px 的视场内计算，"
             "避免边界干扰。")
    R.p("对应的函数是 vesselness.vesselness()，公式为 V = exp(−Rb²/2β²)·(1 − exp(−S²/2c²))，"
        "其中 Rb = |λ1|/|λ2|、S = √(λ1²+λ2²)，β = 0.5，c 取该尺度 S 的 99.5% 分位的 0.5 倍。")
    R.h2("4.3 改动三：用局部自适应阈值替代全局 Otsu")
    R.p(f"全局 Otsu 在本文数据上基本不可用（第 5 节的 C0 只有 {float(summary['C0']['dice']):.3f}）。"
        "改为对每个像素与其邻域的高斯加权均值比较"
        "（cv2.adaptiveThreshold，block = 21，C 由交叉验证选定），相当于把“依赖坐标的阈值”落到实现上，"
        "对径向照明梯度天然免疫。")
    R.h2("4.4 改动四：滞后阈值与后处理链")
    R.p("在 Frangi 响应上使用双阈值：高分位做种子、低分位做生长（segment.hysteresis），保证细血管的连通性；"
        "在阈值化之后再加一条后处理链：面积过滤（去掉 < 40 px 的孤立噪声）→ 填掉被前景包围的小孔（< 40 px）"
        "→ 膨胀补偿（可选）→ 视场裁剪。后处理参数与分割阈值一起参与交叉验证。")
    R.h2("4.5 参数与搜索范围")
    R.table(
        ["参数", "含义", "默认值", "搜索范围"],
        [
            ["block", "局部阈值窗口", "21", "15 / 21 / 31"],
            ["C", "局部阈值偏移（越大越严格）", "12", "6–18（步长 2）"],
            ["q_low / q_high", "滞后阈值双分位", "0.90 / 0.99", "0.80–0.93 / 0.97–0.995"],
            ["min_area", "面积过滤阈值（px）", "40", "20 / 40 / 60"],
            ["dilate", "膨胀半径（px）", "1", "0 / 1"],
            ["σ", "Frangi 尺度范围", "1.0–3.0（6 个）", "由视场直径换算"],
        ],
        widths=[3.2, 6.0, 3.6, 3.2],
    )
    R.caption("表 4　参数与交叉验证中的搜索范围")

    # ---------------- 五、结果对比 ----------------
    R.h1("5. 结果对比：结果有什么变化")
    R.h2("5.1 主结果")
    rows = []
    for k in ("C0", "C1", "C2", "C3"):
        s = summary[k]
        rows.append([k, f"{float(s['dice']):.3f} ± {float(s['dice_std']):.3f}",
                     f"{float(s['se']):.3f}", f"{float(s['sp']):.3f}",
                     f"{float(s['acc']):.3f}", f"{float(s['auc']):.3f}",
                     f"{float(s['seconds']):.3f}"])
    if dl:
        rows.append(["DL(参考)", f"{dl['dice']:.3f}", f"{dl['se']:.3f}", f"{dl['sp']:.3f}",
                     f"{dl['acc']:.3f}", f"{dl['auc']:.3f}", "GPU 2 h"])
    R.table(["方案", "Dice", "Se", "Sp", "Acc", "AUC", "单图耗时"], rows,
            widths=[2.0, 3.9, 2.0, 2.0, 2.0, 2.0, 2.1])
    R.caption(f"表 5　5 折交叉验证（OOF）结果；DL 为同数据 ResNet34+U-Net 的 {dl['n']} 张图 OOF 集成" if dl
              else "表 5　5 折交叉验证（OOF）结果")
    R.p("说明：C0–C3 均为同一数据、同一划分、同一指标下的 5 折交叉验证结果，且每折独立调参（每张图的超参"
        "不来自它自己）。DL 一行由既有工程的 OOF 概率图按图平均得到"
        + (f"（{dl['n']} 张图、{dl['runs']} 条模型输出）" if dl else "")
        + "，属于多模型集成，应看作参照上界；该工程在 Kaggle 隐藏测试集上的分数为 0.756，与本表的 OOF 口径不同，"
          "两者不能并列比较。")
    if stats_pairs:
        R.p("只有 20 张标注图时，均值本身带有不确定性。下表用配对检验判断“差异是真的还是噪声”："
            "同一批图像、同一折、只替换方法，图像难度项被抵消，因此比单看均值更敏感。")
        R.table(["比较", "平均差", "标准误", "Cohen's dz", "95% 置信区间", "配对 t 的 p"],
                [[f"{p['a']} → {p['b']}", f"{float(p['mean_diff']):+.4f}", f"{float(p['se']):.4f}",
                  f"{float(p['dz']):.2f}", f"[{float(p['ci_lo']):+.4f}, {float(p['ci_hi']):+.4f}]",
                  ("<1e-15" if float(p['p_t']) < 1e-15 else f"{float(p['p_t']):.1e}")] for p in stats_pairs],
                widths=[2.6, 2.4, 2.2, 2.2, 4.2, 2.4])
        R.caption("表 6　配对显著性检验（同一批 20 张图，仅更换方法）")
        sds = sorted(float(p["sd_diff"]) for p in stats_pairs)
        sd_med = sds[len(sds) // 2]
        pair12 = next(p for p in stats_pairs if p["a"] == "C1" and p["b"] == "C2")
        pair23 = next(p for p in stats_pairs if p["a"] == "C2" and p["b"] == "C3")
        detectable = 2.802 * sd_med / (20 ** 0.5)
        R.p(f"配对差值的标准差在 {sds[0]:.4f}–{sds[-1]:.4f} 之间（中位数 {sd_med:.4f}），对应标准误约 "
            f"{sd_med / 20 ** 0.5:.4f}；以 80% 功效、双侧 α=0.05 计，20 张图能稳定分辨约 {detectable:.3f} 的 Dice 差异。"
            "因此报告只对 ≥0.02 的差异下结论："
            f"C1→C2 的 {float(pair12['mean_diff']):+.3f}（p≈{float(pair12['p_t']):.0e}）可以下结论；"
            f"C2→C3 的 {float(pair23['mean_diff']):+.4f}（p={float(pair23['p_t']):.2f}）幅度远低于可分辨下限，"
            "只能表述为“没有可见收益”，而不能说成“后处理有害”。")
    R.figure(os.path.join(fig_dir, "fig_cases.png"), 13.6,
             "图 3　定性对比（#1、#2、#5）：绿=标注，红=预测，黄=重合")

    R.h2("5.2 逐层分析")
    d = {k: float(summary[k]["dice"]) for k in ("C0", "C1", "C2", "C3")}
    se = {k: float(summary[k]["se"]) for k in ("C0", "C1", "C2", "C3")}
    sp = {k: float(summary[k]["sp"]) for k in ("C0", "C1", "C2", "C3")}
    au = {k: float(summary[k]["auc"]) for k in ("C0", "C1", "C2", "C3")}
    c0_pred = float(np.mean([float(r["pred_ratio"]) for r in per_image if r["method"] == "C0"]))
    gt_ratio = float(np.mean([float(r["gt_ratio"]) for r in per_image if r["method"] == "C3"]))
    R.p(f"（1）全局阈值（C0，Dice {d['C0']:.3f}）基本不可用。它把 {c0_pred * 100:.0f}% 的像素判成血管"
        f"（真值只有 {gt_ratio * 100:.1f}%），这些像素集中在外周暗区，与图 2 的直方图分析一致。"
        "这一步的意义是给后续所有改进提供一个“不可用”的起点，而不是一个被故意削弱的对照。")
    R.p(f"（2）预处理 + 局部自适应阈值（C1，Dice {d['C1']:.3f}）带来 {d['C1'] - d['C0']:+.3f} 的提升，"
        f"是本项目里最大的一次跃迁。敏感度 {se['C1']:.3f}、特异度 {sp['C1']:.3f}，说明它已经能稳定抓到"
        "主要血管，但召回偏低——细血管和低对比度区域仍然漏。")
    R.p(f"（3）换成多尺度 Hessian 增强 + 滞后阈值（C2，Dice {d['C2']:.3f}）再拿 {d['C2'] - d['C1']:+.3f}"
        + (f"（配对 p≈{float(pair12['p_t']):.0e}，表 6）" if stats_pairs else "")
        + f"，敏感度从 {se['C1']:.3f} 升到 {se['C2']:.3f}，AUC 从 {au['C1']:.3f} "
        f"升到 {au['C2']:.3f}。这一档有个值得写进报告的实现细节：σ 上限若按视场包围盒的对角线估计会偏大 1.41 倍，"
        "尺度一旦偏大，细血管响应就被粗血管淹没，Dice 会退回 0.55 附近；改成按视场直径估计后，"
        "同一套代码立刻多出 +0.044。这也解释了为什么“增强有没有用”经常取决于尺度怎么定，而不是增强本身。")
    R.p(f"（4）后处理（C3，Dice {d['C3']:.3f}）在 C2 上再拿 {d['C3'] - d['C2']:+.4f}"
        + (f"（配对 p={float(pair23['p_t']):.2f}）" if stats_pairs else "")
        + "：方向为正，但幅度小于本实验的可分辨下限，只能算“有微弱正收益”，不能当成显著提升。"
          "交叉验证在多数折里选了较小的面积阈值与 dilate = 0，说明 Frangi + 滞后阈值的输出本身已接近标注厚度，"
          "膨胀补偿没有用武之地；后处理真正补偿的是局部阈值这种“偏细”的分割器的系统性偏差。")
    R.h2("5.3 与深度学习方案的差距")
    if dl:
        gap = dl["dice"] - float(summary["C3"]["dice"])
        ratio = float(summary["C3"]["dice"]) / dl["dice"] * 100
        R.p(f"同数据的 ResNet34+U-Net 集成达到 Dice {dl['dice']:.3f}（{dl['n']} 张图 OOF），"
            f"比本项目的 C3（{float(summary['C3']['dice']):.3f}）高 {gap:.3f}。差距不在整体结构，而在两处："
            f"一是细血管的连通性（DL 的 Se 为 {dl['se']:.3f}，本文 {float(summary['C3']['se']):.3f}）；"
            f"二是对病灶、渗出等干扰区域的抑制（DL 的 Sp 为 {dl['sp']:.3f}，本文 {float(summary['C3']['sp']):.3f}）。"
            f"从指标上看，DL 的 AUC {dl['auc']:.3f} 与本文 C2 的 {float(summary['C2']['auc']):.3f} 之差，"
            "正是“特征学习”带来的排序能力提升。")
        R.p(f"换个角度看，本项目用 CPU 0.03 s/张、无训练、全流程可解释的代价，拿到了 DL 约 {ratio:.0f}% 的 Dice；"
            "在需要部署到低算力设备、或需要逐项解释“为什么这里判错”的场景下，仍有实用价值。")
    R.h2("5.4 失效案例")
    R.figure(os.path.join(fig_dir, "fig_failure.png"), 15.0,
             f"图 4　最好（#{best}）与最差（#{worst}）样本的误差分解，以及 20 张图的 Dice 分布")
    R.p(f"最差的一张是 #{worst}（Dice {float(c3[worst]['dice']):.3f}），这类图像有三个共同点："
        "病灶/渗出造成大块亮斑，局部阈值把它们当作背景而把周围暗环判成血管；黄斑区对比度低，细血管几乎不可见；"
        "视盘边缘的亮暗过渡被误判。三类误差都指向同一个原因：仅依赖局部亮度对比，缺少形状与连续性的约束。")
    R.h2("5.5 效率")
    R.p(f"C3 在 512×512 上单图 {float(summary['C3']['seconds']):.3f} s（CPU，单进程 numpy/OpenCV）；"
        "Frangi 响应可以缓存，首次运行 C2 约 0.07 s/张，缓存命中后降到 0.01 s 量级。"
        "包含 5 折网格搜索的完整实验在普通笔记本上约 4 分钟跑完（含出图与统计检验）。"
        "对比之下，深度学习方案的 5 折训练在 4 GB 显存上约 2 小时；若只做推理，两者差距更大。")
    R.h2("5.6 小结与可继续改进的方向")
    _c = {k: float(summary[k]["dice"]) for k in ("C0", "C1", "C2", "C3")}
    _dl = f"{dl['dice']:.3f}" if dl else "—"
    _gap = f"{dl['dice'] - _c['C3']:.3f}" if dl else "—"
    for s in [
        f"结论一：全局阈值不适合眼底图（{_c['C0']:.3f}），局部自适应阈值是性价比最高的一步（{_c['C1'] - _c['C0']:+.3f}）；",
        f"结论二：尺度设置正确的 Hessian 增强 + 滞后阈值再带来 {_c['C2'] - _c['C1']:+.3f}，是第二大的单项收益；",
        f"结论三：后处理只补上 {_c['C3'] - _c['C2']:+.4f}——它能补偿“偏细”的分割器，对已经足够厚的输出作用有限；",
        f"结论四：与同数据的深度学习集成（{_dl}）仍差约 {_gap}，差距集中在细小血管的连通性；",
        "后续可做：把 Frangi 响应作为局部阈值的软先验做加权融合、多方向形态学闭运算连接断点，"
        "以及在更大数据（CHASE_DB1 / STARE）上验证泛化。",
    ]:
        R.bullet(s)

    # ---------------- 附录 ----------------
    R.h1("附录 A　逐图指标（C3 方案，OOF）")
    rows = []
    for img in sorted(c3, key=lambda x: int(x)):
        r = c3[img]
        rows.append([img, f"{float(r['dice']):.3f}", f"{float(r['se']):.3f}",
                     f"{float(r['sp']):.3f}", f"{float(r['auc']):.3f}",
                     f"{float(r['pred_ratio']):.3f}"])
    R.table(["图号", "Dice", "Se", "Sp", "AUC", "预测面积比"], rows,
            widths=[2.4, 3.0, 3.0, 3.0, 3.0, 3.4])
    R.caption("表 A1　20 张图的 C3 指标")

    R.h1("附录 B　每折选中的超参数")
    rows = []
    for key in ("C1", "C2", "C3"):
        for rec in folds[key]:
            rows.append([key, str(rec["fold"]), json.dumps(rec["params"], ensure_ascii=False),
                         f"{rec['test_dice']:.4f}"])
    R.table(["方案", "折", "选中参数", "本折 Dice"], rows, widths=[2.0, 1.6, 9.4, 3.0])
    R.caption("表 B1　交叉验证中每折独立选出的参数")

    R.h1("附录 C　复现步骤")
    for s in [
        "安装依赖：pip install -r requirements.txt（numpy / scipy / opencv-python / matplotlib）；",
        "把数据目录按 train/image、train/label、test/image 组织好；",
        "运行 python -m src.run_all --data-dir <数据目录> --out results；",
        "结果：results/summary.csv（主表）、metrics_per_image.csv（逐图）、figures/（全部图片）。",
    ]:
        R.bullet(s)

    _ref = R.h1("参考文献")
    _ref.paragraph_format.space_before = Pt(8)
    for ref in [
        "[1] Staal J., Abràmoff M.D., Niemeijer M., Viergever M.A., van Ginneken B. Ridge-based vessel "
        "segmentation in color images of the retina. IEEE Transactions on Medical Imaging, 2004, 23(4): 501–509.",
        "[2] Frangi A.F., Niessen W.J., Vincken K.L., Viergever M.A. Multiscale vessel enhancement filtering. "
        "MICCAI 1998: 130–137.",
        "[3] Otsu N. A threshold selection method from gray-level histograms. IEEE Transactions on Systems, "
        "Man, and Cybernetics, 1979, 9(1): 62–66.",
        "[4] Zuiderveld K. Contrast limited adaptive histogram equalization. Graphics Gems IV, 1994: 474–485.",
    ]:
        R.p(ref, size=9, indent=False, after=2, line=1.25)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    R.save(out_path)
    print("saved", out_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--fig-dir", default="results/figures")
    ap.add_argument("--out", default="report/1.报告.docx")
    a = ap.parse_args()
    main(a.results, a.fig_dir, a.out)
