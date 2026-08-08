#!/usr/bin/env python3
"""
可视化: 一个 128x128 C tile 需要从 A 和 B 加载什么数据到 shared memory?
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import numpy as np

_CN_FONT = None
for _fname in ["Microsoft YaHei", "SimHei"]:
    for _f in fm.fontManager.ttflist:
        if _f.name == _fname:
            _CN_FONT = _f; break
    if _CN_FONT: break
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT.name
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 演示参数
M_TILE = 128
N_TILE = 128
K_TOTAL = 64
K_TILE = 16
NUM_K_TILES = K_TOTAL // K_TILE  # 4

def draw_smem_loading():
    fig = plt.figure(figsize=(20, 12))

    # ====== 图 1: 全景 —— 一个 block 需要 A 和 B 的哪些数据 ======
    ax_main = fig.add_axes([0.02, 0.05, 0.96, 0.93])
    ax_main.axis("off")
    ax_main.set_xlim(0, 24)
    ax_main.set_ylim(0, 14)

    # 标题
    ax_main.text(12, 13.5, "一个 128x128 C Tile 需要从 A 和 B 加载什么到 Shared Memory?",
                ha="center", fontsize=16, fontweight="bold")
    ax_main.text(12, 12.8,
                f"C[128x128] = A[128x{K_TOTAL}] x B[{K_TOTAL}x128]   |   "
                f"K_tile=16,  共 {NUM_K_TILES} 次迭代",
                ha="center", fontsize=11, color="#555")

    # ---- A 矩阵 (左侧) ----
    # A 矩阵: M=128 行, K=64 列
    A_left = 0.5
    A_top = 11.5
    A_width = 7    # K 维度宽度
    A_height = 5.5  # M 维度高度 (128 行按比例)

    # A 背景
    ax_main.add_patch(FancyBboxPatch((A_left, A_top - A_height), A_width, A_height,
                                     boxstyle="round,pad=0.1",
                                     facecolor="#FFCDD2", edgecolor="#E53935", linewidth=2))
    ax_main.text(A_left + A_width/2, A_top + 0.3, "A 矩阵 (128 rows x K cols)",
                ha="center", fontsize=11, fontweight="bold", color="#E53935")

    # A 的行标签
    ax_main.text(A_left - 0.4, A_top - A_height/2, "128\nrows", ha="center",
                va="center", fontsize=9, color="#E53935", fontweight="bold",
                rotation=90)
    ax_main.text(A_left + A_width/2, A_top - A_height - 0.3, f"K = {K_TOTAL} cols",
                ha="center", fontsize=9, color="#E53935")

    # A 的 4 个 K-tile 竖条
    ktile_colors = ["#FFE082", "#FFCC80", "#FFAB91", "#EF9A9A"]
    for kt in range(NUM_K_TILES):
        x0 = A_left + kt * (A_width / NUM_K_TILES)
        w = A_width / NUM_K_TILES
        ax_main.add_patch(Rectangle((x0, A_top - A_height), w, A_height,
                                     facecolor=ktile_colors[kt], edgecolor="#333",
                                     linewidth=1.5, alpha=0.6))
        ax_main.text(x0 + w/2, A_top - A_height + 0.2,
                    f"K=[{kt*16}:{(kt+1)*16}]",
                    ha="center", fontsize=6, fontweight="bold", rotation=90)

    # 高亮当前 K-tile (kt=1)
    kt_highlight = 1
    xh = A_left + kt_highlight * (A_width / NUM_K_TILES)
    wh = A_width / NUM_K_TILES
    ax_main.add_patch(Rectangle((xh, A_top - A_height), wh, A_height,
                                 facecolor="none", edgecolor="#FF6F00",
                                 linewidth=4, zorder=10))
    ax_main.text(xh + wh/2, A_top - A_height - 0.6,
                "当前 K-tile\n加载到 smem",
                ha="center", fontsize=7, color="#FF6F00", fontweight="bold")

    # ---- B 矩阵 (右侧) ----
    B_left = 10
    B_top = 11.5
    B_width = 7
    B_height = 5.5

    ax_main.add_patch(FancyBboxPatch((B_left, B_top - B_height), B_width, B_height,
                                     boxstyle="round,pad=0.1",
                                     facecolor="#BBDEFB", edgecolor="#1E88E5", linewidth=2))
    ax_main.text(B_left + B_width/2, B_top + 0.3, "B 矩阵 (K rows x 128 cols)",
                ha="center", fontsize=11, fontweight="bold", color="#1E88E5")

    ax_main.text(B_left + B_width/2, B_top - B_height - 0.3, f"N = {N_TILE} cols",
                ha="center", fontsize=9, color="#1E88E5")
    ax_main.text(B_left - 0.4, B_top - B_height/2, f"K={K_TOTAL}\nrows", ha="center",
                va="center", fontsize=9, color="#1E88E5", fontweight="bold", rotation=90)

    # B 的 4 个 K-tile 横条
    for kt in range(NUM_K_TILES):
        y0 = B_top - (kt + 1) * (B_height / NUM_K_TILES)
        h = B_height / NUM_K_TILES
        ax_main.add_patch(Rectangle((B_left, y0), B_width, h,
                                     facecolor=ktile_colors[kt], edgecolor="#333",
                                     linewidth=1.5, alpha=0.6))
        ax_main.text(B_left + B_width/2, y0 + h/2,
                    f"K=[{kt*16}:{(kt+1)*16}]",
                    ha="center", fontsize=6, fontweight="bold")

    # 高亮 B 的相同 K-tile
    yh_b = B_top - (kt_highlight + 1) * (B_height / NUM_K_TILES)
    hh_b = B_height / NUM_K_TILES
    ax_main.add_patch(Rectangle((B_left, yh_b), B_width, hh_b,
                                 facecolor="none", edgecolor="#FF6F00",
                                 linewidth=4, zorder=10))

    # ---- 中间箭头: 加载到 Shared Memory ----
    arrow_y = 5
    # A -> A_buf
    ax_main.annotate("", xy=(A_left + A_width/2, arrow_y + 1.5),
                    xytext=(A_left + A_width/2, A_top - A_height - 0.5),
                    arrowprops=dict(arrowstyle="->", color=C_LOAD, lw=3))
    ax_main.text(A_left + A_width/2, arrow_y + 2.2,
                "加载 A[0:128, kb:kb+16]\n→ A_buf[128][16]",
                ha="center", fontsize=9, fontweight="bold", color="#E53935")

    # B -> B_buf
    ax_main.annotate("", xy=(B_left + B_width/2, arrow_y + 1.5),
                    xytext=(B_left + B_width/2, B_top - B_height - 0.5),
                    arrowprops=dict(arrowstyle="->", color=C_LOAD, lw=3))

    ax_main.text(B_left + B_width/2, arrow_y + 2.2,
                "加载 B[kb:kb+16, 0:128]\n→ B_buf[16][128]",
                ha="center", fontsize=9, fontweight="bold", color="#1E88E5")

    # ---- Shared Memory 区域 ----
    smem_y = 0.8
    smem_h = 3.5

    # A_buf
    rect_abuf = FancyBboxPatch((A_left + 0.3, smem_y), 3.2, smem_h,
                                boxstyle="round,pad=0.1",
                                facecolor="#C8E6C9", edgecolor="#2E7D32", linewidth=2.5)
    ax_main.add_patch(rect_abuf)
    ax_main.text(A_left + 0.3 + 1.6, smem_y + 2.7,
                "Shared Memory\nA_buf[128][16]",
                ha="center", fontsize=10, fontweight="bold", color="#1B5E20")
    ax_main.text(A_left + 0.3 + 1.6, smem_y + 1.3,
                "128 rows of A\n× 16 cols of K\n= 2048 FP16 = 4KB",
                ha="center", fontsize=8, color="#333")
    ax_main.text(A_left + 0.3 + 1.6, smem_y + 0.3,
                "A vertical slice:\nall 128 rows of this block\nx 16 cols of current K window",
                ha="center", fontsize=7.5, color="#555")

    # B_buf
    rect_bbuf = FancyBboxPatch((B_left + 1.5, smem_y), 4.5, smem_h,
                                boxstyle="round,pad=0.1",
                                facecolor="#BBDEFB", edgecolor="#1565C0", linewidth=2.5)
    ax_main.add_patch(rect_bbuf)
    ax_main.text(B_left + 1.5 + 2.25, smem_y + 2.7,
                "Shared Memory\nB_buf[16][128]",
                ha="center", fontsize=10, fontweight="bold", color="#0D47A1")
    ax_main.text(B_left + 1.5 + 2.25, smem_y + 1.3,
                "16 rows of K\n× 128 cols of B\n= 2048 FP16 = 4KB",
                ha="center", fontsize=8, color="#333")
    ax_main.text(B_left + 1.5 + 2.25, smem_y + 0.3,
                "B horizontal slice:\n16 rows of current K window\nx all 128 cols of this block",
                ha="center", fontsize=7.5, color="#555")

    # ---- WMMA 如何消费 smem ----
    wmma_y = 0.2
    ax_main.text(12, wmma_y - 0.4,
                "WMMA 消费: 每个 warp 从 A_buf 取 16 行(a[mi]), "
                "从 B_buf 取 16 列(b[ni]), 做 16x16x16 乘加 → 累加到 c[16x16]",
                ha="center", fontsize=9, fontweight="bold", color="#7B1FA2")

    # ---- 右侧总结 ----
    summary_x = 18
    summary_y = 11
    ax_main.text(summary_x, summary_y, "关键理解", fontsize=11, fontweight="bold", color="#333")
    summary_lines = [
        "1. C[128x128] = A[128xK] x B[Kx128]",
        "",
        "2. K 维度分 4 次迭代 (K_tile=16):",
        "   每次加载 A 的 128x16 子块",
        "   每次加载 B 的 16x128 子块",
        "",
        "3. Shared Memory 布局:",
        "   A_buf: 128行 x 16列 (所有行, K窗口)",
        "   B_buf: 16行 x 128列 (K窗口, 所有列)",
        "",
        "4. 双缓冲: A_buf[2], B_buf[2]",
        "   一个在读(WMMA计算), 一个在写(cp.async)",
        "",
        "5. 每个 warp 从 smem 读:",
        "   a[mi]: A_buf[wy*64+mi*16][0:16] (16x16)",
        "   b[ni]: B_buf[0:16][wx*64+ni*16] (16x16)",
        "   mma_sync: 16x16x16 → Tensor Core!",
        "",
        f"6. 总计 smem 用量 (双缓冲):",
        f"   A_buf[2][128][16] = {2*128*16*2//1024}KB (FP16)",
        f"   B_buf[2][16][128] = {2*16*128*2//1024}KB (FP16)",
        f"   合计 = {2*(128*16+16*128)*2//1024}KB",
    ]
    for i, line in enumerate(summary_lines):
        color = "#333"
        if line.startswith("   A_buf"):
            color = "#2E7D32"
        elif line.startswith("   B_buf"):
            color = "#1565C0"
        ax_main.text(summary_x, summary_y - 0.3 - i * 0.28, line,
                    fontsize=7.5, color=color, fontfamily="monospace")

    # C_LOAD 颜色定义
    # (定义在文件末尾引用之前)

    path = os.path.join(OUT_DIR, "gemm_v9_smem_loading.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[done] {path}")


C_LOAD = "#FF6F00"

if __name__ == "__main__":
    draw_smem_loading()
