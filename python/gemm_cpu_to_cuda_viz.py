#!/usr/bin/env python3
"""
CPU tiled GEMM → CUDA SGEMM 循环映射可视化
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# 字体
_CN_FONT = None
for _fname in ["Microsoft YaHei", "SimHei"]:
    for _f in fm.fontManager.ttflist:
        if _f.name == _fname:
            _CN_FONT = _f; break
    if _CN_FONT: break
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT.name
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = __file__.rsplit("\\", 1)[0] if "\\" in __file__ else "."

# ============================================================
def draw_mapping():
    fig = plt.figure(figsize=(22, 14))

    # ====== 顶部: CPU 六层循环 ======
    ax_cpu = fig.add_axes([0.02, 0.72, 0.45, 0.26])
    ax_cpu.axis("off")
    ax_cpu.set_xlim(0, 10); ax_cpu.set_ylim(0, 10)

    cpu_loops = [
        (0.5, 9.0, "for (int i = 0; i < M; i += 32)",        "#E53935", "C 行块 (outermost)"),
        (1.5, 8.2, "  for (int j = 0; j < K; j += 32)",      "#1E88E5", "C 列块"),
        (2.5, 7.4, "    for (int k = 0; k < N; k += 32)",    "#43A047", "归约维 K (innermost)"),
        (3.5, 6.6, "      for (int ii = i; ii < i+32; ii++)", "#E53935", "块内行 (串行)"),
        (4.5, 5.8, "        for (int jj = j; jj < j+32; jj++)","#1E88E5","块内列 (串行)"),
        (5.5, 5.0, "          for (int kk = k; kk < k+32; kk++)","#43A047","块内归约 (串行)"),
    ]
    for x, y, text, color, note in cpu_loops:
        ax_cpu.text(x, y, text, fontsize=11, color=color, fontweight="bold")
        ax_cpu.text(x + 4.5, y, f"// {note}", fontsize=8, color="#999")

    ax_cpu.text(5, 9.6, "CPU: 1 个线程串行执行 6 层循环", fontsize=14,
                fontweight="bold", ha="center")
    ax_cpu.text(5, 4.2, "每次迭代访问全局内存 (DRAM)\n延迟 ~200-500 cycles", fontsize=10,
                ha="center", color="#999")

    # 大括号分组
    ax_cpu.add_patch(FancyBboxPatch((0, 7.0), 8, 2.6, boxstyle="round,pad=0.1",
                                     facecolor="#FFCDD2", edgecolor="#E53935",
                                     linewidth=2, alpha=0.3))
    ax_cpu.text(0.2, 9.3, "外层: 块级迭代\n(block-level tiling)", fontsize=9, color="#E53935")

    ax_cpu.add_patch(FancyBboxPatch((0, 4.8), 8, 2.2, boxstyle="round,pad=0.1",
                                     facecolor="#BBDEFB", edgecolor="#1E88E5",
                                     linewidth=2, alpha=0.3))
    ax_cpu.text(0.2, 6.9, "内层: 块内朴素乘加\n(BxB naive GEMM)", fontsize=9, color="#1E88E5")


    # ====== 顶部右: CUDA 对应 ======
    ax_cuda = fig.add_axes([0.50, 0.72, 0.48, 0.26])
    ax_cuda.axis("off")
    ax_cuda.set_xlim(0, 10); ax_cuda.set_ylim(0, 10)

    cuda_items = [
        (0.5, 9.0, "blockIdx.y  ← i 的并行化",   "#E53935", "每个 block 处理一个 32 行的 tile"),
        (1.5, 8.2, "blockIdx.x  ← j 的并行化",   "#1E88E5", "每个 block 处理一个 32 列的 tile"),
        (2.5, 7.4, "for k_block  ← k (一样!)",   "#43A047", "外层串行迭代, 每次加载 32×32 tile 到 smem"),
        (3.5, 6.6, "threadIdx.y (ty) ← ii",       "#E53935", "16 个线程覆盖 32 行 (每线程 2 行)"),
        (4.5, 5.8, "threadIdx.x (tx) ← jj",       "#1E88E5", "16 个线程覆盖 32 列 (每线程 2 列)"),
        (5.5, 5.0, "for k in 0..31 ← kk",         "#43A047", "在 shared memory 上迭代 (超快!)"),
    ]
    for x, y, text, color, note in cuda_items:
        ax_cuda.text(x, y, text, fontsize=11, color=color, fontweight="bold")
        ax_cuda.text(x + 4.5, y, f"// {note}", fontsize=8, color="#999")

    ax_cuda.text(5, 9.6, "CUDA: 256 个线程并行执行", fontsize=14,
                 fontweight="bold", ha="center")
    ax_cuda.text(5, 4.2, "Shared Memory 延迟 ~20-30 cycles\nRegister 延迟 ~1 cycle", fontsize=10,
                 ha="center", color="#999")

    ax_cuda.add_patch(FancyBboxPatch((0, 6.6), 8.5, 2.9, boxstyle="round,pad=0.1",
                                      facecolor="#C8E6C9", edgecolor="#43A047",
                                      linewidth=2, alpha=0.3))
    ax_cuda.text(0.2, 9.3, "Shared Memory Tiling + Cooperative Load",
                 fontsize=9, color="#43A047")

    ax_cuda.add_patch(FancyBboxPatch((0, 4.8), 8.5, 1.8, boxstyle="round,pad=0.1",
                                      facecolor="#BBDEFB", edgecolor="#1E88E5",
                                      linewidth=2, alpha=0.3))
    ax_cuda.text(0.2, 6.5, "Register Blocking (2x2 per thread)", fontsize=9, color="#1E88E5")


    # ====== 中部: 详细映射图 ======
    ax_map = fig.add_axes([0.02, 0.10, 0.96, 0.58])
    ax_map.axis("off")
    ax_map.set_xlim(0, 30); ax_map.set_ylim(0, 20)

    # --- 左列: CPU 概念 ---
    cpu_x = 1.5
    ax_map.text(cpu_x, 18.5, "CPU (1 thread)", fontsize=16, fontweight="bold",
                ha="center", color="#555")

    # 32×32 C 块, 32×32 个点 = 1024 次串行计算
    cpu_tile_x = cpu_x - 2
    cpu_tile_y = 12
    for r in range(32):
        for c in range(32):
            ax_map.add_patch(Rectangle((cpu_tile_x + c*0.12, cpu_tile_y + r*0.12),
                                        0.11, 0.11, facecolor="#E53935", edgecolor="none",
                                        alpha=0.15))

    # 箭头: 1 thread → 1024 outputs
    ax_map.annotate("", xy=(cpu_tile_x + 2, cpu_tile_y - 0.5),
                    xytext=(cpu_tile_x + 2, cpu_tile_y + 4.5),
                    arrowprops=dict(arrowstyle="->", color="#E53935", lw=2))
    ax_map.text(cpu_tile_x + 2, cpu_tile_y + 4.8, "1 个线程\n串行计算全部\n32×32 = 1024 个元素",
                ha="center", fontsize=9, color="#E53935")

    ax_map.text(cpu_tile_x + 2, cpu_tile_y - 1.0, "C 的一个 32×32 Tile",
                ha="center", fontsize=9, color="#555")

    # [---> 对应 --->]
    arrow_y = 15.5
    ax_map.annotate("", xy=(20, arrow_y), xytext=(9, arrow_y),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=3,
                                    connectionstyle="arc3,rad=0"))
    ax_map.text(14.5, arrow_y + 0.5, "并行化映射", ha="center", fontsize=12,
                fontweight="bold", color="#333")

    # --- 右列: CUDA 概念 ---
    cuda_x = 22.5
    ax_map.text(cuda_x, 18.5, "CUDA (256 threads/block)", fontsize=16,
                fontweight="bold", ha="center", color="#555")

    # 32×32 grid 分成 4 个象限, 每个线程负责 4 个元素
    cuda_tile_x = cuda_x - 2
    cuda_tile_y = 12

    quad_colors = ["#FFCDD2", "#BBDEFB", "#C8E6C9", "#FFE0B2"]
    quad_labels = ["Q0", "Q1", "Q2", "Q3"]
    for qi, (r0, c0) in enumerate([(0,0), (0,16), (16,0), (16,16)]):
        ax_map.add_patch(Rectangle((cuda_tile_x + c0*0.12, cuda_tile_y + r0*0.12),
                                    1.92, 1.92,
                                    facecolor=quad_colors[qi], edgecolor="#555",
                                    linewidth=1.5, alpha=0.5))
        ax_map.text(cuda_tile_x + c0*0.12 + 0.96, cuda_tile_y + r0*0.12 + 0.96,
                    quad_labels[qi], ha="center", va="center", fontsize=10,
                    fontweight="bold")

    # 高亮一个线程的 4 个输出
    htx, hty = 5, 3
    for qi, (r, c) in enumerate([(hty, htx), (hty, htx+16),
                                   (hty+16, htx), (hty+16, htx+16)]):
        ax_map.add_patch(Rectangle((cuda_tile_x + c*0.12, cuda_tile_y + r*0.12),
                                    0.22, 0.22,
                                    facecolor="#FFD700", edgecolor="#333",
                                    linewidth=2.5, zorder=10))
        ax_map.text(cuda_tile_x + c*0.12 + 0.11, cuda_tile_y + r*0.12 + 0.11,
                    f"c{['00','01','10','11'][qi]}", ha="center", va="center",
                    fontsize=5, color="#333", zorder=11)

    # 标注
    ax_map.annotate(f"thread({htx},{hty})\n计算 4 个元素",
                    xy=(cuda_tile_x + htx*0.12, cuda_tile_y + hty*0.12),
                    xytext=(cuda_tile_x - 3, cuda_tile_y + 1),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5),
                    fontsize=8, fontweight="bold", color="#333",
                    bbox=dict(boxstyle="round", facecolor="#FFE082", alpha=0.7))

    ax_map.text(cuda_tile_x + 2, cuda_tile_y - 1.0,
                "同一 32×32 Tile\n256 线程并行, 每线程 4 元素",
                ha="center", fontsize=9, color="#555")

    # --- 底部: 内存层次对比 ---
    mem_y = 8
    # CPU 内存
    ax_map.add_patch(FancyBboxPatch((1, mem_y), 6.5, 3.5, boxstyle="round,pad=0.2",
                                     facecolor="#FFCDD2", edgecolor="#E53935", lw=2))
    ax_map.text(4.25, mem_y + 3.0, "CPU 内存层次", fontsize=11, fontweight="bold",
                ha="center", color="#E53935")
    ax_map.text(4.25, mem_y + 2.2, "所有数据在 DRAM\n"
                "每次 FMA: 从 DRAM 读 A,B, 写 C\n"
                "≈ 3 次 DRAM 访问 / FMA",
                fontsize=8.5, ha="center", color="#555")

    # CUDA 内存
    ax_map.add_patch(FancyBboxPatch((10, mem_y), 7.5, 3.5, boxstyle="round,pad=0.2",
                                     facecolor="#C8E6C9", edgecolor="#43A047", lw=2))
    ax_map.text(13.75, mem_y + 3.0, "CUDA 内存层次", fontsize=11, fontweight="bold",
                ha="center", color="#43A047")
    ax_map.text(13.75, mem_y + 2.2,
                "Global Mem → Shared Mem (协作加载)\n"
                "Shared Mem → Register (内层 k 循环)\n"
                "Register → Register (2×2 FMA)\n"
                "DRAM 访问减少 ~32x!",
                fontsize=8.5, ha="center", color="#555")

    # CUDA 特定的 shared memory 说明
    ax_map.add_patch(FancyBboxPatch((20, mem_y), 8, 3.5, boxstyle="round,pad=0.2",
                                     facecolor="#E1BEE7", edgecolor="#8E24AA", lw=2))
    ax_map.text(24, mem_y + 3.0, "2×2 Register Blocking", fontsize=11,
                fontweight="bold", ha="center", color="#8E24AA")
    ax_map.text(24, mem_y + 2.2,
                "for k in 0..31:\n"
                "  a0=As[ty][k]  a1=As[ty+16][k]\n"
                "  b0=Bs[k][tx]  b1=Bs[k][tx+16]\n"
                "  c00+=a0*b0  c01+=a0*b1\n"
                "  c10+=a1*b0  c11+=a1*b1\n"
                "→ 4 reads + 4 FMAs / iteration",
                fontsize=8.5, ha="center", color="#555")

    # ====== 底部: 对照表 ======
    table_y = 4.5
    ax_map.text(15, table_y, "循环变量 —— 对应关系", fontsize=13, fontweight="bold",
                ha="center")

    # 表头
    cols = [
        (3, "CPU 循环"),
        (9, "CUDA 对应"),
        (15, "并行度"),
        (20, "数据位置"),
        (25, "延迟"),
    ]
    for x, label in cols:
        ax_map.text(x, table_y - 0.8, label, fontsize=9, fontweight="bold",
                    ha="center", color="#333")

    ax_map.axhline(y=table_y - 1.2, xmin=0.05, xmax=0.95, color="#ccc", lw=1)

    rows = [
        ("for i (C行块)", "blockIdx.y", "1 block / tile", "—", "—"),
        ("for j (C列块)", "blockIdx.x", "1 block / tile", "—", "—"),
        ("for k (归约)",   "for k_block", "串行 ~K/32 次", "Global → Shared", "~200→20 cyc"),
        ("for ii (行)",    "threadIdx.y (ty)", "16 线程并行", "Shared → Reg", "~20 cyc"),
        ("for jj (列)",    "threadIdx.x (tx)", "16 线程并行", "Shared → Reg", "~20 cyc"),
        ("for kk (归约)",  "for k (0..31)",    "32 次迭代",     "Reg → Reg", "~1 cyc"),
    ]
    for ri, (cpu, cuda, para, mem, lat) in enumerate(rows):
        y = table_y - 1.8 - ri * 0.5
        bg = "#F5F5F5" if ri % 2 == 0 else "white"
        ax_map.add_patch(Rectangle((1.5, y - 0.2), 26, 0.45, facecolor=bg, edgecolor="none", zorder=-1))
        ax_map.text(3, y, cpu, fontsize=9, ha="center", color="#E53935")
        ax_map.text(9, y, cuda, fontsize=9, ha="center", color="#1E88E5", fontweight="bold")
        ax_map.text(15, y, para, fontsize=8.5, ha="center", color="#555")
        ax_map.text(20, y, mem, fontsize=8.5, ha="center", color="#43A047")
        ax_map.text(25, y, lat, fontsize=8.5, ha="center", color="#999")

    ax_map.text(1.5, table_y + 0.3, "CPU vs CUDA 逐层对应 (M=N=K=1024, TILE=32)",
                fontsize=10, color="#555")

    fig.suptitle("CPU Tiled GEMM → CUDA SGEMM Kernel: 循环映射",
                 fontsize=17, fontweight="bold", y=0.99)

    out_path = f"{OUT_DIR}/gemm_cpu_to_cuda.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    draw_mapping()
