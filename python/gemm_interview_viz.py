#!/usr/bin/env python3
"""
CUDA SGEMM Kernel 可视化 (gemm_interview.cu)
=============================================
可视化三层优化:
  1. 32x32 shared memory tiling
  2. 2x2 register blocking (每线程 4 个输出)
  3. Cooperative loading (16x16 线程协作填 smem)

输出:
  - gemm_cuda_overview.png    全景: block grid + 线程映射
  - gemm_cuda_loading.png     协作加载: A→As, B→Bs
  - gemm_cuda_compute.png     寄存器累加: 2x2 外积
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe

# ============================================================
# 字体配置 (复用已验证的)
# ============================================================
_CN_FONT = None
for _fname in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
    for _f in fm.fontManager.ttflist:
        if _f.name == _fname:
            _CN_FONT = _f
            break
    if _CN_FONT:
        break
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT.name
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 常量 (与 kernel 一致)
# ============================================================
TILE = 32
THREADS_X = 16
THREADS_Y = 16

# 颜色
C4 = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]  # 4 象限颜色
C4_LIGHT = ["#FFCDD2", "#BBDEFB", "#C8E6C9", "#FFE0B2"]
Q_LABELS = ["Q0 (c00)", "Q1 (c01)", "Q2 (c10)", "Q3 (c11)"]

OUT_DIR = os.path.dirname(__file__)


# ============================================================
# 图 1: 全景 — block grid + 线程映射
# ============================================================
def draw_overview():
    fig = plt.figure(figsize=(20, 12))

    # --- 左上: C 输出矩阵, block grid 划分 ---
    ax_c = fig.add_axes([0.04, 0.45, 0.40, 0.52])
    ax_c.set_title("C 输出矩阵 — Block Grid 划分\n"
                   f"gridDim = ((N+31)/32, (M+31)/32), 每 block 覆盖 32×32",
                   fontsize=13, fontweight="bold")
    # 画 8×8 的 block grid
    grid_rows, grid_cols = 8, 8
    for by in range(grid_rows):
        for bx in range(grid_cols):
            color = plt.cm.tab20((by * grid_cols + bx) % 20)
            rect = Rectangle((bx * TILE, by * TILE), TILE, TILE,
                             facecolor=color, edgecolor="white", linewidth=1, alpha=0.35)
            ax_c.add_patch(rect)
            if grid_rows <= 8:
                ax_c.text(bx * TILE + 16, by * TILE + 16,
                          f"B({bx},{by})", ha="center", va="center",
                          fontsize=6, fontweight="bold")

    ax_c.set_xlim(0, grid_cols * TILE)
    ax_c.set_ylim(grid_rows * TILE, 0)
    ax_c.set_xlabel("N (列)")
    ax_c.set_ylabel("M (行)")
    ax_c.set_aspect("equal")

    # --- 右上: 单个 Block 的 32×32 tile, 标注线程分工 ---
    ax_tile = fig.add_axes([0.50, 0.45, 0.46, 0.52])
    ax_tile.set_title("一个 Block 的 32×32 输出 Tile — 线程映射\n"
                      f"blockDim=(16,16), 256 threads, 每线程 4 个输出",
                      fontsize=13, fontweight="bold")

    # 4 个象限的背景
    quadrants = [
        (0, 0, "Q0\nc00"), (0, 16, "Q1\nc01"),
        (16, 0, "Q2\nc10"), (16, 16, "Q3\nc11"),
    ]
    for qi, (r0, c0, label) in enumerate(quadrants):
        ax_tile.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), 16, 16,
                                     facecolor=C4_LIGHT[qi], edgecolor=C4[qi],
                                     linewidth=2, alpha=0.5))
        ax_tile.text(c0 + 8, r0 + 8, label, ha="center", va="center",
                     fontsize=12, fontweight="bold", color=C4[qi], alpha=0.6)

    # 画一个特定线程的 4 个输出位置 (highlight 线程 (5, 3))
    hx, hy = 5, 3  # 高亮线程
    thread_outputs = [
        (hy, hx, 0), (hy, hx + 16, 1),    # row0 = ty,     col0/col1 = tx/tx+16
        (hy + 16, hx, 2), (hy + 16, hx + 16, 3),  # row1 = ty+16
    ]
    for r, c, qi in thread_outputs:
        ax_tile.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                     facecolor="#FFD700", edgecolor="#333333",
                                     linewidth=2.5, zorder=10))
        ax_tile.text(c, r, f"th({hx},{hy})", ha="center", va="center",
                     fontsize=6, color="#333", fontweight="bold", zorder=11)

    # 画其他线程的映射 (稀疏采样, 网格点)
    for ty in [0, 7, 15]:
        for tx in [0, 7, 15]:
            if tx == hx and ty == hy:
                continue
            for qi, (r, c) in enumerate([(ty, tx), (ty, tx + 16),
                                          (ty + 16, tx), (ty + 16, tx + 16)]):
                ax_tile.plot(c, r, "o", color=C4[qi], markersize=3, alpha=0.5)

    # 网格线
    for k in range(0, TILE + 1, 16):
        ax_tile.axhline(y=k - 0.5, color="#333", linewidth=2)
        ax_tile.axvline(x=k - 0.5, color="#333", linewidth=2)
    for k in range(0, TILE + 1):
        ax_tile.axhline(y=k - 0.5, color="#ccc", linewidth=0.3)
        ax_tile.axvline(x=k - 0.5, color="#ccc", linewidth=0.3)

    ax_tile.set_xlim(-1, TILE + 1)
    ax_tile.set_ylim(TILE + 1, -1)
    ax_tile.set_aspect("equal")

    # 标注 row0/row1/col0/col1 公式
    ax_tile.annotate(f"row0 = by*32 + ty\nrow1 = row0 + 16",
                     xy=(0, 0), xytext=(-3, -4), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="red"), color="red")
    ax_tile.annotate(f"col0 = bx*32 + tx\ncol1 = col0 + 16",
                     xy=(0, 0), xytext=(0, -8), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="blue"), color="blue")

    # --- 左下: 代码片段 ---
    ax_code = fig.add_axes([0.04, 0.02, 0.42, 0.38])
    ax_code.axis("off")
    code = (
        "// 线程索引与输出映射\n"
        "int tx = threadIdx.x, ty = threadIdx.y;\n"
        "int row0 = blockIdx.y * 32 + ty;     // rows  0..15\n"
        "int row1 = row0 + 16;                // rows 16..31\n"
        "int col0 = blockIdx.x * 32 + tx;     // cols  0..15\n"
        "int col1 = col0 + 16;                // cols 16..31\n"
        "\n"
        "// 每个线程负责 4 个输出 (2×2 register blocking)\n"
        "// c00 → C[row0][col0],  c01 → C[row0][col1]\n"
        "// c10 → C[row1][col0],  c11 → C[row1][col1]\n"
        "\n"
        "// 256 threads x 4 outputs = 1024 = 32x32  OK"
    )
    ax_code.text(0.02, 0.98, code, transform=ax_code.transAxes,
                 fontsize=9.5, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#ccc"))

    # --- 右下: 关键数字 ---
    ax_info = fig.add_axes([0.50, 0.02, 0.46, 0.38])
    ax_info.axis("off")
    info = (
        "Kernel 参数总览\n"
        "═══════════════\n\n"
        f"• Shared memory tile:   As[{TILE}][{TILE}] + Bs[{TILE}][{TILE}]\n"
        f"  = {TILE*TILE*2} floats = {TILE*TILE*2*4//1024} KB\n\n"
        f"• Thread block:  ({THREADS_X}, {THREADS_Y}) = {THREADS_X*THREADS_Y} threads\n\n"
        f"• Outputs per thread:  4  (2×2 register block)\n"
        f"• Outputs per block:   {THREADS_X*THREADS_Y*4} = {TILE}×{TILE}\n\n"
        f"• Elements loaded per thread per k_block:\n"
        f"  A: 4 floats  →  As[ty][tx], As[ty][tx+16],\n"
        f"                  As[ty+16][tx], As[ty+16][tx+16]\n"
        f"  B: 4 floats  →  Bs[ty][tx], Bs[ty][tx+16],\n"
        f"                  Bs[ty+16][tx], Bs[ty+16][tx+16]\n\n"
        f"• Inner product loop:  {TILE} iterations\n"
        f"  a0=As[ty][k], a1=As[ty+16][k]  ← 同一列\n"
        f"  b0=Bs[k][tx], b1=Bs[k][tx+16]  ← 同一行\n"
        f"  c00+=a0*b0  c01+=a0*b1\n"
        f"  c10+=a1*b0  c11+=a1*b1         ← 2×1 × 1×2 外积!"
    )
    ax_info.text(0.02, 0.98, info, transform=ax_info.transAxes,
                 fontsize=9.5, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="#FAFAFA", edgecolor="#ccc"))

    fig.suptitle("CUDA SGEMM Kernel — 全景 (Overview)",
                 fontsize=16, fontweight="bold", y=0.995)
    path = os.path.join(OUT_DIR, "gemm_cuda_overview.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[1/4] {path}")


# ============================================================
# 图 2: 协作加载 — A→As, B→Bs
# ============================================================
def draw_loading():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(22, 10))

    # 高亮线程
    hx, hy = 4, 3

    for ax, mat_name, smem_name, dim_label in [
        (ax_a, "A (全局内存)", "As[32][32] (共享内存)", "K"),
        (ax_b, "B (全局内存)", "Bs[32][32] (共享内存)", "K"),
    ]:
        ax.set_title(f"{mat_name}  →  {smem_name}\n"
                     f"thread({hx},{hy}) 加载 4 个元素到 smem 的 4 个位置",
                     fontsize=12, fontweight="bold")

        # 32×32 共享内存网格
        for r in range(TILE):
            for c in range(TILE):
                qi = (1 if c >= 16 else 0) + (2 if r >= 16 else 0)
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                        facecolor=C4_LIGHT[qi], edgecolor=C4[qi],
                                        linewidth=0.3, alpha=0.6))

        # 高亮线程加载的 4 个位置
        load_positions = [
            (hy, hx, 0), (hy, hx + 16, 1),
            (hy + 16, hx, 2), (hy + 16, hx + 16, 3),
        ]
        for r, c, qi in load_positions:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                    facecolor="#FFD700", edgecolor="#333",
                                    linewidth=3, zorder=10))
            label = f"({r},{c})"
            ax.text(c, r, label, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="#333", zorder=11)

        # 象限分隔线
        ax.axhline(y=15.5, color="#333", linewidth=2.5)
        ax.axvline(x=15.5, color="#333", linewidth=2.5)

        # 象限标签
        for qi, (r0, c0, qlabel) in enumerate([
            (0, 0, "Q0"), (0, 16, "Q1"), (16, 0, "Q2"), (16, 16, "Q3"),
        ]):
            ax.text(c0 + 8, r0 + 8, qlabel, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=C4[qi], alpha=0.35)

        ax.set_xlim(-1, TILE + 1)
        ax.set_ylim(TILE + 1, -1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # A 特化标注
    ax_a.annotate("As[ty][tx]\n= A[row0][kb+tx]",
                  xy=(hx, hy), xytext=(hx + 8, hy - 6),
                  arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                  fontsize=8, color="red")
    ax_a.annotate("As[ty][tx+16]\n= A[row0][kb+tx+16]",
                  xy=(hx + 16, hy), xytext=(hx + 22, hy - 6),
                  arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
                  fontsize=8, color="blue")
    ax_a.annotate("As[ty+16][tx]\n= A[row1][kb+tx]",
                  xy=(hx, hy + 16), xytext=(hx + 8, hy + 22),
                  arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                  fontsize=8, color="green")
    ax_a.annotate("As[ty+16][tx+16]\n= A[row1][kb+tx+16]",
                  xy=(hx + 16, hy + 16), xytext=(hx + 22, hy + 22),
                  arrowprops=dict(arrowstyle="->", color="orange", lw=1.5),
                  fontsize=8, color="orange")

    # B 特化标注
    ax_b.annotate("Bs[ty][tx]\n= B[kb+ty][col0]",
                  xy=(hx, hy), xytext=(hx + 8, hy - 6),
                  arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                  fontsize=8, color="red")
    ax_b.annotate("Bs[ty][tx+16]\n= B[kb+ty][col1]",
                  xy=(hx + 16, hy), xytext=(hx + 22, hy - 6),
                  arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
                  fontsize=8, color="blue")
    ax_b.annotate("Bs[ty+16][tx]\n= B[kb+ty+16][col0]",
                  xy=(hx, hy + 16), xytext=(hx + 8, hy + 22),
                  arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                  fontsize=8, color="green")
    ax_b.annotate("Bs[ty+16][tx+16]\n= B[kb+ty+16][col1]",
                  xy=(hx + 16, hy + 16), xytext=(hx + 22, hy + 22),
                  arrowprops=dict(arrowstyle="->", color="orange", lw=1.5),
                  fontsize=8, color="orange")

    fig.suptitle("Cooperative Loading — 256 线程协作填充 Shared Memory\n"
                 f"每个线程负责 4 个 A 元素 + 4 个 B 元素, "
                 f"共 {TILE*TILE*2} 个位置恰好被填满一次",
                 fontsize=14, fontweight="bold", y=1.01)

    fig.tight_layout(pad=3)
    path = os.path.join(OUT_DIR, "gemm_cuda_loading.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[2/4] {path}")


# ============================================================
# 图 3: 寄存器累加 — 2×2 外积
# ============================================================
def draw_compute():
    fig = plt.figure(figsize=(16, 10))

    # --- 左: Shared Memory 读端口 ---
    ax_smem = fig.add_axes([0.03, 0.15, 0.35, 0.75])
    ax_smem.set_title("Shared Memory 读取 (内层 k 循环)\n"
                      f"for k in 0..{TILE-1}: 读 As 的列, Bs 的行",
                      fontsize=12, fontweight="bold")

    # As 矩阵 (左半) — 高亮第 ty 行 + 第 ty+16 行, 第 k 列
    SMEM_H = 20
    for r in range(SMEM_H):
        for c in range(SMEM_H):
            color = "#E8E8E8"
            ax_smem.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                         facecolor=color, edgecolor="#ccc", linewidth=0.3))

    # As 区域标签
    ax_smem.text(10, -1.5, "As[32][32]", ha="center", fontsize=11, fontweight="bold")

    # 高亮: 第 k 列 (黄色), 第 ty 行 (红色), 第 ty+16 行 (蓝色)
    k_pos = 8
    ty_pos, ty2_pos = 5, 5 + 16
    for rr in range(32):
        ax_smem.add_patch(Rectangle((k_pos - 0.5, rr - 0.5), 1, 1,
                                     facecolor="#FFD700", edgecolor="#B8860B",
                                     linewidth=1.5, alpha=0.4, zorder=5))
    for cc in range(32):
        ax_smem.add_patch(Rectangle((cc - 0.5, ty_pos - 0.5), 1, 1,
                                     facecolor=C4_LIGHT[0], edgecolor=C4[0],
                                     linewidth=1.5, alpha=0.5, zorder=5))
        ax_smem.add_patch(Rectangle((cc - 0.5, ty2_pos - 0.5), 1, 1,
                                     facecolor=C4_LIGHT[2], edgecolor=C4[2],
                                     linewidth=1.5, alpha=0.5, zorder=5))
    # 交点高亮
    for (rr, label, clr) in [(ty_pos, "a0", C4[0]), (ty2_pos, "a1", C4[2])]:
        ax_smem.add_patch(Rectangle((k_pos - 0.5, rr - 0.5), 1, 1,
                                     facecolor=clr, edgecolor="#333",
                                     linewidth=3, zorder=10))
        ax_smem.text(k_pos, rr, label, ha="center", va="center",
                     fontsize=12, fontweight="bold", color="white", zorder=11)

    ax_smem.set_xlim(-1, 32)
    ax_smem.set_ylim(32, -1)
    ax_smem.set_aspect("equal")
    ax_smem.set_xticks([])
    ax_smem.set_yticks([])

    # 标注
    ax_smem.annotate("As[ty][k]\n= A[row0][kb+k]",
                     xy=(k_pos, ty_pos), xytext=(k_pos + 12, ty_pos + 12),
                     arrowprops=dict(arrowstyle="->", color=C4[0], lw=1.5),
                     fontsize=9, color=C4[0])
    ax_smem.annotate("As[ty+16][k]\n= A[row1][kb+k]",
                     xy=(k_pos, ty2_pos), xytext=(k_pos + 12, ty2_pos - 6),
                     arrowprops=dict(arrowstyle="->", color=C4[2], lw=1.5),
                     fontsize=9, color=C4[2])

    # --- 中: 2×2 外积计算 ---
    ax_fma = fig.add_axes([0.42, 0.25, 0.22, 0.55])
    ax_fma.axis("off")
    ax_fma.set_xlim(-3, 3)
    ax_fma.set_ylim(-3, 3)

    # 列向量 (a0, a1)^T
    ax_fma.add_patch(Rectangle((-2, -0.5), 1, 2, facecolor=C4[0], edgecolor="#333",
                                linewidth=2, alpha=0.5))
    ax_fma.text(-1.5, 0.3, "a0", ha="center", va="center", fontsize=18, fontweight="bold")
    ax_fma.text(-1.5, 1.3, "a1", ha="center", va="center", fontsize=18, fontweight="bold",
                color="white")

    # 乘号
    ax_fma.text(-0.5, 0.5, "×", ha="center", va="center", fontsize=20, fontweight="bold")

    # 行向量 (b0, b1)
    ax_fma.add_patch(Rectangle((-0.5, 0.5), 2, 1, facecolor=C4[1], edgecolor="#333",
                                linewidth=2, alpha=0.5))
    ax_fma.text(0.3, 1.2, "b0", ha="center", va="center", fontsize=18, fontweight="bold")
    ax_fma.text(1.3, 1.2, "b1", ha="center", va="center", fontsize=18, fontweight="bold",
                color="white")

    # 等于号
    ax_fma.text(2.2, 0.5, "=", ha="center", va="center", fontsize=20, fontweight="bold")

    # 2×2 结果矩阵
    result_cells = [
        (0.5, 0.5, "a0·b0", "c00", C4[0]),
        (1.5, 0.5, "a0·b1", "c01", "#8E24AA"),
        (0.5, 1.5, "a1·b0", "c10", "#8E24AA"),
        (1.5, 1.5, "a1·b1", "c11", C4[2]),
    ]
    for x, y, expr, reg, clr in result_cells:
        ax_fma.add_patch(Rectangle((x - 0.5, y - 0.5 + 1.5), 1, 1,
                                    facecolor=clr, edgecolor="#333", linewidth=2, alpha=0.5))
        ax_fma.text(x, y + 1.7, reg, ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white")

    # 大括号和标注
    ax_fma.annotate("列向量 2×1\n取自 As 同一列",
                    xy=(-2, 1.5), xytext=(-3.5, 2.2),
                    fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->", color="gray"))
    ax_fma.annotate("行向量 1×2\n取自 Bs 同一行",
                    xy=(0.5, 0.5), xytext=(1.0, -1.0),
                    fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->", color="gray"))

    # --- 右: 代码 + 循环展开 ---
    ax_code = fig.add_axes([0.68, 0.15, 0.30, 0.75])
    ax_code.axis("off")
    code = (
        "内层循环 (寄存器累加)\n"
        "══════════════════════\n\n"
        "for (int k = 0; k < 32; k++) {\n\n"
        "  // 从 shared memory 读取\n"
        "  float a0 = As[ty][k];\n"
        f"  float a1 = As[ty+16][k];\n"
        "      ↑ 同一列, 上下两个元素\n\n"
        "  float b0 = Bs[k][tx];\n"
        f"  float b1 = Bs[k][tx+16];\n"
        "      ↑ 同一行, 左右两个元素\n\n"
        "  // 2×1 × 1×2 = 2×2 外积\n"
        "  c00 += a0 * b0;\n"
        "  c01 += a0 * b1;\n"
        "  c10 += a1 * b0;\n"
        "  c11 += a1 * b1;\n"
        "}\n\n"
        "──────────────────────\n"
        "每次迭代:  4 reads + 4 FMAs\n"
        "32 次迭代: 128 reads + 128 FMAs\n"
        "→ 4 个输出元素\n"
        "→ 读:FMA = 1:1 (完美平衡!)\n\n"
        "──────────────────────\n"
        "关键: a0, a1 沿 K 方向复用\n"
        "      b0, b1 沿 K 方向复用\n"
        "      只用 2+2 个寄存器就能\n"
        "      算 4 个 FMA!"
    )
    ax_code.text(0.02, 0.98, code, transform=ax_code.transAxes,
                 fontsize=9.5, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#ccc"))

    fig.suptitle("Register Blocking — 2×2 外积 (每线程 4 个寄存器累加器)",
                 fontsize=15, fontweight="bold", y=1.005)

    path = os.path.join(OUT_DIR, "gemm_cuda_compute.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[3/4] {path}")


# ============================================================
# 图 4: k_block 滑动动画
# ============================================================
def draw_animation():
    """K 维度上 block 滑动, C tile 逐步累加"""
    from matplotlib.animation import FuncAnimation

    fig, (ax_ab, ax_c) = plt.subplots(1, 2, figsize=(16, 7))

    # 显示 K=128, 单个 block 迭代 k_block=0,32,64,96
    K_TOTAL = 128
    M_BLOCK, N_BLOCK = 32, 32
    k_steps = list(range(0, K_TOTAL, TILE))  # [0, 32, 64, 96]

    def update(frame_idx):
        ax_ab.clear()
        ax_c.clear()

        k_block = k_steps[frame_idx]

        # --- 左: A 的一行 + B 的一列, 标注当前 k_block 窗口 ---
        ax_ab.set_xlim(-1, K_TOTAL + 2)
        ax_ab.set_ylim(-5, 35)
        ax_ab.set_title(f"A 的一行 (共 M 行) + B 的一列 (共 N 列)\n"
                        f"k_block = {k_block}, 窗口 [{k_block}, {k_block + TILE})",
                        fontsize=12, fontweight="bold")

        # A 行: 显示 K 维度上的元素
        for kk in range(K_TOTAL):
            in_window = k_block <= kk < k_block + TILE
            color = C4[0] if in_window else "#E0E0E0"
            alpha = 0.8 if in_window else 0.3
            ax_ab.add_patch(Rectangle((kk - 0.5, 4.5), 1, 2,
                                       facecolor=color, edgecolor="none", alpha=alpha))
        ax_ab.text(K_TOTAL / 2, 6.5, "A[i][:] — K 维度",
                   ha="center", fontsize=10)
        ax_ab.add_patch(Rectangle((k_block - 0.5, 4.5), TILE, 2,
                                   facecolor="none", edgecolor=C4[0],
                                   linewidth=3, zorder=10))

        # B 列: 显示 K 维度上的元素
        for kk in range(K_TOTAL):
            in_window = k_block <= kk < k_block + TILE
            color = C4[1] if in_window else "#E0E0E0"
            alpha = 0.8 if in_window else 0.3
            ax_ab.add_patch(Rectangle((kk - 0.5, 0), 1, 2,
                                       facecolor=color, edgecolor="none", alpha=alpha))
        ax_ab.text(K_TOTAL / 2, 1.8, "B[:][j] — K 维度",
                   ha="center", fontsize=10)
        ax_ab.add_patch(Rectangle((k_block - 0.5, 0), TILE, 2,
                                   facecolor="none", edgecolor=C4[1],
                                   linewidth=3, zorder=10))

        # K 维度迭代指示器
        for idx, ks in enumerate(k_steps):
            y_pos = 8.5
            marker = "▼" if ks == k_block else "│"
            clr = "red" if ks == k_block else "#999"
            fs = 14 if ks == k_block else 8
            ax_ab.text(ks + TILE / 2, y_pos, marker, ha="center", fontsize=fs,
                       color=clr, fontweight="bold")
            ax_ab.text(ks + TILE / 2, y_pos + 0.8, f"k=0" if ks == 0 else f"k={ks}",
                       ha="center", fontsize=7, color=clr)

        ax_ab.set_ylim(-2, 10)
        ax_ab.set_yticks([])

        # --- 右: C tile 的累加进度 ---
        ax_c.set_title(f"C tile 累加进度 (已完成 {frame_idx + 1}/{len(k_steps)} 个 k_block)",
                       fontsize=12, fontweight="bold")
        # 32×32 网格, 色彩强度 = 已完成的 k_block 比例
        progress = (frame_idx + 1) / len(k_steps)
        green = plt.cm.Greens(0.2 + 0.6 * progress)
        for r in range(TILE):
            for c in range(TILE):
                ax_c.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                          facecolor=green, edgecolor="#999",
                                          linewidth=0.3, alpha=0.8))

        ax_c.set_xlim(-3, TILE + 3)
        ax_c.set_ylim(TILE + 3, -3)
        ax_c.set_aspect("equal")
        ax_c.set_xticks([])
        ax_c.set_yticks([])

        # 进度文字
        ax_c.text(TILE / 2, TILE + 1.5,
                  f"C[row0:row1, col0:col1] += A[row0:row1, {k_block}:{k_block+TILE}]\n"
                  f"                          × B[{k_block}:{k_block+TILE}, col0:col1]",
                  ha="center", fontsize=9, fontweight="bold",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        return [ax_ab, ax_c]

    anim = FuncAnimation(fig, update, frames=len(k_steps), interval=800, blit=False)

    path = os.path.join(OUT_DIR, "gemm_cuda_animation.gif")
    writer = matplotlib.animation.PillowWriter(fps=1.5)
    anim.save(path, writer=writer, dpi=80)
    plt.close(fig)
    print(f"[4/4] {path}  ({len(k_steps)} frames)")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("CUDA SGEMM Kernel Visualization")
    print(f"  TILE={TILE}, blockDim=({THREADS_X},{THREADS_Y})")
    print(f"  Output dir: {OUT_DIR}\n")

    draw_overview()
    draw_loading()
    draw_compute()
    draw_animation()

    print("\nDone. Generated:")
    print("  gemm_cuda_overview.png   — block grid + 线程映射")
    print("  gemm_cuda_loading.png    — 协作加载 As/Bs")
    print("  gemm_cuda_compute.png    — 寄存器 2×2 外积")
    print("  gemm_cuda_animation.gif  — k_block 滑动动画")
