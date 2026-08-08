#!/usr/bin/env python3
"""
GPU 的一个 k_tile 步骤详解:
  A 的哪部分 × B 的哪部分 → 累加到 C 的哪部分

展示:
  1. 全局矩阵 A[M,K], B[K,N], C[M,N] 中当前 block 和 K 窗口的位置
  2. Shared Memory 中 A_buf[128][16] 和 B_buf[16][128] 的内容
  3. 4 个 warp 各自从 smem 读哪些 16x16 块, 算到 C 的哪个 64x64 区域
  4. 矩阵乘法公式: C[i,j] = sum_k A[i,k] * B[k,j]
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon
import matplotlib.patheffects as pe

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

# 参数
M = 384   # 总行数 (3 blocks of 128)
N = 384   # 总列数 (3 blocks of 128)
K = 64    # K 维度 (4 k_tiles of 16)
BLOCK_M = 128
BLOCK_N = 128
K_TILE = 16
WMMA_MN = 16

# 当前演示: block (1,1), k_tile=1
CUR_BLOCK_Y = 1
CUR_BLOCK_X = 1
CUR_K_TILE = 1
KB = CUR_K_TILE * K_TILE  # 16

# 颜色
C_ABLOCK = "#E53935"   # A 块
C_BBLOCK = "#1E88E5"   # B 块
C_CBLOCK = "#8E24AA"   # C 块
C_KWINDOW = "#FF6F00"  # K 窗口高亮
C_WARP = ["#FFCDD2", "#BBDEFB", "#C8E6C9", "#FFE0B2"]
C_WARP_E = ["#E53935", "#1565C0", "#2E7D32", "#E65100"]

def draw_one_step():
    fig = plt.figure(figsize=(24, 16))

    # ===== 顶部大图：三个全局矩阵 =====
    # A 矩阵 (M x K)
    ax_a = fig.add_axes([0.02, 0.55, 0.25, 0.42])
    ax_a.set_title("A 矩阵 (M=384 rows x K=64 cols)\n"
                   f"highlight: rows for block_y={CUR_BLOCK_Y}, K window [{KB}:{KB+K_TILE}]",
                   fontsize=9, fontweight="bold")

    # 画 A 矩阵轮廓
    A_h = M  # 384
    A_w = K  # 64
    ax_a.set_xlim(-2, A_w + 2)
    ax_a.set_ylim(A_h + 2, -2)
    ax_a.set_aspect("equal")

    # A 的背景 (浅灰)
    ax_a.add_patch(Rectangle((-0.5, -0.5), A_w, A_h, facecolor="#F5F5F5",
                              edgecolor="#999", linewidth=1))

    # A 的所有 row blocks 的水平线
    for bi in range(0, M + 1, BLOCK_M):
        ax_a.axhline(y=bi - 0.5, color="#999", linewidth=0.5, linestyle="--")
    # K tile 竖线
    for kt in range(0, K + 1, K_TILE):
        ax_a.axvline(x=kt - 0.5, color="#999", linewidth=0.5, linestyle="--")

    # 高亮当前 block 的行 (128 rows)
    r_start = CUR_BLOCK_Y * BLOCK_M
    ax_a.add_patch(Rectangle((-0.5, r_start - 0.5), A_w, BLOCK_M,
                              facecolor="#FFCDD2", edgecolor=C_ABLOCK,
                              linewidth=2.5, alpha=0.5))
    ax_a.text(A_w/2, r_start + BLOCK_M/2, f"block_y={CUR_BLOCK_Y}\n128 rows",
             ha="center", va="center", fontsize=8, fontweight="bold", color=C_ABLOCK)

    # 高亮当前 K 窗口 (16 cols)
    ax_a.add_patch(Rectangle((KB - 0.5, -0.5), K_TILE, A_h,
                              facecolor="none", edgecolor=C_KWINDOW,
                              linewidth=2, alpha=0.6, linestyle="--", zorder=5))
    ax_a.text(KB + K_TILE/2, -1.5, f"kb={KB}", ha="center", fontsize=7,
             color=C_KWINDOW, fontweight="bold")

    # 交叉区域 = 真正加载到 smem 的！
    ax_a.add_patch(Rectangle((KB - 0.5, r_start - 0.5), K_TILE, BLOCK_M,
                              facecolor=C_ABLOCK, edgecolor="#333",
                              linewidth=4, alpha=0.8, zorder=10))
    ax_a.text(KB + K_TILE/2, r_start + BLOCK_M/2,
             "A_buf[128][16]\nloaded to smem!",
             ha="center", va="center", fontsize=9, fontweight="bold",
             color="white", zorder=11)

    ax_a.set_xticks([0, 16, 32, 48, 64])
    ax_a.set_yticks([0, 128, 256, 384])
    ax_a.set_xlabel("K dimension")
    ax_a.set_ylabel("M dimension")

    # B 矩阵 (K x N)
    ax_b = fig.add_axes([0.30, 0.55, 0.25, 0.42])
    ax_b.set_title("B 矩阵 (K=64 rows x N=384 cols)\n"
                   f"highlight: K window [{KB}:{KB+K_TILE}], cols for block_x={CUR_BLOCK_X}",
                   fontsize=9, fontweight="bold")

    B_h = K   # 64
    B_w = N   # 384
    ax_b.set_xlim(-2, B_w + 2)
    ax_b.set_ylim(B_h + 2, -2)
    ax_b.set_aspect("equal")

    ax_b.add_patch(Rectangle((-0.5, -0.5), B_w, B_h, facecolor="#F5F5F5",
                              edgecolor="#999", linewidth=1))

    for bj in range(0, N + 1, BLOCK_N):
        ax_b.axvline(x=bj - 0.5, color="#999", linewidth=0.5, linestyle="--")
    for kt in range(0, K + 1, K_TILE):
        ax_b.axhline(y=kt - 0.5, color="#999", linewidth=0.5, linestyle="--")

    # 高亮当前 block 的列 (128 cols)
    c_start = CUR_BLOCK_X * BLOCK_N
    ax_b.add_patch(Rectangle((c_start - 0.5, -0.5), BLOCK_N, B_h,
                              facecolor="#BBDEFB", edgecolor=C_BBLOCK,
                              linewidth=2.5, alpha=0.5))
    ax_b.text(c_start + BLOCK_N/2, B_h/2, f"block_x={CUR_BLOCK_X}\n128 cols",
             ha="center", va="center", fontsize=8, fontweight="bold", color=C_BBLOCK)

    # 高亮 K 窗口 (16 rows)
    ax_b.add_patch(Rectangle((-0.5, KB - 0.5), B_w, K_TILE,
                              facecolor="none", edgecolor=C_KWINDOW,
                              linewidth=2, alpha=0.6, linestyle="--", zorder=5))
    ax_b.text(-1.5, KB + K_TILE/2, f"kb={KB}", ha="center", va="center",
             fontsize=7, color=C_KWINDOW, fontweight="bold", rotation=90)

    # 交叉区域
    ax_b.add_patch(Rectangle((c_start - 0.5, KB - 0.5), BLOCK_N, K_TILE,
                              facecolor=C_BBLOCK, edgecolor="#333",
                              linewidth=4, alpha=0.8, zorder=10))
    ax_b.text(c_start + BLOCK_N/2, KB + K_TILE/2,
             "B_buf[16][128]\nloaded to smem!",
             ha="center", va="center", fontsize=9, fontweight="bold",
             color="white", zorder=11)

    ax_b.set_xticks([0, 128, 256, 384])
    ax_b.set_yticks([0, 16, 32, 48, 64])
    ax_b.set_xlabel("N dimension")
    ax_b.set_ylabel("K dimension")

    # C 矩阵 (M x N)
    ax_c = fig.add_axes([0.58, 0.55, 0.40, 0.42])
    ax_c.set_title("C 矩阵 (M=384 rows x N=384 cols)\n"
                   f"current block ({CUR_BLOCK_Y},{CUR_BLOCK_X}) highlighted, "
                   f"accumulating k_tile={CUR_K_TILE}",
                   fontsize=9, fontweight="bold")

    C_h = M  # 384
    C_w = N  # 384
    ax_c.set_xlim(-2, C_w + 2)
    ax_c.set_ylim(C_h + 2, -2)
    ax_c.set_aspect("equal")

    ax_c.add_patch(Rectangle((-0.5, -0.5), C_w, C_h, facecolor="#F5F5F5",
                              edgecolor="#999", linewidth=1))

    for bi in range(0, M + 1, BLOCK_M):
        ax_c.axhline(y=bi - 0.5, color="#999", linewidth=0.5, linestyle="--")
    for bj in range(0, N + 1, BLOCK_N):
        ax_c.axvline(x=bj - 0.5, color="#999", linewidth=0.5, linestyle="--")

    # 高亮当前 block
    ax_c.add_patch(Rectangle((c_start - 0.5, r_start - 0.5), BLOCK_N, BLOCK_M,
                              facecolor="#E1BEE7", edgecolor=C_CBLOCK,
                              linewidth=4, alpha=0.6, zorder=10))
    ax_c.text(c_start + BLOCK_N/2, r_start + BLOCK_M/2,
             f"C[{r_start}:{r_start+128},\n   {c_start}:{c_start+128}]\n"
             f"+= A_slice x B_slice",
             ha="center", va="center", fontsize=10, fontweight="bold",
             color="#4A148C", zorder=11)

    # 4 warp 区域
    warp_labels = [
        (r_start, c_start, "W0\na[0..3]\nb[0..3]"),
        (r_start, c_start+64, "W1\na[0..3]\nb[0..3]"),
        (r_start+64, c_start, "W2\na[0..3]\nb[0..3]"),
        (r_start+64, c_start+64, "W3\na[0..3]\nb[0..3]"),
    ]
    for wi, (wr, wc, wlabel) in enumerate(warp_labels):
        # 每个 warp 64x64
        ax_c.add_patch(Rectangle((wc - 0.5, wr - 0.5), 64, 64,
                                  facecolor="none", edgecolor=C_WARP_E[wi],
                                  linewidth=2.5, alpha=0.7, zorder=12))
        ax_c.text(wc + 32, wr + 32, wlabel, ha="center", va="center",
                 fontsize=7, fontweight="bold", color=C_WARP_E[wi], zorder=13)

        # 16x16 fragment 网格
        for mi in range(4):
            for ni in range(4):
                fr = wr + mi * 16
                fc = wc + ni * 16
                ax_c.add_patch(Rectangle((fc - 0.5, fr - 0.5), 16, 16,
                                          facecolor="none", edgecolor=C_WARP_E[wi],
                                          linewidth=0.3, alpha=0.3, linestyle=":", zorder=11))

    ax_c.set_xticks([0, 128, 256, 384])
    ax_c.set_yticks([0, 128, 256, 384])
    ax_c.set_xlabel("N dimension")
    ax_c.set_ylabel("M dimension")

    # ===== 底部左: Shared Memory 详细数据流 =====
    ax_smem = fig.add_axes([0.02, 0.05, 0.50, 0.45])
    ax_smem.axis("off")
    ax_smem.set_xlim(0, 20)
    ax_smem.set_ylim(0, 12)

    ax_smem.text(10, 11.5, "Shared Memory 数据 → WMMA Fragment 映射",
                ha="center", fontsize=12, fontweight="bold")

    # A_buf
    ax_smem.add_patch(FancyBboxPatch((0.3, 6.5), 6.5, 4.5, boxstyle="round,pad=0.15",
                                      facecolor="#FFCDD2", edgecolor=C_ABLOCK, linewidth=2.5))
    ax_smem.text(3.55, 10.5, "A_buf[128][16] — from A[256:384, 16:32]",
                ha="center", fontsize=9, fontweight="bold", color=C_ABLOCK)
    ax_smem.text(3.55, 10, "128 rows of A (this block) x 16 cols of K",
                ha="center", fontsize=7.5, color="#555")

    # 标注如何划分给 4 个 warp
    for wi in range(4):
        wy = wi // 2  # 0 or 1
        wx = wi % 2  # 0 or 1
        row_start_in_abuf = wy * 64
        y_pos = 9.2 - wy * 3.2
        x_pos = 1.5

        ax_smem.add_patch(Rectangle((x_pos, y_pos - 0.8), 4.5, 1.6,
                                     facecolor=C_WARP[wi], edgecolor=C_WARP_E[wi],
                                     linewidth=2))
        ax_smem.text(x_pos + 2.25, y_pos,
                    f"Warp {wi} reads: rows[{row_start_in_abuf}:{row_start_in_abuf}+64]",
                    ha="center", fontsize=7.5, fontweight="bold")
        # 细分 4 个 a[mi]
        for mi in range(4):
            ax_smem.add_patch(Rectangle((x_pos + mi*1.1, y_pos - 0.6), 1.0, 0.7,
                                         facecolor="none", edgecolor=C_WARP_E[wi],
                                         linewidth=1.5, alpha=0.6))
            ax_smem.text(x_pos + mi*1.1 + 0.5, y_pos - 0.25,
                        f"a[{mi}]", ha="center", fontsize=6, fontweight="bold")

    # B_buf
    ax_smem.add_patch(FancyBboxPatch((8, 6.5), 11, 4.5, boxstyle="round,pad=0.15",
                                      facecolor="#BBDEFB", edgecolor=C_BBLOCK, linewidth=2.5))
    ax_smem.text(13.5, 10.5, "B_buf[16][128] — from B[16:32, 256:384]",
                ha="center", fontsize=9, fontweight="bold", color=C_BBLOCK)
    ax_smem.text(13.5, 10, "16 rows of K x 128 cols of B (this block)",
                ha="center", fontsize=7.5, color="#555")

    # 每列的 warp 划分
    for wi in range(4):
        wx = wi % 2
        col_start_in_bbuf = wx * 64
        y_pos = 9.2 - (wi // 2) * 3.2
        x_pos = 8.5 + wx * 5.5

        ax_smem.add_patch(Rectangle((x_pos, y_pos - 0.8), 4.5, 1.6,
                                     facecolor=C_WARP[wi], edgecolor=C_WARP_E[wi],
                                     linewidth=2))
        ax_smem.text(x_pos + 2.25, y_pos,
                    f"W{wi}: cols[{col_start_in_bbuf}:{col_start_in_bbuf}+64]",
                    ha="center", fontsize=7.5, fontweight="bold")
        for ni in range(4):
            ax_smem.add_patch(Rectangle((x_pos + ni*1.1, y_pos - 0.6), 1.0, 0.7,
                                         facecolor="none", edgecolor=C_WARP_E[wi],
                                         linewidth=1.5, alpha=0.6))
            ax_smem.text(x_pos + ni*1.1 + 0.5, y_pos - 0.25,
                        f"b[{ni}]", ha="center", fontsize=6, fontweight="bold")

    # ===== 底部右: 矩阵乘法公式 =====
    ax_formula = fig.add_axes([0.55, 0.05, 0.43, 0.45])
    ax_formula.axis("off")
    ax_formula.set_xlim(0, 20)
    ax_formula.set_ylim(0, 12)

    ax_formula.text(10, 11.5, "WMMA 矩阵乘法: C[16x16] += A[16x16] x B[16x16]",
                   ha="center", fontsize=12, fontweight="bold")

    # 画一个 WMMA 的具体例子
    # 展示 warp 0 的 fragment (mi=1, ni=2)
    # a[1] = A_buf[0:16] (rows 16:32 of the 128), all 16 K cols
    # b[2] = B_buf[0:16] (all 16 K rows), cols 32:48 of the 128
    # c[1*4+2] = C sub-block at rows 16:32, cols 32:48

    # A fragment
    ax_formula.add_patch(FancyBboxPatch((1, 7), 4, 4, boxstyle="round,pad=0.1",
                                         facecolor="#FFCDD2", edgecolor=C_ABLOCK, linewidth=2))
    ax_formula.text(3, 9.5, "a[mi] = 16x16 FP16", ha="center", fontsize=8, fontweight="bold")
    ax_formula.text(3, 8.8, "From A_buf:\nrows [wy*64+mi*16:\n      wy*64+(mi+1)*16]\ncols [0:16]", ha="center", fontsize=7)
    ax_formula.text(3, 7.3, "e.g. warp 0, mi=1:\nA_buf[16:32][0:16]", ha="center", fontsize=6.5, color="#555")

    # 乘号
    ax_formula.text(5.8, 9, "x", fontsize=20, fontweight="bold", ha="center", va="center")

    # B fragment
    ax_formula.add_patch(FancyBboxPatch((6.5, 7), 4, 4, boxstyle="round,pad=0.1",
                                         facecolor="#BBDEFB", edgecolor=C_BBLOCK, linewidth=2))
    ax_formula.text(8.5, 9.5, "b[ni] = 16x16 FP16", ha="center", fontsize=8, fontweight="bold")
    ax_formula.text(8.5, 8.8, "From B_buf:\nrows [0:16]\ncols [wx*64+ni*16:\n      wx*64+(ni+1)*16]", ha="center", fontsize=7)
    ax_formula.text(8.5, 7.3, "e.g. warp 0, ni=2:\nB_buf[0:16][32:48]", ha="center", fontsize=6.5, color="#555")

    # 等号
    ax_formula.text(11.3, 9, "=", fontsize=20, fontweight="bold", ha="center", va="center")

    # C fragment
    ax_formula.add_patch(FancyBboxPatch((12, 7), 4.5, 4, boxstyle="round,pad=0.1",
                                         facecolor="#E1BEE7", edgecolor=C_CBLOCK, linewidth=2))
    ax_formula.text(14.25, 9.5, "c[mi*4+ni] = 16x16 FP32", ha="center", fontsize=8, fontweight="bold")
    ax_formula.text(14.25, 8.5, "Accumulates:\nC[block_y*128+wy*64\n       +mi*16:\n       +mi*16+16]\n       [block_x*128+wx*64\n       +ni*16:\n       +ni*16+16]", ha="center", fontsize=6.5)
    ax_formula.text(14.25, 7.3, "e.g. C[256+0+16:\n       256+0+32]\n       [256+0+32:\n       256+0+48]", ha="center", fontsize=6, color="#555")

    # 底部: 最终公式
    ax_formula.text(10, 5.5, "单次 k_tile 迭代的完整计算 (4 warps x 16 fragments)", ha="center",
                   fontsize=9, fontweight="bold")
    formula_box = ("C[block_y*128:block_y*128+128][block_x*128:block_x*128+128] +=\n"
        "    A[block_y*128:block_y*128+128][kb:kb+16] x B[kb:kb+16][block_x*128:block_x*128+128]\n"
        "    \\_________________  _________________/   \\___________________  __________________/\n"
        "                      128x16                                16x128\n"
        "                  = A_buf[128][16]                    = B_buf[16][128]\n\n"
        "Then 4 warps consume: each warp decomposes the 128x16 x 16x128 outer product\n"
        "into 4x4=16 WMMA 16x16x16 ops, each accumulating to a 64x64 C sub-region.")
    ax_formula.text(10, 3.2, formula_box, ha="center", fontsize=7.5,
                   fontfamily="monospace",
                   bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#ccc"))


    fig.suptitle(f"GPU GEMM: 一个步骤的数据流动详解\n"
                 f"(block_y={CUR_BLOCK_Y}, block_x={CUR_BLOCK_X}, k_tile={CUR_K_TILE}, kb={KB})",
                 fontsize=14, fontweight="bold", y=1.005)

    path = os.path.join(OUT_DIR, "gemm_v9_one_step_detail.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[done] {path}")


if __name__ == "__main__":
    draw_one_step()
