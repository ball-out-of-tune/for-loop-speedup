#!/usr/bin/env python3
"""
V9 gemm_async_wmma 可视化: cp.async 双缓冲 + WMMA Tensor Core + Warp 级并行
=============================================================================

核心概念:
  1. 128×128 block tile, 4 warps (128 threads), 每个 warp 处理 64×64
  2. 双缓冲 + cp.async: 从 buf[read] 计算的同时, DMA 在后台加载 buf[write]
  3. WMMA: 每个 warp 做 4×4=16 个 16×16×16 的 Tensor Core 运算
  4. K 维度 tile=16 (WMMA 的 K=16)

输出:
  - gemm_v9_overview.png      静态总览: block grid + warp 分解 + WMMA fragment 映射
  - gemm_v9_animation.gif     动画: 展示 cp.async 双缓冲 + compute 重叠
  - gemm_v9_step*.png         关键帧截图
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ============================================================
# 字体配置
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

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# V9 核心参数
# ============================================================
BLOCK_M = 128   # block tile 行数
BLOCK_N = 128   # block tile 列数
WMMA_K  = 16    # WMMA K 维度
WMMA_MN = 16    # WMMA M/N 维度
NUM_WARPS = 4   # 128 threads / 32 = 4 warps
WARPS_PER_ROW = 2  # wy: 2 warps horizontally
WARPS_PER_COL = 2  # wx: 2 warps vertically
WARP_M = 64     # 每个 warp 覆盖的行数 (BLOCK_M / WARPS_PER_ROW)
WARP_N = 64     # 每个 warp 覆盖的列数 (BLOCK_N / WARPS_PER_COL)
FRAGS_PER_WARP_M = WARP_M // WMMA_MN  # 4
FRAGS_PER_WARP_N = WARP_N // WMMA_MN  # 4
FRAGS_PER_WARP = FRAGS_PER_WARP_M * FRAGS_PER_WARP_N  # 16

TOTAL_K = 64  # 简化的 K 维度大小 (实际更大，但用 64 方便展示)
NUM_K_TILES = TOTAL_K // WMMA_K  # 4 个 k-tile

# ============================================================
# 颜色方案
# ============================================================
C_WARP = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]  # 4 warp 颜色
C_WARP_LIGHT = ["#FFCDD2", "#BBDEFB", "#C8E6C9", "#FFE0B2"]
C_DMA = "#FF6F00"        # DMA 活动的橙色
C_COMPUTE = "#7B1FA2"    # 计算的紫色
C_BUFFER0 = "#90CAF9"    # buf[0] 蓝
C_BUFFER1 = "#A5D6A7"    # buf[1] 绿
C_ACTIVE = "#FFD700"     # 当前活动高亮

# ============================================================
# 图 1: 静态总览 — Block Grid + Warp 分解 + WMMA Fragment 映射
# ============================================================
def draw_overview():
    fig = plt.figure(figsize=(22, 16))

    # --- 左上: C 输出矩阵 + Block Grid ---
    ax_c = fig.add_axes([0.02, 0.48, 0.38, 0.50])
    ax_c.set_title("C 输出矩阵 — Block Grid 划分\n"
                   f"gridDim = ((N+127)/128, (M+127)/128), 每 block 128×128",
                   fontsize=12, fontweight="bold")

    # 画 4×4 的 block grid
    grid_rows, grid_cols = 4, 4
    mat_size = grid_rows * BLOCK_M  # 512
    for by in range(grid_rows):
        for bx in range(grid_cols):
            color = plt.cm.tab20((by * grid_cols + bx) % 20)
            rect = Rectangle((bx * BLOCK_N, by * BLOCK_M), BLOCK_N, BLOCK_M,
                             facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.3)
            ax_c.add_patch(rect)
            if grid_rows <= 4:
                ax_c.text(bx * BLOCK_N + 64, by * BLOCK_M + 64,
                          f"B({bx},{by})", ha="center", va="center",
                          fontsize=8, fontweight="bold")
    # 高亮 block(1,1)
    rect = Rectangle((1 * BLOCK_N, 1 * BLOCK_M), BLOCK_N, BLOCK_M,
                     facecolor="none", edgecolor="#FF6F00", linewidth=4, zorder=10)
    ax_c.add_patch(rect)
    ax_c.text(1 * BLOCK_N + 64, 1 * BLOCK_M - 15, "← V9 正在处理的 block",
              fontsize=9, color="#FF6F00", fontweight="bold")

    ax_c.set_xlim(0, grid_cols * BLOCK_N)
    ax_c.set_ylim(grid_rows * BLOCK_M, 0)
    ax_c.set_xlabel("N (列)")
    ax_c.set_ylabel("M (行)")
    ax_c.set_aspect("equal")

    # --- 右上: 一个 128×128 Block 的 Warp 分解 ---
    ax_warp = fig.add_axes([0.45, 0.48, 0.28, 0.50])
    ax_warp.set_title("一个 Block (128×128) 的 Warp 分解\n"
                      f"4 warps × 32 threads, 每 warp 负责 64×64",
                      fontsize=12, fontweight="bold")

    # 4 个 warp 的 64×64 区域
    warp_labels = [
        (0, 0, "Warp 0\nwy=0,wx=0"),
        (0, 64, "Warp 1\nwy=0,wx=1"),
        (64, 0, "Warp 2\nwy=1,wx=0"),
        (64, 64, "Warp 3\nwy=1,wx=1"),
    ]
    for wi, (r0, c0, label) in enumerate(warp_labels):
        ax_warp.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), WARP_N, WARP_M,
                                     facecolor=C_WARP_LIGHT[wi], edgecolor=C_WARP[wi],
                                     linewidth=3, alpha=0.5))
        ax_warp.text(c0 + WARP_N / 2, r0 + WARP_M / 2, label,
                     ha="center", va="center", fontsize=10, fontweight="bold",
                     color=C_WARP[wi])

    # 在每个 warp 区域内画 WMMA 16×16 fragment 网格
    for wi, (r0, c0, _) in enumerate(warp_labels):
        for mi in range(FRAGS_PER_WARP_M):
            for ni in range(FRAGS_PER_WARP_N):
                fr = r0 + mi * WMMA_MN
                fc = c0 + ni * WMMA_MN
                ax_warp.add_patch(Rectangle((fc - 0.5, fr - 0.5), WMMA_MN, WMMA_MN,
                                           facecolor="none", edgecolor=C_WARP[wi],
                                           linewidth=0.5, alpha=0.3, linestyle="--"))

    # 高亮一个 WMMA fragment 并标注
    highlight_warp = 2  # warp 2
    highlight_mi, highlight_ni = 1, 2
    hr0 = warp_labels[highlight_warp][0] + highlight_mi * WMMA_MN
    hc0 = warp_labels[highlight_warp][1] + highlight_ni * WMMA_MN
    ax_warp.add_patch(Rectangle((hc0 - 0.5, hr0 - 0.5), WMMA_MN, WMMA_MN,
                                 facecolor="none", edgecolor="#FFD700",
                                 linewidth=3, zorder=10))
    ax_warp.annotate(f"c[{highlight_mi}*4+{highlight_ni}]\n"
                     f"=wmma::mma_sync(\n"
                     f"  a[{highlight_mi}], b[{highlight_ni}],\n"
                     f"  c[{highlight_mi}*4+{highlight_ni}])\n"
                     f"16×16×16 → Tensor Core",
                     xy=(hc0 + 8, hr0 + 8),
                     xytext=(hc0 + 30, hr0 - 5),
                     arrowprops=dict(arrowstyle="->", color="#FF6F00", lw=2),
                     fontsize=7, color="#333",
                     bbox=dict(boxstyle="round", facecolor="#FFF9C4", alpha=0.9))

    # 粗线分隔 warp 区域
    ax_warp.axhline(y=63.5, color="#333", linewidth=2.5)
    ax_warp.axvline(x=63.5, color="#333", linewidth=2.5)

    ax_warp.set_xlim(-1, BLOCK_N + 1)
    ax_warp.set_ylim(BLOCK_M + 1, -1)
    ax_warp.set_aspect("equal")
    ax_warp.set_xticks([0, 32, 64, 96, 127])
    ax_warp.set_yticks([0, 32, 64, 96, 127])

    # --- 右下: WMMA Fragment 细节 + 数据流 ---
    ax_frag = fig.add_axes([0.78, 0.48, 0.20, 0.50])
    ax_frag.set_title("WMMA Fragment 数据流\n(单个 warp 的视角)",
                      fontsize=12, fontweight="bold")

    # 画出 A fragment (16×16, row_major), B fragment (16×16, row_major),
    # C fragment (accumulator 16×16) 的关系
    ax_frag.axis("off")
    ax_frag.set_xlim(0, 10)
    ax_frag.set_ylim(0, 10)

    # A fragment 示意
    rect_a = FancyBboxPatch((0.5, 6.5), 3, 2.5, boxstyle="round,pad=0.1",
                             facecolor="#FFCDD2", edgecolor="#E53935", linewidth=2)
    ax_frag.add_patch(rect_a)
    ax_frag.text(2, 7.75, "a[mi]\n16×16 FP16\n(row_major)\n← A_buf[read]\n[wy*64+mi*16][:]",
                ha="center", va="center", fontsize=7.5, fontweight="bold")

    # B fragment 示意
    rect_b = FancyBboxPatch((4.5, 6.5), 3, 2.5, boxstyle="round,pad=0.1",
                             facecolor="#BBDEFB", edgecolor="#1E88E5", linewidth=2)
    ax_frag.add_patch(rect_b)
    ax_frag.text(6, 7.75, "b[ni]\n16×16 FP16\n(row_major)\n← B_buf[read]\n[:][wx*64+ni*16]",
                ha="center", va="center", fontsize=7.5, fontweight="bold")

    # 乘号
    ax_frag.text(3.9, 7.75, "×", fontsize=16, fontweight="bold", ha="center", va="center")

    # C fragment 示意
    rect_c = FancyBboxPatch((2, 2.5), 4.5, 3, boxstyle="round,pad=0.1",
                             facecolor="#E1BEE7", edgecolor="#8E24AA", linewidth=2)
    ax_frag.add_patch(rect_c)
    ax_frag.text(4.25, 4.0, "c[mi*4+ni]\n16×16 FP32 accumulator\n+= a[mi] × b[ni]\n"
                 "→ wmma::mma_sync()\n→ Tensor Core 硬件",
                ha="center", va="center", fontsize=8, fontweight="bold", color="#4A148C")

    # 箭头
    ax_frag.annotate("", xy=(2.5, 6.5), xytext=(2.5, 5.5),
                     arrowprops=dict(arrowstyle="->", color="#E53935", lw=2))
    ax_frag.annotate("", xy=(5.5, 6.5), xytext=(5.5, 5.5),
                     arrowprops=dict(arrowstyle="->", color="#1E88E5", lw=2))

    ax_frag.text(8.5, 1.5, "每个 warp:\n4 a[] × 4 b[]\n= 16 c[]\n覆盖 64×64",
                 ha="center", fontsize=8, fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor="#F5F5F5"))

    # --- 左中: Shared Memory 双缓冲布局 ---
    ax_smem = fig.add_axes([0.02, 0.18, 0.42, 0.25])
    ax_smem.set_title("Shared Memory 双缓冲布局 (Double Buffering)\n"
                      f"A_buf[2][128][16] + B_buf[2][16][128], "
                      f"每个 buffer {128*16*2 + 16*128*2} bytes = 6KB FP16",
                      fontsize=11, fontweight="bold")

    ax_smem.axis("off")
    ax_smem.set_xlim(0, 20)
    ax_smem.set_ylim(0, 10)

    # Buffer 0
    rect_a0 = FancyBboxPatch((0.5, 5), 6, 4, boxstyle="round,pad=0.1",
                              facecolor=C_BUFFER0, edgecolor="#1565C0", linewidth=2, alpha=0.3)
    ax_smem.add_patch(rect_a0)
    ax_smem.text(3.5, 8.5, "A_buf[0][128][16]\n(read buffer 或 write buffer)",
                ha="center", fontsize=9, fontweight="bold")
    ax_smem.text(3.5, 6, "128 rows × 16 cols\nK-tile=16, 每次存一个 K-tile 的 A 切片",
                ha="center", fontsize=8, color="#555")

    rect_b0 = FancyBboxPatch((7.5, 5), 7, 4, boxstyle="round,pad=0.1",
                              facecolor=C_BUFFER0, edgecolor="#1565C0", linewidth=2, alpha=0.3)
    ax_smem.add_patch(rect_b0)
    ax_smem.text(11, 8.5, "B_buf[0][16][128]\n(read buffer 或 write buffer)",
                ha="center", fontsize=9, fontweight="bold")
    ax_smem.text(11, 6, "16 rows × 128 cols\nK-tile=16, 每次存一个 K-tile 的 B 切片",
                ha="center", fontsize=8, color="#555")

    # Buffer 1
    rect_a1 = FancyBboxPatch((0.5, 0.3), 6, 4, boxstyle="round,pad=0.1",
                              facecolor=C_BUFFER1, edgecolor="#2E7D32", linewidth=2, alpha=0.3)
    ax_smem.add_patch(rect_a1)
    ax_smem.text(3.5, 3.8, "A_buf[1][128][16]\n(另一块 buffer)",
                ha="center", fontsize=9, fontweight="bold")

    rect_b1 = FancyBboxPatch((7.5, 0.3), 7, 4, boxstyle="round,pad=0.1",
                              facecolor=C_BUFFER1, edgecolor="#2E7D32", linewidth=2, alpha=0.3)
    ax_smem.add_patch(rect_b1)
    ax_smem.text(11, 3.8, "B_buf[1][16][128]\n(另一块 buffer)",
                ha="center", fontsize=9, fontweight="bold")

    # 乒乓箭头
    ax_smem.annotate("乒乓\n切换", xy=(15.5, 7), xytext=(17, 7),
                    arrowprops=dict(arrowstyle="<->", color="#FF6F00", lw=3),
                    fontsize=10, fontweight="bold", color="#FF6F00", ha="center")
    ax_smem.text(17.5, 4, "read_buf\n<-->\nwrite_buf\n= 1 - read_buf",
                ha="center", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#FFF9C4"))

    # --- 右中: cp.async 异步拷贝流程 ---
    ax_async = fig.add_axes([0.48, 0.18, 0.50, 0.25])
    ax_async.set_title("cp.async 异步拷贝 + 计算重叠 (单个 k-tile 迭代)",
                       fontsize=11, fontweight="bold")
    ax_async.axis("off")
    ax_async.set_xlim(0, 20)
    ax_async.set_ylim(0, 10)

    # 流程步骤
    steps_text = [
        (0.5, 8.5, "① 发起 cp.async", "#FF6F00",
         "for chunk in 0..N: cp.async.ca.shared.global\n"
         "[smem_addr], [global_addr], 16  ← 16 bytes = 8×FP16\n"
         "→ DMA 引擎开始在后台搬运数据"),
        (0.5, 6.0, "② commit_group", "#E53935",
         "cp.async.commit_group  ← 把上面所有拷贝打包\n"
         "→ 线程继续执行, DMA 仍在后台运行!"),
        (0.5, 3.5, "③ WMMA 计算 (同时进行!)", "#7B1FA2",
         "从 read_buf 加载 a[mi], b[ni]\n"
         "wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni])\n"
         "→ Tensor Core 计算上一块, DMA 搬运下一块!"),
        (0.5, 1.0, "④ wait_group + sync", "#43A047",
         "cp.async.wait_group 0  ← 等待 DMA 全部完成\n"
         "__syncthreads()  ← 确保整个 block 的 smem 更新\n"
         "read_buf = write_buf  ← 乒乓切换"),
    ]
    for x, y, title, color, desc in steps_text:
        ax_async.text(x, y, title, fontsize=10, fontweight="bold", color=color,
                      bbox=dict(boxstyle="round", facecolor="white",
                                edgecolor=color, alpha=0.8))
        ax_async.text(x + 4.5, y, desc, fontsize=7.5, color="#555", va="top")

    # 向下箭头连接步骤
    for i in range(3):
        ax_async.annotate("", xy=(3, 8.2 - i * 2.5), xytext=(3, 8.8 - i * 2.5),
                         arrowprops=dict(arrowstyle="->", color="#999", lw=1.5))

    # --- 底部: 代码片段 + 关键数字 ---
    ax_code = fig.add_axes([0.02, 0.01, 0.96, 0.14])
    ax_code.axis("off")

    code_lines = (
        "╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
        "║  V9 核心循环结构:                                                                                      ║\n"
        "║  for kb in 0..K step 16:                         // K 维度 tile=16 (WMMA 的 K=16)                    ║\n"
        "║      write_buf = 1 - read_buf                     // 乒乓: 当前读 buf[r], 新数据写 buf[1-r]           ║\n"
        "║      cp.async A→A_buf[write_buf], B→B_buf[write_buf]  // 异步: 线程立刻返回, DMA 后台搬运              ║\n"
        "║      cp.async.commit_group                        // 打包: 上面的拷贝作为一个完成组                    ║\n"
        "║      WMMA compute from read_buf  ←───────────────── 同时: Tensor Core 计算上一块!                    ║\n"
        "║      cp.async.wait_group 0                        // 等待: 确保 DMA 完成                               ║\n"
        "║      __syncthreads(); read_buf = write_buf        // 同步 + 切换                                      ║\n"
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    )
    ax_code.text(0.5, 0.95, "\n".join(code_lines), transform=ax_code.transAxes,
                 fontsize=7.5, verticalalignment="top", ha="center",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#263238", edgecolor="#37474F",
                          alpha=0.95),
                 color="#ECEFF1")

    fig.suptitle("V9 gemm_async_wmma — 全景 Overview\n"
                 "cp.async 双缓冲 + WMMA Tensor Core + 4-Warp 并行分解",
                 fontsize=16, fontweight="bold", y=0.995)

    path = os.path.join(OUT_DIR, "gemm_v9_overview.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[1/3] {path}")


# ============================================================
# 图 2-3: 动画 — 展示 cp.async 双缓冲 + 计算重叠
# ============================================================
def draw_animation():
    """动画: 展示一个 block 在 K 维度上的迭代过程,
    重点显示双缓冲和 cp.async 重叠。"""

    fig = plt.figure(figsize=(20, 12))

    # 布局: 上排 (block tile + warp 活动) + 中排 (时间线) + 下排 (状态面板)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.5, 2, 2], hspace=0.35)

    # 上排: 左右两个子图 — block tile 视图 + shared memory 状态
    gs_top = gs[0].subgridspec(1, 2, wspace=0.3)
    ax_tile = fig.add_subplot(gs_top[0])   # 128×128 tile + warp 高亮
    ax_smem = fig.add_subplot(gs_top[1])   # shared memory 双缓冲状态

    # 中排: 时间线 (DMA + Compute overlap)
    ax_timeline = fig.add_subplot(gs[1])

    # 下排: 状态面板
    ax_status = fig.add_subplot(gs[2])
    ax_status.axis("off")

    # 预计算帧
    # 动画结构: 对每个 k_tile (0..NUM_K_TILES-1), 展示以下阶段:
    #   0: cp.async load A → write_buf
    #   1: cp.async load B → write_buf
    #   2: commit_group + start WMMA from read_buf (overlap!)
    #   3: WMMA computing (展示 warp 活动)
    #   4: wait_group + sync + swap
    PHASES_PER_KTILE = 5
    # k_tile=0 特殊: 只有手动预取 (no cp.async for first tile)
    # 第一个 tile: 手动 short2* 加载 → sync → 直接计算

    frames = []
    # Frame 0: 初始预取 buf[0] (手动 short2* 加载)
    frames.append({"type": "prefetch", "desc": "预取: 手动 short2* 加载 buf[0]"})

    # k_tile 0 的计算 (buf[0] 已在共享内存)
    frames.append({"type": "compute_only", "k_tile": 0, "desc": "k_tile=0 计算: WMMA from buf[0]"})

    # k_tile 1..NUM_K_TILES-1: cp.async 加载到 write_buf + 计算 read_buf
    for kt in range(1, NUM_K_TILES):
        frames.append({"type": "async_load", "k_tile": kt, "phase": "load_A",
                       "desc": f"k_tile={kt}: cp.async 加载 A → write_buf"})
        frames.append({"type": "async_load", "k_tile": kt, "phase": "load_B",
                       "desc": f"k_tile={kt}: cp.async 加载 B → write_buf"})
        frames.append({"type": "overlap", "k_tile": kt,
                       "desc": f"k_tile={kt}: commit_group → WMMA (read_buf) || DMA (write_buf)"})
        frames.append({"type": "wmma_detail", "k_tile": kt,
                       "desc": f"k_tile={kt}: 4 warps × 16 WMMA fragments (read_buf)"})
        frames.append({"type": "sync_swap", "k_tile": kt,
                       "desc": f"k_tile={kt}: wait_group 0 + __syncthreads() + swap"})

    # 最后: 最后一个 tile 单独计算 + writeback
    frames.append({"type": "final_compute", "k_tile": NUM_K_TILES - 1,
                   "desc": "最后一个 k_tile: WMMA from read_buf"})
    frames.append({"type": "writeback", "desc": "Writeback: wmma::store_matrix_sync → C (global mem)"})

    NUM_FRAMES = len(frames)

    # 选择缩小规模的 K 以便展示
    K_SHOW = 4 * WMMA_K  # 64
    N_WMMA_X = 4
    N_WMMA_Y = 4

    def draw_tile_view(ax, frame_info, frame_idx):
        """画 128×128 block tile, 显示 warp 区域 + 当前活动。"""
        ax.clear()
        ax.set_xlim(-1, BLOCK_N + 2)
        ax.set_ylim(BLOCK_M + 2, -1)
        ax.set_aspect("equal")
        ax.set_xticks([0, 32, 64, 96, 128])
        ax.set_yticks([0, 32, 64, 96, 128])

        ftype = frame_info["type"]

        # 4 个 warp 区域的背景
        warp_positions = [(0, 0), (0, 64), (64, 0), (64, 64)]
        for wi, (r0, c0) in enumerate(warp_positions):
            ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), WARP_N, WARP_M,
                                    facecolor=C_WARP_LIGHT[wi], edgecolor=C_WARP[wi],
                                    linewidth=2, alpha=0.35))

        # 画 16×16 WMMA fragment 细分线
        for wi, (r0, c0) in enumerate(warp_positions):
            for mi in range(FRAGS_PER_WARP_M):
                for ni in range(FRAGS_PER_WARP_N):
                    fr = r0 + mi * WMMA_MN
                    fc = c0 + ni * WMMA_MN
                    ax.add_patch(Rectangle((fc - 0.5, fr - 0.5), WMMA_MN, WMMA_MN,
                                           facecolor="none", edgecolor=C_WARP[wi],
                                           linewidth=0.3, alpha=0.25, linestyle=":"))

        # 粗线分隔 4 个 warp
        ax.axhline(y=63.5, color="#333", linewidth=2.5)
        ax.axvline(x=63.5, color="#333", linewidth=2.5)

        # 根据帧类型高亮活动
        if ftype == "prefetch":
            ax.set_title("预取阶段: 手动 vectorized load (short2*)\n"
                        "128 threads 协作加载 A_buf[0] 和 B_buf[0]",
                        fontsize=11, fontweight="bold")
            # 高亮整个 tile (加载中)
            for wi, (r0, c0) in enumerate(warp_positions):
                ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), WARP_N, WARP_M,
                                        facecolor="none", edgecolor=C_DMA,
                                        linewidth=4, alpha=0.7, linestyle="--"))

        elif ftype == "compute_only":
            kt = frame_info.get("k_tile", 0)
            ax.set_title(f"k_tile={kt}: WMMA 计算 (buf[0])\n"
                        f"4 warps 各自加载 a[0..3], b[0..3] → mma_sync × 16 fragments",
                        fontsize=11, fontweight="bold", color=C_COMPUTE)
            # 高亮所有 warp 区域 (所有 warp 都在计算)
            for wi, (r0, c0) in enumerate(warp_positions):
                ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), WARP_N, WARP_M,
                                        facecolor=C_WARP_LIGHT[wi], edgecolor=C_COMPUTE,
                                        linewidth=3, alpha=0.6))

        elif ftype == "async_load":
            kt = frame_info["ktile"] if "ktile" in frame_info else frame_info.get("k_tile", 0)
            ab = frame_info.get("phase", "load_A")
            mat_name = "A" if ab == "load_A" else "B"
            ax.set_title(f"k_tile={kt}: cp.async 加载 {mat_name} → write_buf[{1 - (kt % 2)}]\n"
                        f"DMA 引擎在后台搬运, 线程不阻塞!",
                        fontsize=11, fontweight="bold", color=C_DMA)
            # 脉冲式高亮 (用进度条方式)
            pulse = (frame_idx % 4) / 4.0
            for wi, (r0, c0) in enumerate(warp_positions):
                alpha = 0.3 + 0.3 * pulse
                ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), WARP_N, WARP_M,
                                        facecolor="none", edgecolor=C_DMA,
                                        linewidth=4, alpha=alpha, linestyle="--"))

        elif ftype == "overlap":
            kt = frame_info.get("k_tile", 1)
            ax.set_title(f"k_tile={kt}: [重叠阶段]\n"
                        f"Tensor Core 计算 read_buf[{kt % 2}] 的同时,\n"
                        f"DMA 在后台搬运 write_buf[{1 - (kt % 2)}]",
                        fontsize=11, fontweight="bold", color="#FF6F00")
            # 半边显示计算 (绿色), 半边显示加载 (橙色)
            for wi, (r0, c0) in enumerate(warp_positions):
                if wi % 2 == 0:
                    ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), WARP_N, WARP_M,
                                            facecolor="#C8E6C9", edgecolor=C_COMPUTE,
                                            linewidth=3, alpha=0.5))
                else:
                    ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), WARP_N, WARP_M,
                                            facecolor="none", edgecolor=C_DMA,
                                            linewidth=3, alpha=0.6, linestyle="--"))

        elif ftype == "wmma_detail":
            kt = frame_info.get("k_tile", 1)
            ax.set_title(f"k_tile={kt}: WMMA Fragment 加载 & 计算\n"
                        f"for mi in 0..3, ni in 0..3:\n"
                        f"  load a[mi], b[ni]; mma_sync(c[mi*4+ni])",
                        fontsize=11, fontweight="bold", color=C_COMPUTE)

            # 逐个高亮 WMMA fragments (模拟 mi, ni 循环)
            progress = (frame_idx % 16)  # 16 fragments
            mi = progress // 4
            ni = progress % 4
            for wi, (r0, c0) in enumerate(warp_positions):
                # 高亮当前正在计算的 fragment
                fr = r0 + mi * WMMA_MN
                fc = c0 + ni * WMMA_MN
                ax.add_patch(Rectangle((fc - 0.5, fr - 0.5), WMMA_MN, WMMA_MN,
                                        facecolor=C_ACTIVE, edgecolor="#333",
                                        linewidth=3, alpha=0.8, zorder=10))
                # 已经完成的 fragments
                for done_i in range(mi):
                    for done_j in range(4):
                        dr = r0 + done_i * WMMA_MN
                        dc = c0 + done_j * WMMA_MN
                        ax.add_patch(Rectangle((dc - 0.5, dr - 0.5), WMMA_MN, WMMA_MN,
                                                facecolor="#C8E6C9", edgecolor="#666",
                                                linewidth=1, alpha=0.5, zorder=5))
                for done_j in range(ni):
                    dr = r0 + mi * WMMA_MN
                    dc = c0 + done_j * WMMA_MN
                    ax.add_patch(Rectangle((dc - 0.5, dr - 0.5), WMMA_MN, WMMA_MN,
                                            facecolor="#C8E6C9", edgecolor="#666",
                                            linewidth=1, alpha=0.5, zorder=5))

        elif ftype == "sync_swap":
            kt = frame_info.get("k_tile", 1)
            ax.set_title(f"k_tile={kt}: 同步 + 乒乓切换\n"
                        f"cp.async.wait_group 0 → __syncthreads() → read_buf = {1 - (kt % 2)}",
                        fontsize=11, fontweight="bold", color="#43A047")
            # 绿色边框表示同步完成
            for wi, (r0, c0) in enumerate(warp_positions):
                ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), WARP_N, WARP_M,
                                        facecolor="none", edgecolor="#43A047",
                                        linewidth=3, alpha=0.8))

        elif ftype == "final_compute":
            ax.set_title(f"最后一个 k_tile: WMMA 收尾计算\n"
                        f"不需要再发起 cp.async (没有下一个 tile)",
                        fontsize=11, fontweight="bold", color=C_COMPUTE)
            for wi, (r0, c0) in enumerate(warp_positions):
                ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), WARP_N, WARP_M,
                                        facecolor=C_WARP_LIGHT[wi], edgecolor=C_COMPUTE,
                                        linewidth=3, alpha=0.6))

        elif ftype == "writeback":
            ax.set_title("Writeback 阶段: wmma::store_matrix_sync\n"
                        "每个 warp 把 16 个 c[] fragment 写回 C (global memory)",
                        fontsize=11, fontweight="bold", color="#0D47A1")
            for wi, (r0, c0) in enumerate(warp_positions):
                ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), WARP_N, WARP_M,
                                        facecolor="#BBDEFB", edgecolor="#0D47A1",
                                        linewidth=3, alpha=0.6))

        # warp 标签
        for wi, (r0, c0) in enumerate(warp_positions):
            ax.text(c0 + WARP_N / 2, r0 + WARP_M / 2,
                    f"Warp\n{wi}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=C_WARP[wi], alpha=0.6)

    def draw_smem_view(ax, frame_info):
        """画 shared memory 双缓冲状态。"""
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        ftype = frame_info["type"]
        kt = frame_info.get("k_tile", 0)

        # 确定 read_buf 和 write_buf
        if ftype == "prefetch":
            read_buf = 0
            write_buf_active = False
        elif ftype == "compute_only":
            read_buf = 0
            write_buf_active = False
        elif ftype in ("async_load", "overlap", "wmma_detail"):
            read_buf = kt % 2
            write_buf_active = True
            write_buf = 1 - read_buf
        elif ftype == "sync_swap":
            read_buf = 1 - (kt % 2)  # 已经切换了
            write_buf_active = False
        elif ftype in ("final_compute", "writeback"):
            read_buf = NUM_K_TILES % 2  # 最后一个 tile
            write_buf_active = False
        else:
            read_buf = 0
            write_buf_active = False

        # 画 A_buf[0] 和 A_buf[1]
        for buf_id in [0, 1]:
            y_base = 7.5 - buf_id * 4
            is_read = (buf_id == read_buf)
            is_write = write_buf_active and (buf_id == (1 - read_buf))

            # 背景框
            if is_read and is_write:
                facecolor = "#FFE082"
                edgecolor = "#FF6F00"
                label = f"A_buf[{buf_id}]\nREAD + WRITE"
            elif is_read:
                facecolor = C_BUFFER0 if buf_id == 0 else C_BUFFER1
                edgecolor = "#1565C0" if buf_id == 0 else "#2E7D32"
                label = f"A_buf[{buf_id}]\n← READ (计算用)"
            elif is_write:
                facecolor = "#FFE0B2"
                edgecolor = C_DMA
                label = f"A_buf[{buf_id}]\n← WRITE (DMA 加载中)"
            else:
                facecolor = "#EEEEEE"
                edgecolor = "#999"
                label = f"A_buf[{buf_id}]\n(空闲)"

            ax.add_patch(FancyBboxPatch((0.3, y_base), 4, 2.5,
                                        boxstyle="round,pad=0.1",
                                        facecolor=facecolor, edgecolor=edgecolor,
                                        linewidth=2.5 if (is_read or is_write) else 1))
            ax.text(2.3, y_base + 1.25, label, ha="center", va="center",
                    fontsize=8, fontweight="bold")

            # B_buf
            is_read_b = (buf_id == read_buf)
            is_write_b = write_buf_active and (buf_id == (1 - read_buf))
            if is_read_b and is_write_b:
                fb, eb = "#FFE082", "#FF6F00"
                blabel = f"B_buf[{buf_id}]\nREAD + WRITE"
            elif is_read_b:
                fb = C_BUFFER0 if buf_id == 0 else C_BUFFER1
                eb = "#1565C0" if buf_id == 0 else "#2E7D32"
                blabel = f"B_buf[{buf_id}]\n← READ (计算用)"
            elif is_write_b:
                fb, eb = "#FFE0B2", C_DMA
                blabel = f"B_buf[{buf_id}]\n← WRITE (DMA 加载中)"
            else:
                fb, eb = "#EEEEEE", "#999"
                blabel = f"B_buf[{buf_id}]\n(空闲)"

            ax.add_patch(FancyBboxPatch((5.0, y_base), 4.5, 2.5,
                                        boxstyle="round,pad=0.1",
                                        facecolor=fb, edgecolor=eb,
                                        linewidth=2.5 if (is_read_b or is_write_b) else 1))
            ax.text(7.25, y_base + 1.25, blabel, ha="center", va="center",
                    fontsize=8, fontweight="bold")

        ax.set_title("Shared Memory 双缓冲状态", fontsize=11, fontweight="bold")

    def draw_timeline(ax, frame_info, frame_idx):
        """画时间线: 展示 DMA 和 Compute 在时间轴上的重叠。"""
        ax.clear()
        ax.set_xlim(-0.5, NUM_K_TILES + 1.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Compute\n(Tensor Core)", "DMA\n(cp.async)", "Sync"])
        ax.set_xticks(range(NUM_K_TILES))
        ax.set_xticklabels([f"k_tile\n{i}" for i in range(NUM_K_TILES)])
        ax.set_title("计算与数据传输时间线 (重叠视图)", fontsize=11, fontweight="bold")

        ftype = frame_info["type"]

        for kt in range(NUM_K_TILES):
            # Compute bar (row 0)
            if kt == 0:
                # k_tile 0: 用 buf[0], 手动加载后计算
                compute_end = 1.0
                compute_color = C_COMPUTE
            else:
                compute_end = 1.0
                compute_color = C_COMPUTE

            ax.add_patch(Rectangle((kt - 0.4, -0.4), 0.8, 0.8,
                                    facecolor=compute_color, edgecolor="#333",
                                    linewidth=1, alpha=0.7))
            ax.text(kt, 0, f"C", ha="center", va="center", fontsize=14,
                    fontweight="bold", color="white")

            # DMA bar (row 1)
            if kt >= 1:
                ax.add_patch(Rectangle((kt - 0.4, 0.6), 0.8, 0.8,
                                        facecolor=C_DMA, edgecolor="#333",
                                        linewidth=1, alpha=0.7))
                ax.text(kt, 1, f"D", ha="center", va="center", fontsize=14,
                        fontweight="bold", color="white")

            # Sync bar (row 2)
            if kt >= 1:
                ax.add_patch(Rectangle((kt - 0.4, 1.6), 0.8, 0.8,
                                        facecolor="#43A047", edgecolor="#333",
                                        linewidth=1, alpha=0.5))
                ax.text(kt, 2, f"S", ha="center", va="center", fontsize=14,
                        fontweight="bold", color="white")

        # 高亮当前帧
        if "k_tile" in frame_info:
            kt_cur = frame_info["k_tile"]
            ftype_cur = frame_info["type"]

            if ftype_cur in ("async_load", "overlap"):
                # 高亮 DMA 行
                if kt_cur >= 1:
                    ax.add_patch(Rectangle((kt_cur - 0.5, 0.5), 1.0, 1.0,
                                            facecolor="none", edgecolor=C_DMA,
                                            linewidth=4, zorder=10))
            if ftype_cur in ("compute_only", "overlap", "wmma_detail", "final_compute"):
                if kt_cur < NUM_K_TILES:
                    ax.add_patch(Rectangle((kt_cur - 0.5, -0.5), 1.0, 1.0,
                                            facecolor="none", edgecolor=C_COMPUTE,
                                            linewidth=4, zorder=10))
            if ftype_cur == "sync_swap":
                if kt_cur >= 1:
                    ax.add_patch(Rectangle((kt_cur - 0.5, 1.5), 1.0, 1.0,
                                            facecolor="none", edgecolor="#43A047",
                                            linewidth=4, zorder=10))

        # 图例
        ax.legend([Rectangle((0, 0), 1, 1, facecolor=C_COMPUTE, alpha=0.7, label="Compute (Tensor Core)"),
                   Rectangle((0, 0), 1, 1, facecolor=C_DMA, alpha=0.7, label="DMA (cp.async)"),
                   Rectangle((0, 0), 1, 1, facecolor="#43A047", alpha=0.5, label="Sync + Swap")],
                  ["Compute", "DMA", "Sync"],
                  loc="upper right", fontsize=7)

    def draw_status(ax, frame_info, frame_idx):
        """画状态面板。"""
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        ftype = frame_info["type"]
        desc = frame_info.get("desc", "")
        kt = frame_info.get("k_tile", 0)

        # 标题
        ax.text(5, 9.5, f"Frame {frame_idx + 1}/{NUM_FRAMES}: {desc}",
                ha="center", fontsize=13, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#FFF9C4", edgecolor="#FF6F00", alpha=0.9))

        # 状态信息
        if ftype == "prefetch":
            read_buf, write_buf = 0, "N/A"
            dma_active = False
            compute_active = False
        elif ftype == "compute_only":
            read_buf, write_buf = 0, "N/A"
            dma_active = False
            compute_active = True
        elif ftype in ("async_load",):
            read_buf = kt % 2
            write_buf = 1 - read_buf
            dma_active = True
            compute_active = False
        elif ftype == "overlap":
            read_buf = kt % 2
            write_buf = 1 - read_buf
            dma_active = True
            compute_active = True
        elif ftype == "wmma_detail":
            read_buf = kt % 2
            write_buf = 1 - read_buf
            dma_active = True
            compute_active = True
        elif ftype == "sync_swap":
            read_buf = 1 - (kt % 2)
            write_buf = "N/A"
            dma_active = False
            compute_active = False
        elif ftype in ("final_compute", "writeback"):
            read_buf = NUM_K_TILES % 2
            write_buf = "N/A"
            dma_active = False
            compute_active = (ftype == "final_compute")
        else:
            read_buf, write_buf = 0, "N/A"
            dma_active = False
            compute_active = False

        info_lines = [
            f"read_buf  = {read_buf}",
            f"write_buf = {write_buf}",
            f"DMA 搬运中: {'YES !!' if dma_active else 'no'}",
            f"Tensor Core 计算中: {'YES !!' if compute_active else 'no'}",
            f"重叠: {'YES!!! >>' if dma_active and compute_active else 'no'}",
            "",
            "Key: DMA 加载 write_buf 的同时,",
            "Tensor Core 从 read_buf 计算上一块!",
            "这就是 cp.async 的核心价值。",
        ]

        for i, line in enumerate(info_lines):
            color = "#333"
            if "YES" in line:
                color = "#E53935"
            elif "重叠" in line and ">>" in line:
                color = "#FF6F00"
            ax.text(0.5, 7 - i * 0.7, line, fontsize=10, color=color,
                    fontweight="bold" if "YES" in line or "重叠" in line else "normal")

        # 图例
        ax.text(5.5, 7, "颜色图例:", fontsize=9, fontweight="bold", color="#555")
        legend_items = [
            ("Warp 0-3", C_WARP[0], C_WARP[1], C_WARP[2], C_WARP[3]),
            ("DMA 活动", C_DMA, None, None, None),
            ("计算活动", C_COMPUTE, None, None, None),
        ]
        for i, (label, c1, c2, c3, c4) in enumerate(legend_items):
            y = 6.2 - i * 0.6
            ax.text(5.5, y, label, fontsize=8, color="#555")
            for j, c in enumerate([c1, c2, c3, c4]):
                if c is not None:
                    ax.add_patch(Rectangle((6.5 + j * 0.8, y - 0.2), 0.6, 0.4,
                                            facecolor=c, edgecolor="#333", linewidth=1))

    def update(frame_idx):
        frame_info = frames[frame_idx]
        draw_tile_view(ax_tile, frame_info, frame_idx)
        draw_smem_view(ax_smem, frame_info)
        draw_timeline(ax_timeline, frame_info, frame_idx)
        draw_status(ax_status, frame_info, frame_idx)

        fig.suptitle("V9 gemm_async_wmma — 双缓冲 + cp.async + WMMA 计算流程\n"
                    f"K={TOTAL_K}, K-tile=16, 4 warps, 16 WMMA fragments/warp",
                    fontsize=14, fontweight="bold", y=0.995)
        return [ax_tile, ax_smem, ax_timeline, ax_status]

    anim = FuncAnimation(fig, update, frames=NUM_FRAMES, interval=900, blit=False)

    path = os.path.join(OUT_DIR, "gemm_v9_animation.gif")
    writer = matplotlib.animation.PillowWriter(fps=1.2)
    anim.save(path, writer=writer, dpi=80)
    plt.close(fig)
    print(f"[2/3] {path}  ({NUM_FRAMES} frames)")


# ============================================================
# 图 3: 关键帧静态截图
# ============================================================
def save_keyframes():
    """保存几个关键帧作为静态截图。"""
    # 使用相同的 fig 布局保存关键帧
    pass  # 关键帧由动画自动覆盖


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  V9 gemm_async_wmma Visualization")
    print(f"  Block: {BLOCK_M}×{BLOCK_N}, {NUM_WARPS} warps")
    print(f"  WMMA: 16×16×16, {FRAGS_PER_WARP} fragments/warp")
    print(f"  K tiles: {NUM_K_TILES} (K={TOTAL_K}, WMMA_K={WMMA_K})")
    print("=" * 60)

    draw_overview()
    draw_animation()

    print("\nDone. Generated:")
    print("  gemm_v9_overview.png   — 静态总览: block grid + warp 分解 + WMMA + 双缓冲")
    print("  gemm_v9_animation.gif  — 动画: cp.async 双缓冲 + WMMA 计算流程")
