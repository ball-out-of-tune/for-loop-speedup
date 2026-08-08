#!/usr/bin/env python3
"""
V9 gemm_async_wmma 逐步串行演示
===============================
按照代码执行顺序，一步一步展示:
  1. 预取 buf[0] (short2* 向量化加载)
  2. K 维度迭代 (kb = 16, 32, 48)
     a. cp.async 加载下一个 tile 到 write_buf
     b. WMMA 加载 a[0..3], b[0..3]
     c. WMMA mma_sync × 16 fragments
     d. wait_group + sync + swap
  3. 最后一个 tile 计算
  4. Writeback

每帧 = 一个具体操作, 配合代码行高亮
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
# 字体
# ============================================================
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

# ============================================================
# 参数 (简化为演示用)
# ============================================================
M_SHOW = 128   # block tile 行
N_SHOW = 128   # block tile 列
K_TOTAL = 64   # K 维度
K_TILE = 16    # WMMA K=16
WMMA_MN = 16   # WMMA M=N=16

# 4 warps 的布局
WARP_POS = [(0,0), (0,64), (64,0), (64,64)]  # (row_start, col_start)
WARP_SIZE = 64

# 颜色
C_WARPS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]
C_WARPS_L = ["#FFCDD2", "#BBDEFB", "#C8E6C9", "#FFE0B2"]
C_LOAD = "#FF6F00"    # 加载中
C_COMP = "#7B1FA2"    # 计算中
C_DONE = "#4CAF50"    # 已完成
C_SYNC = "#1565C0"    # 同步
C_BUF0 = "#90CAF9"
C_BUF1 = "#A5D6A7"
C_HIGHLIGHT = "#FFD700"

# ============================================================
# 构建帧序列
# ============================================================
frames = []

def add_frames(count, **base):
    """添加 count 个相同类型的帧, 只是 step 递增"""
    for i in range(count):
        f = base.copy()
        f["step_in_phase"] = i
        frames.append(f)

# ---- Phase 0: 预取 buf[0] ----
frames.append({"phase": "prefetch_intro", "desc": "开始: 预取第一个 K-tile 到 buf[0]"})
add_frames(3, phase="prefetch_A", desc="手动加载 A[0:128, 0:16] -> A_buf[0], 用 short2* 向量化")
add_frames(3, phase="prefetch_B", desc="手动加载 B[0:16, 0:128] -> B_buf[0], 用 short2* 向量化")
frames.append({"phase": "prefetch_sync", "desc": "__syncthreads() — 确保 buf[0] 完全加载"})

# ---- Phase 1: k_tile 0 的计算 (buf[0]) ----
frames.append({"phase": "compute_intro", "k_tile": 0, "read_buf": 0,
               "desc": "k_tile=0: 开始从 buf[0] 做 WMMA 计算"})

# 4 warps, 每个 warp 4x4=16 fragments
# 逐步展示: 先加载 a[mi] 和 b[ni], 再 mma_sync
for mi in range(4):
    for ni in range(4):
        frag_idx = mi * 4 + ni
        # 加载 a[mi]
        frames.append({
            "phase": "wmma_load_a", "k_tile": 0, "read_buf": 0,
            "mi": mi, "ni": ni, "frag_idx": frag_idx,
            "desc": f"Warp加载: wmma::load_matrix_sync(a[{mi}], &A_buf[0][wy*64+{mi}*16][0], 16)"
        })
        # 加载 b[ni]
        frames.append({
            "phase": "wmma_load_b", "k_tile": 0, "read_buf": 0,
            "mi": mi, "ni": ni, "frag_idx": frag_idx,
            "desc": f"Warp加载: wmma::load_matrix_sync(b[{ni}], &B_buf[0][0][wx*64+{ni}*16], 128)"
        })
        # mma_sync
        frames.append({
            "phase": "wmma_mma", "k_tile": 0, "read_buf": 0,
            "mi": mi, "ni": ni, "frag_idx": frag_idx,
            "desc": f"计算: wmma::mma_sync(c[{frag_idx}], a[{mi}], b[{ni}], c[{frag_idx}]) — Tensor Core!"
        })

# ---- Phase 2-4: k_tile 1,2,3 的 cp.async + 计算 ----
for kt in range(1, K_TOTAL // K_TILE):
    read_buf = kt % 2
    write_buf = 1 - read_buf

    # cp.async 加载 A
    frames.append({
        "phase": "async_A", "k_tile": kt, "read_buf": read_buf, "write_buf": write_buf,
        "desc": f"k_tile={kt}: cp.async 异步加载 A[0:128, {kt*16}:{kt*16+16}] -> A_buf[{write_buf}]"
    })
    # cp.async 加载 B
    frames.append({
        "phase": "async_B", "k_tile": kt, "read_buf": read_buf, "write_buf": write_buf,
        "desc": f"k_tile={kt}: cp.async 异步加载 B[{kt*16}:{kt*16+16}, 0:128] -> B_buf[{write_buf}]"
    })
    # commit_group
    frames.append({
        "phase": "commit", "k_tile": kt, "read_buf": read_buf, "write_buf": write_buf,
        "desc": f"k_tile={kt}: cp.async.commit_group — DMA 在后台搬运, 线程继续执行!"
    })

    # WMMA 计算 (从 read_buf, 此时 DMA 在后台搬运 write_buf!)
    for mi in range(4):
        for ni in range(4):
            frag_idx = mi * 4 + ni
            frames.append({
                "phase": "overlap_mma", "k_tile": kt, "read_buf": read_buf,
                "write_buf": write_buf, "mi": mi, "ni": ni, "frag_idx": frag_idx,
                "desc": f"重叠! k_tile={kt}: mma_sync(c[{frag_idx}]) from buf[{read_buf}] 同时 DMA->buf[{write_buf}]"
            })

    # wait + sync + swap
    frames.append({
        "phase": "wait_sync_swap", "k_tile": kt, "read_buf": read_buf, "write_buf": write_buf,
        "desc": f"k_tile={kt}: cp.async.wait_group 0 -> __syncthreads() -> read_buf={write_buf}"
    })

# ---- Phase 5: 最后一个 tile 收尾 ----
last_kt = K_TOTAL // K_TILE - 1
last_buf = last_kt % 2
frames.append({
    "phase": "last_compute_intro", "k_tile": last_kt, "read_buf": last_buf,
    "desc": f"最后一个 k_tile={last_kt}: 从 buf[{last_buf}] 做最后的 WMMA 计算"
})
for mi in range(4):
    for ni in range(4):
        frag_idx = mi * 4 + ni
        frames.append({
            "phase": "last_mma", "k_tile": last_kt, "read_buf": last_buf,
            "mi": mi, "ni": ni, "frag_idx": frag_idx,
            "desc": f"最后: mma_sync(c[{frag_idx}]) — 累加 buf[{last_buf}] 的贡献"
        })

# ---- Phase 6: Writeback ----
for wi in range(4):
    r0, c0 = WARP_POS[wi]
    frames.append({
        "phase": "writeback", "warp": wi, "r0": r0, "c0": c0,
        "desc": f"Writeback Warp {wi}: store_matrix_sync → C[block_y*128+{r0}:{r0}+64][block_x*128+{c0}:{c0}+64]"
    })

NUM_FRAMES = len(frames)
print(f"Total frames: {NUM_FRAMES}")

# ============================================================
# 动画
# ============================================================
def create_step_animation():
    fig = plt.figure(figsize=(20, 14))

    # 布局: 左(C tile 128x128) + 中(数据流 A/B) + 右(代码/状态)
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.5],
                          hspace=0.4, wspace=0.35,
                          left=0.03, right=0.97, top=0.93, bottom=0.05)

    ax_tile = fig.add_subplot(gs[0, 0])    # 128x128 C tile + warp 分解
    ax_data = fig.add_subplot(gs[0, 1:])   # A/B 矩阵 + 数据流
    ax_buf  = fig.add_subplot(gs[1, :2])   # Shared memory 缓冲状态
    ax_info = fig.add_subplot(gs[1, 2])    # 状态信息

    def draw_c_tile(ax, frame):
        """画 128x128 输出 tile"""
        ax.clear()
        ax.set_xlim(-2, N_SHOW + 2)
        ax.set_ylim(M_SHOW + 2, -2)
        ax.set_aspect("equal")
        ax.set_xticks([0, 32, 64, 96, 128])
        ax.set_yticks([0, 32, 64, 96, 128])

        phase = frame["phase"]
        kt = frame.get("k_tile", 0)
        mi = frame.get("mi", -1)
        ni = frame.get("ni", -1)
        frag_idx = frame.get("frag_idx", -1)

        # 4 个 warp 区域背景
        for wi, (r0, c0) in enumerate(WARP_POS):
            alpha = 0.25
            edge_w = 2
            fc = C_WARPS_L[wi]
            ec = C_WARPS[wi]
            label = f"W{wi}"

            # 根据状态调整颜色
            if phase in ("wmma_load_a", "wmma_load_b", "wmma_mma",
                         "overlap_mma", "last_mma"):
                # 当前正在计算的 fragment 高亮
                fc = C_WARPS_L[wi]
                ec = C_WARPS[wi]
                alpha = 0.4
            elif phase == "writeback" and frame.get("warp") == wi:
                fc = "#FFF9C4"
                ec = C_HIGHLIGHT
                alpha = 0.7
                edge_w = 4
            elif phase == "writeback" and frame.get("warp") < wi:
                fc = "#C8E6C9"
                ec = C_DONE
                alpha = 0.6

            ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), 64, 64,
                                    facecolor=fc, edgecolor=ec,
                                    linewidth=edge_w, alpha=alpha))
            ax.text(c0 + 32, r0 + 32, label, ha="center", va="center",
                   fontsize=11, fontweight="bold", color=ec, alpha=0.5)

        # WMMA 16x16 fragment 网格线
        for wi, (r0, c0) in enumerate(WARP_POS):
            for mi2 in range(4):
                for ni2 in range(4):
                    fr = r0 + mi2 * 16
                    fc = c0 + ni2 * 16
                    ax.add_patch(Rectangle((fc - 0.5, fr - 0.5), 16, 16,
                                           facecolor="none", edgecolor=C_WARPS[wi],
                                           linewidth=0.3, alpha=0.2, linestyle=":"))

        # 高亮当前 fragment
        if mi >= 0 and ni >= 0:
            for wi, (r0, c0) in enumerate(WARP_POS):
                fr = r0 + mi * 16
                fc = c0 + ni * 16
                # 根据 k_tile 进度着色
                kt_progress = kt / (K_TOTAL // K_TILE)
                color = plt.cm.YlOrRd(0.3 + 0.6 * kt_progress)
                ax.add_patch(Rectangle((fc - 0.5, fr - 0.5), 16, 16,
                                        facecolor=color, edgecolor="#333",
                                        linewidth=3, alpha=0.85, zorder=10))
                ax.text(fc + 8, fr + 8, f"c\n[{frag_idx}]", ha="center", va="center",
                       fontsize=7, fontweight="bold", color="white", zorder=11)

        # 粗分隔线
        ax.axhline(y=63.5, color="#333", linewidth=2.5)
        ax.axvline(x=63.5, color="#333", linewidth=2.5)

        # title
        if phase in ("prefetch_intro", "prefetch_A", "prefetch_B", "prefetch_sync"):
            title = "预取阶段: 加载 buf[0]"
        elif phase == "compute_intro":
            title = f"k_tile=0: 准备 WMMA 计算"
        elif phase in ("wmma_load_a", "wmma_load_b", "wmma_mma"):
            title = f"k_tile={kt}: WMMA fragment [{frag_idx}/16] (mi={mi}, ni={ni})"
        elif phase in ("async_A", "async_B", "commit"):
            title = f"k_tile={kt}: cp.async 异步加载 write_buf[{frame.get('write_buf','?')}]"
        elif phase == "overlap_mma":
            title = f"k_tile={kt}: 重叠! 计算 buf[{frame['read_buf']}] + DMA buf[{frame['write_buf']}]"
        elif phase == "wait_sync_swap":
            title = f"k_tile={kt}: 同步 + 切换"
        elif phase == "last_compute_intro":
            title = f"k_tile={kt}: 最后计算"
        elif phase == "last_mma":
            title = f"最后 fragment [{frag_idx}/16]"
        elif phase == "writeback":
            title = f"Writeback Warp {frame.get('warp','?')}"
        else:
            title = ""

        ax.set_title(f"128x128 C Tile — {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("N (列)", fontsize=9)
        ax.set_ylabel("M (行)", fontsize=9)

    def draw_data_flow(ax, frame):
        """画 A/B 矩阵 + 当前数据窗口"""
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 10)

        phase = frame["phase"]
        kt = frame.get("k_tile", 0)
        read_buf = frame.get("read_buf", 0)
        write_buf = frame.get("write_buf", -1)

        # --- 左侧: A 矩阵 (128 × K_total) ---
        # 简化显示: 画一个矩形代表 A
        rect_a = FancyBboxPatch((0.3, 5.5), 4, 4, boxstyle="round,pad=0.1",
                                 facecolor="#FFCDD2", edgecolor="#E53935", linewidth=2)
        ax.add_patch(rect_a)
        ax.text(2.3, 8.8, "A 矩阵\n128 rows x K cols", ha="center", fontsize=9, fontweight="bold")

        # 当前 k_tile 窗口
        k_start = kt * K_TILE
        k_end = k_start + K_TILE
        window_width = 4 * (K_TILE / K_TOTAL)
        ax.add_patch(Rectangle((0.5, 6.2), 3.5, 0.6, facecolor=C_HIGHLIGHT,
                                edgecolor="#333", linewidth=2, alpha=0.7))
        ax.text(2.3, 6.5, f"当前窗口: K=[{k_start}:{k_end}]", ha="center",
               fontsize=8, fontweight="bold", color="#333")

        # 总 K 标注
        ax.text(2.3, 5.3, f"K 总长度 = {K_TOTAL}", ha="center", fontsize=7, color="#555")

        # --- 中间: 操作符号 ---
        op_y = 7.5
        if phase in ("prefetch_A", "prefetch_B", "prefetch_sync",
                     "async_A", "async_B", "commit"):
            ax.text(5, op_y, " —[加载]—> ", ha="center", fontsize=12,
                   fontweight="bold", color=C_LOAD)
        elif phase in ("wmma_mma", "overlap_mma", "last_mma"):
            ax.text(5, op_y, " —[计算]—> ", ha="center", fontsize=12,
                   fontweight="bold", color=C_COMP)
        elif phase == "wait_sync_swap":
            ax.text(5, op_y, " —[同步]—> ", ha="center", fontsize=12,
                   fontweight="bold", color=C_SYNC)
        elif phase == "writeback":
            ax.text(5, op_y, " —[写回]—> ", ha="center", fontsize=12,
                   fontweight="bold", color=C_DONE)
        else:
            ax.text(5, op_y, " ... ", ha="center", fontsize=12, color="#999")

        # --- 右侧: Shared Memory 简化表示 ---
        # read buffer
        rb_y = 7.5
        rect_rb = FancyBboxPatch((6.0, rb_y - 0.7), 4.5, 2.2, boxstyle="round,pad=0.1",
                                  facecolor=C_BUF0 if read_buf == 0 else C_BUF1,
                                  edgecolor="#333", linewidth=2)
        ax.add_patch(rect_rb)
        smem_label = f"buf[{read_buf}] (READ)\nA_buf: 128x16\nB_buf: 16x128\n← Tensor Core 从此读取"
        ax.text(8.25, rb_y + 0.4, smem_label, ha="center", fontsize=8, fontweight="bold")

        # write buffer (if active)
        if write_buf >= 0 and phase in ("async_A", "async_B", "commit", "overlap_mma"):
            wb_y = 4
            rect_wb = FancyBboxPatch((6.0, wb_y - 0.7), 4.5, 2.2, boxstyle="round,pad=0.1",
                                      facecolor=C_BUF1 if write_buf == 1 else C_BUF0,
                                      edgecolor=C_LOAD, linewidth=3, linestyle="--")
            ax.add_patch(rect_wb)
            wlabel = f"buf[{write_buf}] (WRITE)\nA_buf: 128x16\nB_buf: 16x128\n← DMA (cp.async) 写入中"
            ax.text(8.25, wb_y + 0.4, wlabel, ha="center", fontsize=8, fontweight="bold",
                   color=C_LOAD)

            # 重叠标注
            if phase == "overlap_mma":
                ax.annotate("重叠!\n计算+搬运\n同时进行",
                           xy=(8.25, rb_y - 0.7), xytext=(13, 6),
                           arrowprops=dict(arrowstyle="->", color="#FF6F00", lw=2),
                           fontsize=10, fontweight="bold", color="#FF6F00",
                           bbox=dict(boxstyle="round", facecolor="#FFF9C4"))

        # --- 右下: C 矩阵 (输出) ---
        rect_c = FancyBboxPatch((13, 1), 5, 3.5, boxstyle="round,pad=0.1",
                                 facecolor="#E1BEE7", edgecolor="#8E24AA", linewidth=2)
        ax.add_patch(rect_c)
        ax.text(15.5, 3.8, "C 输出 Tile\n128x128\n(float32)", ha="center",
               fontsize=9, fontweight="bold")

        # 完成的 k_tile 进度
        c_progress = kt / (K_TOTAL // K_TILE)
        if phase in ("writeback",):
            c_progress = 1.0
        elif phase in ("last_compute_intro", "last_mma"):
            c_progress = 0.85

        # 进度条
        bar_width = 3.5
        ax.add_patch(Rectangle((13.75, 1.5), bar_width, 0.4, facecolor="#ddd",
                                edgecolor="#999", linewidth=1))
        ax.add_patch(Rectangle((13.75, 1.5), bar_width * c_progress, 0.4,
                                facecolor=C_DONE, edgecolor="#333", linewidth=1))
        ax.text(15.5, 1.2, f"C 完成度: {c_progress*100:.0f}%", ha="center",
               fontsize=8, fontweight="bold", color=C_DONE)

    def draw_buffer_status(ax, frame):
        """画 shared memory 双缓冲状态条"""
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 6)

        phase = frame["phase"]
        kt = frame.get("k_tile", 0)
        read_buf = frame.get("read_buf", 0)
        write_buf = frame.get("write_buf", -1)

        # 两个 buffer 的状态条
        for buf_id in [0, 1]:
            y = 4.5 - buf_id * 2.5
            is_read = (buf_id == read_buf)
            is_write = (write_buf >= 0 and buf_id == write_buf)

            # 标签
            if is_read and is_write:
                status = "READ+WRITE"
                fc = "#FFE082"; ec = "#FF6F00"; lw = 3
            elif is_read:
                status = "READ (计算用)"
                fc = C_BUF0 if buf_id == 0 else C_BUF1
                ec = "#1565C0"; lw = 2.5
            elif is_write:
                status = "WRITE (DMA 搬运中)"
                fc = "#FFE0B2"; ec = C_LOAD; lw = 3
            else:
                status = "空闲"
                fc = "#EEEEEE"; ec = "#999"; lw = 1

            # A_buf
            ax.add_patch(FancyBboxPatch((0.3, y - 0.6), 6, 1.5, boxstyle="round,pad=0.05",
                                        facecolor=fc, edgecolor=ec, linewidth=lw))
            ax.text(3.3, y + 0.15, f"A_buf[{buf_id}][128][16]", ha="center",
                   fontsize=9, fontweight="bold")

            # B_buf
            ax.add_patch(FancyBboxPatch((7.5, y - 0.6), 6.5, 1.5, boxstyle="round,pad=0.05",
                                        facecolor=fc, edgecolor=ec, linewidth=lw))
            ax.text(10.75, y + 0.15, f"B_buf[{buf_id}][16][128]", ha="center",
                   fontsize=9, fontweight="bold")

            # 状态标注
            ax.text(15, y + 0.15, status, fontsize=9, fontweight="bold",
                   color=ec, va="center")

        # 标题
        ax.set_title("Shared Memory 双缓冲状态", fontsize=11, fontweight="bold", loc="left")

        # 图例
        ax.text(0.3, 0.3, "图例:", fontsize=7, color="#555")
        for i, (color, label) in enumerate([
            (C_BUF0, "buf[0]"), (C_BUF1, "buf[1]"),
            (C_LOAD, "DMA 加载中"), ("#FFF9C4", "重叠进行中")
        ]):
            ax.add_patch(Rectangle((2.5 + i * 3.5, 0.1), 0.5, 0.3,
                                    facecolor=color, edgecolor="#333", linewidth=1))
            ax.text(3.1 + i * 3.5, 0.25, label, fontsize=7, va="center")

    def draw_info_panel(ax, frame, frame_idx):
        """画代码位置 + 状态信息"""
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        phase = frame["phase"]
        desc = frame.get("desc", "")
        kt = frame.get("k_tile", 0)
        mi = frame.get("mi", -1)
        ni = frame.get("ni", -1)

        # 总进度
        progress_pct = (frame_idx + 1) / NUM_FRAMES * 100
        ax.text(5, 9.8, f"Step {frame_idx + 1}/{NUM_FRAMES} ({progress_pct:.1f}%)",
               ha="center", fontsize=12, fontweight="bold",
               bbox=dict(boxstyle="round", facecolor="#263238", edgecolor="#37474F",
                        alpha=0.9), color="white")

        # 当前操作描述
        ax.text(5, 9.0, desc, ha="center", fontsize=10, fontweight="bold",
               color="#FF6F00",
               bbox=dict(boxstyle="round", facecolor="#FFF9C4", edgecolor="#FF6F00", alpha=0.9))

        # 代码位置高亮
        code_lines = [
            ("", ""),
            ("// === 预取 buf[0] ===", "comment"),
            ("for (...) *(short2*)&A_buf[0][r][c] = ...;", "prefetch_A"),
            ("for (...) *(short2*)&B_buf[0][r][c] = ...;", "prefetch_B"),
            ("__syncthreads();", "prefetch_sync"),
            ("int read_buf = 0;", "compute_intro"),
            ("", ""),
            ("// === K 维度循环 ===", "comment"),
            ("for (int kb = 16; kb < K; kb += 16) {", "loop"),
            ("  int write_buf = 1 - read_buf;", "loop"),
            ("  cp.async A->A_buf[write_buf];  ", "async_A"),
            ("  cp.async B->B_buf[write_buf];  ", "async_B"),
            ("  cp.async.commit_group;         ", "commit"),
            ("  // WMMA 从 read_buf 计算        ", "comment"),
            ("  for mi: for ni:                 ", "wmma_loop"),
            ("    load a[mi]; load b[ni];       ", "wmma_load"),
            ("    mma_sync(c, a, b, c);         ", "wmma_mma"),
            ("  cp.async.wait_group 0;          ", "wait_sync_swap"),
            ("  __syncthreads(); read_buf=write_buf;", "wait_sync_swap"),
            ("}", "loop"),
            ("", ""),
            ("// === 最后 tile + writeback ===", "comment"),
            ("// (同上, 不发起 cp.async)", "last_compute_intro"),
            ("store_matrix_sync(C, c[], N, ...);", "writeback"),
        ]

        y_start = 8.2
        for i, (line, tag) in enumerate(code_lines):
            y = y_start - i * 0.35
            if not line:
                continue

            # 匹配当前 phase
            is_current = False
            if phase == tag:
                is_current = True
            elif tag == "wmma_loop" and phase in ("wmma_load_a", "wmma_load_b", "wmma_mma", "overlap_mma", "last_mma"):
                is_current = True
            elif tag == "wmma_load" and phase in ("wmma_load_a", "wmma_load_b"):
                is_current = True
            elif tag == "loop" and phase in ("async_A", "async_B", "commit", "wmma_load_a",
                                              "wmma_load_b", "wmma_mma", "overlap_mma",
                                              "wait_sync_swap"):
                is_current = False  # 只是循环头

            if is_current:
                # 高亮当前行
                ax.add_patch(FancyBboxPatch((0.2, y - 0.15), 9.6, 0.32,
                                            boxstyle="round,pad=0.02",
                                            facecolor="#FFE082", edgecolor="#FF6F00",
                                            linewidth=2))
                ax.text(5, y, line, ha="center", fontsize=8.5, fontweight="bold",
                       color="#333", fontfamily="monospace")
                # 箭头指示
                ax.text(0.1, y, ">>>", fontsize=8, color="#FF6F00", fontweight="bold",
                       va="center", fontfamily="monospace")
            elif tag == "comment":
                ax.text(5, y, line, ha="center", fontsize=8, color="#999",
                       fontfamily="monospace", style="italic")
            elif tag:
                ax.text(5, y, line, ha="center", fontsize=8, color="#555",
                       fontfamily="monospace")
            else:
                ax.text(5, y, line, ha="center", fontsize=8, color="#999",
                       fontfamily="monospace")

        # 关键数值
        mi = frame.get("mi", -1)
        ni = frame.get("ni", -1)
        if mi >= 0 and ni >= 0:
            info_text = (
                f"当前 WMMA fragment:\n"
                f"  mi={mi}, ni={ni}, frag_idx={mi*4+ni}\n"
                f"  a[{mi}]: A_buf[buf][wy*64+{mi}*16][:]\n"
                f"  b[{ni}]: B_buf[buf][:][wx*64+{ni}*16]\n"
                f"  c[{mi*4+ni}]: 累加器 16x16 float32"
            )
            ax.text(5, 0.8, info_text, ha="center", fontsize=7.5,
                   color="#333", fontfamily="monospace",
                   bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8))

    def update(frame_idx):
        frame = frames[frame_idx]
        draw_c_tile(ax_tile, frame)
        draw_data_flow(ax_data, frame)
        draw_buffer_status(ax_buf, frame)
        draw_info_panel(ax_info, frame, frame_idx)

        fig.suptitle("V9 gemm_async_wmma — 逐步串行演示\n"
                    f"M=N=128, K={K_TOTAL}, K_tile=16, 4 warps, 16 WMMA fragments/warp",
                    fontsize=14, fontweight="bold", y=0.995)
        return [ax_tile, ax_data, ax_buf, ax_info]

    anim = FuncAnimation(fig, update, frames=NUM_FRAMES, interval=400, blit=False)

    path = os.path.join(OUT_DIR, "gemm_v9_step_by_step.gif")
    writer = matplotlib.animation.PillowWriter(fps=2.5)
    anim.save(path, writer=writer, dpi=80)
    plt.close(fig)
    print(f"[done] {path} ({NUM_FRAMES} frames, {NUM_FRAMES/2.5:.0f}s)")


if __name__ == "__main__":
    print(f"Generating step-by-step animation with {NUM_FRAMES} frames...")
    create_step_animation()
