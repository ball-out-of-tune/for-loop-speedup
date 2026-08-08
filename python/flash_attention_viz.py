#!/usr/bin/env python3
"""
Flash Attention V1 vs V2 计算顺序可视化
==========================================

展示 Flash Attention V1 和 V2 的块级计算顺序差异。

核心区别:
  V1: 外循环遍历 K,V 块, 内循环遍历 Q 块 → S=Q@K^T 按列优先遍历
  V2: 外循环遍历 Q 块, 内循环遍历 K,V 块 → S=Q@K^T 按行优先遍历

输出文件:
  - fa_v1_order.png          V1 静态计算顺序图
  - fa_v2_order.png          V2 静态计算顺序图
  - fa_v1_animation.gif      V1 逐帧动画
  - fa_v2_animation.gif      V2 逐帧动画
  - fa_comparison.gif        V1 vs V2 左右对比合成 GIF
  - fa_v1_stepNNN.png        V1 关键步骤详解
  - fa_v2_stepNNN.png        V2 关键步骤详解
"""

import gc
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============================================================================
# 中文字体配置
# ============================================================================
_CN_FONT = None
for _fname in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
    for _f in fm.fontManager.ttflist:
        if _f.name == _fname:
            _CN_FONT = _f
            break
    if _CN_FONT:
        break

if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT.name
    plt.rcParams["font.monospace"] = [_CN_FONT.name, "DejaVu Sans Mono"]
else:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]

plt.rcParams["axes.unicode_minus"] = False

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# ============================================================================
# 配置参数
# ============================================================================
N = 8       # 序列长度 (seqlen)
d = 4       # head 维度
Br = 2      # Q 行块大小
Bc = 2      # K/V 块大小 (在 K^T 中为列块)

assert N % Br == 0, "N 必须能被 Br 整除"
assert N % Bc == 0, "N 必须能被 Bc 整除"

NUM_Q_BLOCKS = N // Br       # 4
NUM_KV_BLOCKS = N // Bc      # 4
NUM_STEPS = NUM_Q_BLOCKS * NUM_KV_BLOCKS  # 每个版本 16 步

# ---- 颜色方案 (沿用 tiled_matmul_viz 的配色) ----
Q_EDGE   = "#E53935"    # Q 活动块边框: 红
Q_FILL   = "#FFCDD2"    # Q 活动块填充: 浅红
K_EDGE   = "#1E88E5"    # K^T 活动块边框: 蓝
K_FILL   = "#BBDEFB"    # K^T 活动块填充: 浅蓝
V_EDGE   = "#FB8C00"    # V 活动块边框: 橙
V_FILL   = "#FFE0B2"    # V 活动块填充: 浅橙
O_EDGE   = "#8E24AA"    # O 活动块边框: 紫
O_FILL   = "#E1BEE7"    # O 活动块填充: 浅紫
O_TOUCHED_BASE = "#C8E6C9"  # O 已触及区域底色: 绿

# ---- 随机种子 ----
np.random.seed(42)

# ============================================================================
# 生成矩阵
# ============================================================================
Q = np.round(np.random.randn(N, d), 1)
K = np.round(np.random.randn(N, d), 1)
V = np.round(np.random.randn(N, d), 1)

# K^T 用于显示 (d × N)
KT = K.T  # (d, N)


# ============================================================================
# V1 预计算: 外循环 K,V 块, 内循环 Q 块
# ============================================================================
def compute_v1_history(Q, K, V, Br, Bc):
    """按 V1 算法逐步计算 O, 记录每步状态。

    V1: for k_start in K_blocks:          (outer)
           for q_start in Q_blocks:       (inner)
             compute partial attention
             online rescale O[q_start]

    Returns:
      steps: [(k_start, q_start), ...] 共 NUM_STEPS 步
      O_history: [O after each step]
      touch_history: [(N,d) touch count after each step]
    """
    O_cur = np.zeros((N, d))
    m = np.full(N, -np.inf)   # running max per row
    l = np.zeros(N)            # running sum exp per row
    touch_count = np.zeros((N, d), dtype=int)

    steps = []
    O_history = []
    touch_history = []

    for k_start in range(0, N, Bc):              # outer: K,V blocks
        Kj = K[k_start:k_start + Bc]              # (Bc, d)
        Vj = V[k_start:k_start + Bc]              # (Bc, d)
        for q_start in range(0, N, Br):           # inner: Q blocks
            Qi = Q[q_start:q_start + Br]           # (Br, d)
            S = Qi @ Kj.T                          # (Br, Bc)

            # Online safe softmax rescaling
            row_max = S.max(axis=1)                 # (Br,)
            m_new = np.maximum(m[q_start:q_start + Br], row_max)

            # Rescale old O and l if max changed
            exp_diff = np.exp(m[q_start:q_start + Br] - m_new)  # (Br,)
            O_cur[q_start:q_start + Br] *= exp_diff[:, None]
            l[q_start:q_start + Br] *= exp_diff

            # Add new contributions
            P = np.exp(S - m_new[:, None])          # (Br, Bc) — safe softmax
            l_new_contrib = P.sum(axis=1)           # (Br,)
            l[q_start:q_start + Br] += l_new_contrib
            O_cur[q_start:q_start + Br] += P @ Vj

            m[q_start:q_start + Br] = m_new
            touch_count[q_start:q_start + Br, :] += 1

            steps.append((k_start, q_start))
            # 归一化显示: O_cur 存的是未归一化的分子, 除以 l 得到当前最优估计
            O_display = O_cur.copy()
            valid = l > 0
            O_display[valid] = O_cur[valid] / l[valid, None]
            O_history.append(O_display)
            touch_history.append(touch_count.copy())

    return steps, O_history, touch_history


# ============================================================================
# V2 预计算: 外循环 Q 块, 内循环 K,V 块
# ============================================================================
def compute_v2_history(Q, K, V, Br, Bc):
    """按 V2 算法逐步计算 O, 记录每步状态。

    V2: for q_start in Q_blocks:            (outer)
           local m_i, l_i, O_i
           for k_start in K_blocks:         (inner)
             compute partial attention
             local rescale O_i
           write O_i back to O              (once per Q block)

    Returns:
      steps: [(q_start, k_start), ...] 共 NUM_STEPS 步
      O_history: [O after each step]
      touch_history: [(N,d) touch count after each step]
    """
    O_cur = np.zeros((N, d))
    touch_count = np.zeros((N, d), dtype=int)

    steps = []
    O_history = []
    touch_history = []

    for q_start in range(0, N, Br):              # outer: Q blocks
        Qi = Q[q_start:q_start + Br]              # (Br, d)
        m_i = np.full(Br, -np.inf)
        l_i = np.zeros(Br)
        O_i = np.zeros((Br, d))

        for k_start in range(0, N, Bc):           # inner: K,V blocks
            Kj = K[k_start:k_start + Bc]           # (Bc, d)
            Vj = V[k_start:k_start + Bc]           # (Bc, d)
            S = Qi @ Kj.T                          # (Br, Bc)

            # Local online safe softmax rescaling
            row_max = S.max(axis=1)                 # (Br,)
            m_new = np.maximum(m_i, row_max)

            exp_diff = np.exp(m_i - m_new)          # (Br,)
            O_i *= exp_diff[:, None]
            l_i *= exp_diff

            P = np.exp(S - m_new[:, None])          # (Br, Bc)
            l_i += P.sum(axis=1)
            O_i += P @ Vj

            m_i = m_new

            # 在 V2 中, O 块在内循环每次迭代都被 "touched" (accumulated locally)
            # 我们显示的是如果此时写回 O 的中间状态
            # 实际 V2 只在最后写回, 但可视化展示每一小步的进展
            O_temp = O_cur.copy()
            # 归一化并写回当前 Q 块
            O_temp[q_start:q_start + Br] = O_i / l_i[:, None]
            touch_count_temp = touch_count.copy()
            touch_count_temp[q_start:q_start + Br, :] += 1

            steps.append((q_start, k_start))
            O_history.append(O_temp)
            touch_history.append(touch_count_temp.copy())

        # 外循环结束时真正写回 O (在 V2 实现中只做一次)
        O_cur[q_start:q_start + Br] = O_i / l_i[:, None]
        touch_count[q_start:q_start + Br, :] = NUM_KV_BLOCKS  # 标记为完整

        # 更新已写入的历史记录 (覆盖内循环的临时状态)
        # 我们保留内循环的逐步状态用于动画, 不做覆盖

    return steps, O_history, touch_history


# ============================================================================
# 计算
# ============================================================================
v1_steps, v1_O_hist, v1_touch_hist = compute_v1_history(Q, K, V, Br, Bc)
v2_steps, v2_O_hist, v2_touch_hist = compute_v2_history(Q, K, V, Br, Bc)

# 最终 O (两种算法应该一致)
O_final = v1_O_hist[-1]


# ============================================================================
# 辅助函数: 绘制带高亮的矩阵 (支持非方阵)
# ============================================================================
def draw_rect_matrix(ax, data, hi_row=None, hi_col=None,
                     hi_color=None, hi_fill=None,
                     title="", fmt=".1f", touched=None,
                     row_block_size=None, col_block_size=None,
                     show_values=True):
    """在 ax 上绘制 rows×cols 矩阵, 并高亮指定行块/列块。

    Parameters
    ----------
    ax : Axes
    data : (rows, cols) ndarray
    hi_row : int | None — 高亮行起始索引 (行块高亮)
    hi_col : int | None — 高亮列起始索引 (列块高亮)
    hi_color : str — 高亮边框颜色
    hi_fill  : str — 高亮填充颜色
    title : str
    fmt : str — 数值格式化
    touched : (rows, cols) ndarray | None — 每格被触及次数, 用于着色
    row_block_size : int | None — 行块大小 (画粗分隔线)
    col_block_size : int | None — 列块大小 (画粗分隔线)
    """
    rows, cols = data.shape
    extent = [-0.5, cols - 0.5, rows - 0.5, -0.5]

    # 底色
    if touched is not None:
        max_touch = touched.max()
        if max_touch > 0:
            green_intensity = np.clip(touched / max_touch, 0, 1)
        else:
            green_intensity = np.zeros_like(touched)
        # 从白到绿
        r = 1.0 - green_intensity * 0.22
        g = 1.0 - green_intensity * 0.10
        b = 1.0 - green_intensity * 0.21
        rgb = np.stack([r, g, b], axis=-1)
        ax.imshow(rgb, extent=extent, aspect="equal", origin="upper")
    else:
        bg = np.ones((rows, cols, 3)) * 0.95
        ax.imshow(bg, extent=extent, aspect="equal", origin="upper")

    # 活动块高亮填充
    if hi_fill is not None:
        if hi_row is not None and row_block_size is not None:
            rect_fill = Rectangle(
                (-0.5, hi_row - 0.5), cols, row_block_size,
                linewidth=0, facecolor=hi_fill, zorder=2,
            )
            ax.add_patch(rect_fill)
        if hi_col is not None and col_block_size is not None:
            rect_fill = Rectangle(
                (hi_col - 0.5, -0.5), col_block_size, rows,
                linewidth=0, facecolor=hi_fill, zorder=2,
            )
            ax.add_patch(rect_fill)

    # 数值文字
    if show_values:
        fontsize = max(6, min(8, 80 // max(rows, cols)))
        for i in range(rows):
            for j in range(cols):
                val = data[i, j]
                ax.text(
                    j, i, f"{val:{fmt}}",
                    ha="center", va="center",
                    fontsize=fontsize, color="#333333",
                    zorder=5,
                )

    # 活动块边框
    if hi_color is not None:
        if hi_row is not None and row_block_size is not None:
            rect_edge = Rectangle(
                (-0.5, hi_row - 0.5), cols, row_block_size,
                linewidth=3, edgecolor=hi_color, facecolor="none", zorder=6,
            )
            ax.add_patch(rect_edge)
        if hi_col is not None and col_block_size is not None:
            rect_edge = Rectangle(
                (hi_col - 0.5, -0.5), col_block_size, rows,
                linewidth=3, edgecolor=hi_color, facecolor="none", zorder=6,
            )
            ax.add_patch(rect_edge)

    # 块分隔线 (较粗)
    if row_block_size is not None:
        for k in range(0, rows + 1, row_block_size):
            ax.axhline(y=k - 0.5, color="#666666", linewidth=1.2, zorder=4)
    if col_block_size is not None:
        for k in range(0, cols + 1, col_block_size):
            ax.axvline(x=k - 0.5, color="#666666", linewidth=1.2, zorder=4)

    # 普通网格线 (较细)
    for k in range(0, rows + 1):
        ax.axhline(y=k - 0.5, color="#cccccc", linewidth=0.5, zorder=0)
    for k in range(0, cols + 1):
        ax.axvline(x=k - 0.5, color="#cccccc", linewidth=0.5, zorder=0)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)


# ============================================================================
# 辅助函数: 格式化循环值
# ============================================================================
def format_loop_values(current, values):
    """把值序列格式化为字符串, current 用方括号高亮。"""
    parts = []
    for v in values:
        if v == current:
            parts.append(f"[{v}]")
        else:
            parts.append(str(v))
    return "{" + ", ".join(parts) + "}"


# ============================================================================
# 静态图: S=Q@K^T 矩阵块计算顺序
# ============================================================================
def create_static_order_figure(version="v1"):
    """生成 S=Q@K^T 矩阵的块计算顺序静态图。

    Parameters
    ----------
    version : "v1" | "v2"
    """
    fig, (ax_main, ax_legend) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 2]},
    )

    # S = Q @ K^T: (N, N)
    S_fake = np.random.seed(99) or np.round(Q @ KT, 2)
    S_fake = Q @ KT  # (N, N)

    num_row_blocks = N // Br     # Q blocks = 4
    num_col_blocks = N // Bc     # K^T blocks = 4

    # 为每个块赋予计算顺序编号
    block_order = np.zeros((num_row_blocks, num_col_blocks), dtype=int)
    order = 1
    if version == "v1":
        # V1: 列优先 — K^T 列块外循环, Q 行块内循环
        for c in range(num_col_blocks):        # outer: K blocks
            for r in range(num_row_blocks):    # inner: Q blocks
                block_order[r, c] = order
                order += 1
    else:
        # V2: 行优先 — Q 行块外循环, K^T 列块内循环
        for r in range(num_row_blocks):        # outer: Q blocks
            for c in range(num_col_blocks):    # inner: K blocks
                block_order[r, c] = order
                order += 1

    # 将 block_order 扩展到每个 cell
    order_colors = np.zeros((N, N))
    for r in range(num_row_blocks):
        for c in range(num_col_blocks):
            ri, ci = r * Br, c * Bc
            order_colors[ri:ri + Br, ci:ci + Bc] = block_order[r, c]

    extent = [-0.5, N - 0.5, N - 0.5, -0.5]

    ax_main.imshow(
        order_colors, cmap=plt.cm.YlOrRd, extent=extent,
        aspect="equal", origin="upper", vmin=1,
        vmax=num_row_blocks * num_col_blocks,
    )

    # 块编号
    from matplotlib.patches import Circle
    for r in range(num_row_blocks):
        for c in range(num_col_blocks):
            order_num = block_order[r, c]
            ri_center = r * Br + Br / 2 - 0.5
            ci_center = c * Bc + Bc / 2 - 0.5
            circle = Circle(
                (ci_center, ri_center), 0.55,
                facecolor="white", edgecolor="#333333",
                linewidth=2, zorder=9,
            )
            ax_main.add_patch(circle)
            ax_main.text(
                ci_center, ri_center, str(order_num),
                ha="center", va="center",
                fontsize=14, fontweight="bold", color="#333333", zorder=10,
            )

    # 块分隔线
    for k in range(0, N + 1):
        lw = 1.5 if k % Bc == 0 or k % Br == 0 else 0.3
        color = "white" if k % Bc == 0 or k % Br == 0 else "#dddddd"
        ax_main.axhline(y=k - 0.5, color=color, linewidth=lw)
        ax_main.axvline(x=k - 0.5, color=color, linewidth=lw)

    # 行列标签
    for i in range(N):
        ax_main.text(-0.8, i, f"Q{i}", ha="right", va="center",
                     fontsize=7, color="#E53935")
    for j in range(N):
        ax_main.text(j, N + 0.3, f"K{j}", ha="center", va="top",
                     fontsize=7, color="#1E88E5")

    # 遍历方向箭头
    if version == "v1":
        # V1: 沿列方向箭头 (K block 外循环)
        for c in range(num_col_blocks):
            col_center = c * Bc + Bc / 2 - 0.5
            ax_main.annotate(
                "", xy=(col_center, N + 1.2), xytext=(col_center, N + 0.3),
                arrowprops=dict(arrowstyle="->", color="#1E88E5", lw=2),
            )
        ax_main.text(N / 2 - 0.5, N + 1.8,
                     "K 块遍历 (外循环) >", ha="center",
                     fontsize=9, color="#1E88E5", fontweight="bold")

        # 沿行方向箭头 (Q block 内循环)
        for r in range(num_row_blocks):
            row_center = r * Br + Br / 2 - 0.5
            ax_main.annotate(
                "", xy=(-1.5, row_center), xytext=(-0.8, row_center),
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=2),
            )
        ax_main.text(-1.5, N / 2 - 0.5, "Q 块遍历\n(内循环)",
                     ha="center", va="center", fontsize=9, color="#E53935",
                     fontweight="bold", rotation=90)
    else:
        # V2: 沿行方向箭头 (Q block 外循环)
        for r in range(num_row_blocks):
            row_center = r * Br + Br / 2 - 0.5
            ax_main.annotate(
                "", xy=(-1.5, row_center), xytext=(-0.8, row_center),
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=2),
            )
        ax_main.text(-1.5, N / 2 - 0.5, "Q 块遍历\n(外循环)",
                     ha="center", va="center", fontsize=9, color="#E53935",
                     fontweight="bold", rotation=90)

        # 沿列方向箭头 (K block 内循环)
        for c in range(num_col_blocks):
            col_center = c * Bc + Bc / 2 - 0.5
            ax_main.annotate(
                "", xy=(col_center, N + 1.2), xytext=(col_center, N + 0.3),
                arrowprops=dict(arrowstyle="->", color="#1E88E5", lw=2),
            )
        ax_main.text(N / 2 - 0.5, N + 1.8,
                     "K 块遍历 (内循环) >", ha="center",
                     fontsize=9, color="#1E88E5", fontweight="bold")

    ax_main.set_xlim(-2.5, N + 1.0)
    ax_main.set_ylim(N + 2.5, -2.0)
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    loop_desc = "外: K → 内: Q" if version == "v1" else "外: Q → 内: K"
    ax_main.set_title(
        f"Flash Attention {version.upper()}: S=Q@K^T 块计算顺序 (N={N}, Br={Br}, Bc={Bc})\n"
        f"{loop_desc}  |  颜色越深 = 越早完成  |  编号 = 计算顺序",
        fontsize=14, fontweight="bold", pad=15,
    )

    # --- 右侧: 算法描述 ---
    ax_legend.axis("off")

    if version == "v1":
        algo_text = (
            "Flash Attention V1 算法\n"
            "══════════════════════\n\n"
            "外层循环 (K,V 块):\n"
            "  for k_start in {0, Bc, 2Bc, ...}:\n"
            "    load Kj = K[k_start:k_start+Bc]\n"
            "    load Vj = V[k_start:k_start+Bc]\n\n"
            "  内层循环 (Q 块):\n"
            "    for q_start in {0, Br, 2Br, ...}:\n"
            "      load Qi = Q[q_start:q_start+Br]\n"
            "      S = Qi @ Kj.T\n"
            "      ONLINE RESCALE O[q_start]\n"
            "      O[q_start] += softmax(S) @ Vj\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "特点:\n"
            "• K,V 块加载一次, 所有 Q 块处理\n"
            "• 每个 Q 块都要做 online rescaling\n"
            "• non-matmul FLOPs 较多\n"
            "• 适合 GPU SM 数较少的情况\n\n"
            "S 矩阵遍历: 列优先 (↓)\n"
            "  即沿 K 维度的块遍历"
        )
    else:
        algo_text = (
            "Flash Attention V2 算法\n"
            "══════════════════════\n\n"
            "外层循环 (Q 块):\n"
            "  for q_start in {0, Br, 2Br, ...}:\n"
            "    load Qi = Q[q_start:q_start+Br]\n"
            "    O_i, m_i, l_i = 0, -inf, 0\n\n"
            "  内层循环 (K,V 块):\n"
            "    for k_start in {0, Bc, 2Bc, ...}:\n"
            "      load Kj, Vj\n"
            "      S = Qi @ Kj.T\n"
            "      LOCAL rescale O_i\n"
            "      O_i += softmax(S) @ Vj\n\n"
            "  写回: O[q_start] = O_i / l_i\n"
            "  (整个 Q 块只写回一次!)\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "特点:\n"
            "• Q 块加载一次, 所有 K,V 块处理\n"
            "• rescaling 只在局部进行\n"
            "• non-matmul FLOPs 大幅减少\n"
            "• HBM 写入减少 ~2-3x\n"
            "• 支持 seqlen 维度并行\n\n"
            "S 矩阵遍历: 行优先 (→)\n"
            "  即沿 Q 维度的块遍历"
        )

    ax_legend.text(
        0.05, 0.95, algo_text,
        transform=ax_legend.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8),
        linespacing=1.3,
    )

    fig.tight_layout(pad=2)
    out_path = os.path.join(os.path.dirname(__file__),
                            f"fa_{version}_order.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[static] Saved: {out_path}")


# ============================================================================
# 动画 GIF
# ============================================================================
def create_animation(version="v1"):
    """为指定版本的 Flash Attention 生成逐帧动画 GIF。

    Parameters
    ----------
    version : "v1" | "v2"
    """
    steps_data = v1_steps if version == "v1" else v2_steps
    O_hist = v1_O_hist if version == "v1" else v2_O_hist
    touch_hist = v1_touch_hist if version == "v1" else v2_touch_hist

    fig = plt.figure(figsize=(20, 14))

    gs = fig.add_gridspec(2, 1, height_ratios=[5.5, 2.5], hspace=0.35)
    gs_top = gs[0].subgridspec(2, 2, hspace=0.4, wspace=0.35)
    # Q: (0,0), K^T: (0,1), V: (1,0), O: (1,1)
    ax_q  = fig.add_subplot(gs_top[0, 0])
    ax_kt = fig.add_subplot(gs_top[0, 1])
    ax_v  = fig.add_subplot(gs_top[1, 0])
    ax_o  = fig.add_subplot(gs_top[1, 1])
    ax_loop = fig.add_subplot(gs[1])
    ax_loop.axis("off")

    all_q_vals = list(range(0, N, Br))   # [0, 2, 4, 6]
    all_k_vals = list(range(0, N, Bc))   # [0, 2, 4, 6]

    def update(frame_idx):
        if version == "v1":
            k_start, q_start = steps_data[frame_idx]
        else:
            q_start, k_start = steps_data[frame_idx]

        O_state = O_hist[frame_idx]
        touch_state = touch_hist[frame_idx]

        # 清除所有轴
        for ax in [ax_q, ax_kt, ax_v, ax_o]:
            ax.clear()
        ax_loop.clear()
        ax_loop.axis("off")

        # --- Q 矩阵 (N x d) — 左上 ---
        q_title = f"Q (N={N}, d={d}) — 活动行 [{q_start}:{q_start+Br}]"
        if version == "v2":
            q_title += "  [TB" + "|TB".join(str(i) for i in range(NUM_Q_BLOCKS)) + "]"
        draw_rect_matrix(
            ax_q, Q,
            hi_row=q_start, hi_col=None,
            hi_color=Q_EDGE, hi_fill=Q_FILL,
            title=q_title,
            row_block_size=Br, col_block_size=None,
        )
        # V2: 在 Q 矩阵左侧标注 Thread Block 分配 (体现 seqlen 并行)
        if version == "v2":
            for tb_idx in range(NUM_Q_BLOCKS):
                y_center = tb_idx * Br + Br / 2 - 0.5
                is_active = (tb_idx == q_start // Br)
                tb_color = "#2E7D32" if is_active else "#999999"
                tb_weight = "bold" if is_active else "normal"
                ax_q.text(-2.0, y_center, f"TB{tb_idx}",
                         ha="center", va="center", fontsize=7,
                         color=tb_color, fontweight=tb_weight, zorder=10)
            ax_q.set_xlim(-3.0, d - 0.5)  # 扩展左边距给 TB 标签

        # --- K^T 矩阵 (d x N) — 右上 ---
        draw_rect_matrix(
            ax_kt, KT,
            hi_row=None, hi_col=k_start,
            hi_color=K_EDGE, hi_fill=K_FILL,
            title=f"K^T (d={d}, N={N}) — 活动列 [{k_start}:{k_start+Bc}]",
            row_block_size=None, col_block_size=Bc,
        )

        # --- V 矩阵 (N x d) — 左下 ---
        draw_rect_matrix(
            ax_v, V,
            hi_row=k_start, hi_col=None,
            hi_color=V_EDGE, hi_fill=V_FILL,
            title=f"V (N={N}, d={d}) — 活动行 [{k_start}:{k_start+Bc}]",
            row_block_size=Bc, col_block_size=None,
        )

        # --- O 矩阵 (N x d) — 右下 ---
        draw_rect_matrix(
            ax_o, O_state,
            hi_row=q_start, hi_col=None,
            hi_color=O_EDGE, hi_fill=O_FILL,
            title=f"O (N={N}, d={d}) — 累加到行 [{q_start}:{q_start+Br}]",
            row_block_size=Br, col_block_size=None,
            touched=touch_state,
        )

        # --- 循环变量面板 ---
        progress_pct = (frame_idx + 1) / NUM_STEPS * 100

        q_str = format_loop_values(q_start, all_q_vals)
        k_str = format_loop_values(k_start, all_k_vals)

        # 累计 non-matmul rescaling 次数
        # V1: 每步对 Br 行做 HBM 读写 rescale; V2: 每步在 SRAM 中做 local rescale
        rescale_count = (frame_idx + 1) * Br  # 本步为止累计 rescale 的行数
        if version == "v1":
            hbm_rw_note = f"HBM  O读+rescale+写: {rescale_count}行  |  非矩阵乘操作: {rescale_count}次exp, {rescale_count * d}次mul"
        else:
            hbm_rw_note = f"SRAM  local rescale: {rescale_count}行  |  O写回HBM: 仅当Q块完成时"

        lines = []
        lines.append(f"Step {frame_idx + 1} / {NUM_STEPS}  ({progress_pct:.1f}%)")
        lines.append("")

        if version == "v1":
            lines.append("Flash Attention V1 — 循环结构:")
            lines.append(f"  for k_start in {k_str}    ← K,V 块 (外循环)")
            lines.append(f"    for q_start in {q_str}    ← Q 块 (内循环)")
        else:
            lines.append("Flash Attention V2 — 循环结构:")
            lines.append(f"  for q_start in {q_str}    ← Q 块 (外循环)")
            lines.append(f"    for k_start in {k_str}    ← K,V 块 (内循环)")

        lines.append("")
        lines.append("当前操作:")
        lines.append(f"  S = Q[{q_start}:{q_start+Br}] @ K^T[:, {k_start}:{k_start+Bc}]")
        lines.append(f"  → O[{q_start}:{q_start+Br}] += softmax(S) @ V[{k_start}:{k_start+Bc}]")

        lines.append("")
        lines.append(hbm_rw_note)

        if version == "v1":
            lines.append("")
            lines.append("并行: batch × num_heads (所有TB共享同一K,V块)")
            lines.append("       Q行块在TB内串行切换 (内循环)")
        else:
            lines.append("")
            lines.append("并行: batch × num_heads × seqlen (每Q块→独立TB)")
            lines.append(f"      {NUM_Q_BLOCKS}个Q块→{NUM_Q_BLOCKS}个Thread Block可同时运行")

        panel_text = "\n".join(lines)

        ax_loop.text(
            0.5, 0.95, panel_text,
            transform=ax_loop.transAxes,
            fontsize=11, verticalalignment="top",
            horizontalalignment="center",
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#FAFAFA",
                      edgecolor="#CCCCCC", linewidth=1.5),
        )

        return [ax_loop, ax_q, ax_kt, ax_v, ax_o]

    anim = FuncAnimation(fig, update, frames=NUM_STEPS, interval=600, blit=False)

    out_path = os.path.join(os.path.dirname(__file__),
                            f"fa_{version}_animation.gif")
    writer = matplotlib.animation.PillowWriter(fps=2)
    anim.save(out_path, writer=writer, dpi=72)
    plt.close(fig)
    print(f"[anim]  Saved: {out_path}")


# ============================================================================
# 对比合成 GIF
# ============================================================================
def create_comparison_gif():
    """生成 V1 和 V2 左右并排对比的合成 GIF。

    每帧左右各显示一个版本的矩阵状态。
    """
    import gc
    gc.collect()
    plt.close("all")
    # 使用宽幅 figure: 左侧 V1, 右侧 V2
    fig = plt.figure(figsize=(26, 13))

    # 外层: 2行 (矩阵区 + 面板区) × 2列 (V1 + V2)
    gs_outer = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[5.5, 2.5],
                                 hspace=0.35, wspace=0.25)

    # V1 侧 — 2×2 矩阵网格
    gs_top_v1 = gs_outer[0, 0].subgridspec(2, 2, hspace=0.4, wspace=0.3)
    ax_v1_q  = fig.add_subplot(gs_top_v1[0, 0])
    ax_v1_kt = fig.add_subplot(gs_top_v1[0, 1])
    ax_v1_v  = fig.add_subplot(gs_top_v1[1, 0])
    ax_v1_o  = fig.add_subplot(gs_top_v1[1, 1])
    ax_loop_v1 = fig.add_subplot(gs_outer[1, 0])
    ax_loop_v1.axis("off")

    # V2 侧 — 2×2 矩阵网格
    gs_top_v2 = gs_outer[0, 1].subgridspec(2, 2, hspace=0.4, wspace=0.3)
    ax_v2_q  = fig.add_subplot(gs_top_v2[0, 0])
    ax_v2_kt = fig.add_subplot(gs_top_v2[0, 1])
    ax_v2_v  = fig.add_subplot(gs_top_v2[1, 0])
    ax_v2_o  = fig.add_subplot(gs_top_v2[1, 1])
    ax_loop_v2 = fig.add_subplot(gs_outer[1, 1])
    ax_loop_v2.axis("off")

    # 对比 GIF 不显示 cell 数值以节省内存
    _show_vals = False

    all_q_vals = list(range(0, N, Br))
    all_k_vals = list(range(0, N, Bc))

    def update(frame_idx):
        # --- V1 ---
        k1, q1 = v1_steps[frame_idx]
        for ax in [ax_v1_q, ax_v1_kt, ax_v1_v, ax_v1_o]:
            ax.clear()
        ax_loop_v1.clear()
        ax_loop_v1.axis("off")

        draw_rect_matrix(ax_v1_q, Q,
                         hi_row=q1, hi_color=Q_EDGE, hi_fill=Q_FILL,
                         title=f"V1 Q 行[{q1}:{q1+Br}] (TB内串行)",
                         row_block_size=Br, col_block_size=None,
                         show_values=_show_vals)
        draw_rect_matrix(ax_v1_kt, KT,
                         hi_col=k1, hi_color=K_EDGE, hi_fill=K_FILL,
                         title=f"V1 K^T 列[{k1}:{k1+Bc}]",
                         row_block_size=None, col_block_size=Bc,
                         show_values=_show_vals)
        draw_rect_matrix(ax_v1_v, V,
                         hi_row=k1, hi_color=V_EDGE, hi_fill=V_FILL,
                         title=f"V1 V 行[{k1}:{k1+Bc}]",
                         row_block_size=Bc, col_block_size=None,
                         show_values=_show_vals)
        draw_rect_matrix(ax_v1_o, v1_O_hist[frame_idx],
                         hi_row=q1, hi_color=O_EDGE, hi_fill=O_FILL,
                         title=f"V1 O 行[{q1}:{q1+Br}]",
                         row_block_size=Br, col_block_size=None,
                         touched=v1_touch_hist[frame_idx],
                         show_values=_show_vals)

        q1_str = format_loop_values(q1, all_q_vals)
        k1_str = format_loop_values(k1, all_k_vals)
        rescale_v1 = (frame_idx + 1) * Br
        v1_lines = [
            f"V1 Step {frame_idx + 1}/{NUM_STEPS}",
            f"外: k_start in {k1_str}  内: q_start in {q1_str}",
            f"S = Q[{q1}:{q1+Br}] @ K^T[:, {k1}:{k1+Bc}]",
            f"HBM O读+rescale+写: {rescale_v1}行 (×d={rescale_v1*d}次非矩阵乘)",
            f"并行: batch×heads (Q块在TB内串行)",
        ]
        ax_loop_v1.text(0.5, 0.95, "\n".join(v1_lines),
                        transform=ax_loop_v1.transAxes, fontsize=10,
                        verticalalignment="top", horizontalalignment="center",
                        linespacing=1.5,
                        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF3F3",
                                  edgecolor="#E53935", linewidth=1.5))

        # --- V2 ---
        q2, k2 = v2_steps[frame_idx]
        for ax in [ax_v2_q, ax_v2_kt, ax_v2_v, ax_v2_o]:
            ax.clear()
        ax_loop_v2.clear()
        ax_loop_v2.axis("off")

        draw_rect_matrix(ax_v2_q, Q,
                         hi_row=q2, hi_color=Q_EDGE, hi_fill=Q_FILL,
                         title=f"V2 Q 行[{q2}:{q2+Br}] [TB0|TB1|TB2|TB3并行]",
                         row_block_size=Br, col_block_size=None,
                         show_values=_show_vals)
        # TB 标签
        for tb_idx in range(NUM_Q_BLOCKS):
            y_c = tb_idx * Br + Br / 2 - 0.5
            active = (tb_idx == q2 // Br)
            ax_v2_q.text(-2.0, y_c, f"TB{tb_idx}",
                        ha="center", va="center", fontsize=6,
                        color="#2E7D32" if active else "#999999",
                        fontweight="bold" if active else "normal", zorder=10)
        ax_v2_q.set_xlim(-3.0, d - 0.5)
        draw_rect_matrix(ax_v2_kt, KT,
                         hi_col=k2, hi_color=K_EDGE, hi_fill=K_FILL,
                         title=f"V2 K^T 列[{k2}:{k2+Bc}]",
                         row_block_size=None, col_block_size=Bc,
                         show_values=_show_vals)
        draw_rect_matrix(ax_v2_v, V,
                         hi_row=k2, hi_color=V_EDGE, hi_fill=V_FILL,
                         title=f"V2 V 行[{k2}:{k2+Bc}]",
                         row_block_size=Bc, col_block_size=None,
                         show_values=_show_vals)
        draw_rect_matrix(ax_v2_o, v2_O_hist[frame_idx],
                         hi_row=q2, hi_color=O_EDGE, hi_fill=O_FILL,
                         title=f"V2 O 行[{q2}:{q2+Br}]",
                         row_block_size=Br, col_block_size=None,
                         touched=v2_touch_hist[frame_idx],
                         show_values=_show_vals)

        q2_str = format_loop_values(q2, all_q_vals)
        k2_str = format_loop_values(k2, all_k_vals)
        rescale_v2 = (frame_idx + 1) * Br
        v2_lines = [
            f"V2 Step {frame_idx + 1}/{NUM_STEPS}",
            f"外: q_start in {q2_str}  内: k_start in {k2_str}",
            f"S = Q[{q2}:{q2+Br}] @ K^T[:, {k2}:{k2+Bc}]",
            f"SRAM local rescale: {rescale_v2}行 (不写HBM!)",
            f"并行: batch×heads×seqlen ({NUM_Q_BLOCKS}个TB并行)",
        ]
        ax_loop_v2.text(0.5, 0.95, "\n".join(v2_lines),
                        transform=ax_loop_v2.transAxes, fontsize=10,
                        verticalalignment="top", horizontalalignment="center",
                        linespacing=1.5,
                        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F3F8FF",
                                  edgecolor="#1E88E5", linewidth=1.5))

        return []

    anim = FuncAnimation(fig, update, frames=NUM_STEPS, interval=600, blit=False)

    out_path = os.path.join(os.path.dirname(__file__), "fa_comparison.gif")
    writer = matplotlib.animation.PillowWriter(fps=2)
    anim.save(out_path, writer=writer, dpi=60)
    plt.close(fig)
    print(f"[comp]  Saved: {out_path}")


# ============================================================================
# 单步详解图
# ============================================================================
def create_step_detail(version="v1", step_idx=0):
    """为指定版本的指定步骤生成放大详解图。

    Parameters
    ----------
    version : "v1" | "v2"
    step_idx : int
    """
    steps_data = v1_steps if version == "v1" else v2_steps
    O_hist = v1_O_hist if version == "v1" else v2_O_hist
    touch_hist = v1_touch_hist if version == "v1" else v2_touch_hist

    if version == "v1":
        k_start, q_start = steps_data[step_idx]
    else:
        q_start, k_start = steps_data[step_idx]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    draw_rect_matrix(
        axes[0, 0], Q,
        hi_row=q_start, hi_color=Q_EDGE, hi_fill=Q_FILL,
        title=f"Q — 读取 Q[{q_start}:{q_start+Br}, :]  (shape: {Br}×{d})",
        row_block_size=Br, col_block_size=None,
    )

    draw_rect_matrix(
        axes[0, 1], KT,
        hi_col=k_start, hi_color=K_EDGE, hi_fill=K_FILL,
        title=f"K^T — 读取 K^T[:, {k_start}:{k_start+Bc}]  (shape: {d}×{Bc})",
        row_block_size=None, col_block_size=Bc,
    )

    draw_rect_matrix(
        axes[1, 0], V,
        hi_row=k_start, hi_color=V_EDGE, hi_fill=V_FILL,
        title=f"V — 读取 V[{k_start}:{k_start+Bc}, :]  (shape: {Bc}×{d})",
        row_block_size=Bc, col_block_size=None,
    )

    draw_rect_matrix(
        axes[1, 1], O_hist[step_idx],
        hi_row=q_start, hi_color=O_EDGE, hi_fill=O_FILL,
        title=f"O — 累加到 O[{q_start}:{q_start+Br}, :]  (shape: {Br}×{d})",
        row_block_size=Br, col_block_size=None,
        touched=touch_hist[step_idx],
    )

    loop_desc = f"外: K[{k_start}:{k_start+Bc}] → 内: Q[{q_start}:{q_start+Br}]" \
                if version == "v1" else \
                f"外: Q[{q_start}:{q_start+Br}] → 内: K[{k_start}:{k_start+Bc}]"

    fig.suptitle(
        f"Flash Attention {version.upper()} — Step {step_idx + 1}/{NUM_STEPS}\n"
        f"{loop_desc}\n"
        f"O[{q_start}:{q_start+Br}] += softmax(Q[{q_start}:{q_start+Br}] @ "
        f"K[{k_start}:{k_start+Bc}].T) @ V[{k_start}:{k_start+Bc}]",
        fontsize=14, fontweight="bold", y=1.02,
    )

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__),
                            f"fa_{version}_step{step_idx:03d}.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[step]  Saved: {out_path}")


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Flash Attention V1 vs V2 — 计算顺序可视化")
    print(f"  N={N}, d={d}, Br={Br}, Bc={Bc}")
    print(f"  Q blocks={NUM_Q_BLOCKS}, KV blocks={NUM_KV_BLOCKS}")
    print(f"  Steps per version={NUM_STEPS}")
    print("=" * 60)
    print()

    # 1) 静态计算顺序图
    print("[1/5] Generating static order diagrams...")
    create_static_order_figure("v1")
    create_static_order_figure("v2")
    print()

    # 2) V1 动画
    print("[2/5] Generating V1 animation...")
    create_animation("v1")
    gc.collect(); plt.close("all")
    print()

    # 3) V2 动画
    print("[3/5] Generating V2 animation...")
    create_animation("v2")
    gc.collect(); plt.close("all")
    print()

    # 4) 对比合成 GIF
    print("[4/5] Generating comparison GIF...")
    create_comparison_gif()
    gc.collect(); plt.close("all")
    print()

    # 5) 关键步骤详解图
    print("[5/5] Generating step detail snapshots...")
    for ver in ["v1", "v2"]:
        for idx in [0, NUM_STEPS // 2, NUM_STEPS - 1]:
            create_step_detail(ver, idx)
    print()

    print("=" * 60)
    print("  [Done] Generated files:")
    print("  Static diagrams:")
    print("    - fa_v1_order.png")
    print("    - fa_v2_order.png")
    print("  Animations:")
    print("    - fa_v1_animation.gif  (16 frames, ~8s)")
    print("    - fa_v2_animation.gif  (16 frames, ~8s)")
    print("    - fa_comparison.gif    (16 frames, ~8s, side-by-side)")
    print("  Step details:")
    print(f"    - fa_v1_step000.png, fa_v1_step{NUM_STEPS//2:03d}.png, "
          f"fa_v1_step{NUM_STEPS-1:03d}.png")
    print(f"    - fa_v2_step000.png, fa_v2_step{NUM_STEPS//2:03d}.png, "
          f"fa_v2_step{NUM_STEPS-1:03d}.png")
    print("=" * 60)
