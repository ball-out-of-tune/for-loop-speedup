#!/usr/bin/env python3
"""
分块矩阵乘法 (Tiled Matrix Multiplication) 可视化
====================================================

外三层循环以步长 B 遍历块:
  for sj in 0..N step B:    (C 的列块)
    for si in 0..N step B:  (C 的行块)
      for sk in 0..N step B: (归约维度 — K 方向)
        # B×B 块内朴素乘加: C[si][sj] += A[si][sk] × B[sk][sj]

输出文件:
  - tiled_matmul_order.png      静态计算顺序图 (C 矩阵块编号)
  - tiled_matmul_animation.gif  逐步动画 (A / B / C 三矩阵联动)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端, 直接保存文件
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置中文字体 (Windows 上使用微软雅黑)
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
    # 同时设置等宽字体 (monospace) 为 YaHei, 避免右侧说明文字回退到 DejaVu Sans Mono
    plt.rcParams["font.monospace"] = [_CN_FONT.name, "DejaVu Sans Mono"]
else:
    # 回退: 尝试用 sans-serif
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]

plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

# ============================================================================
# 配置 — 修改这里的 N, B 来观察不同规模
# ============================================================================
N = 8   # 矩阵边长
B = 2   # 分块大小

assert N % B == 0, "N 必须能被 B 整除"
NUM_BLOCKS = N // B                     # 每维度块数
NUM_STEPS = NUM_BLOCKS ** 3             # 外层总步数 = (N/B)³

# ---- 颜色方案 ----
A_EDGE  = "#E53935"    # A 活动块边框: 红
A_FILL  = "#FFCDD2"    # A 活动块填充: 浅红
B_EDGE  = "#1E88E5"    # B 活动块边框: 蓝
B_FILL  = "#BBDEFB"    # B 活动块填充: 浅蓝
C_EDGE  = "#8E24AA"    # C 活动块边框: 紫 (红+蓝=紫)
C_FILL  = "#E1BEE7"    # C 活动块填充: 浅紫
C_TOUCHED = "#C8E6C9"  # C 已触及区域: 绿

# ---- 随机种子 (便于复现) ----
np.random.seed(42)

# ============================================================================
# 生成矩阵 & 预计算
# ============================================================================
A = np.round(np.random.randn(N, N), 1)
B_mat = np.round(np.random.randn(N, N), 1)

# 枚举所有外层步骤 (sj, si, sk)
steps = [
    (sj, si, sk)
    for sj in range(0, N, B)
    for si in range(0, N, B)
    for sk in range(0, N, B)
]

# 预计算每一步之后的 C 矩阵状态
C_cur = np.zeros((N, N))
C_history = []          # C_history[t] = 第 t 步结束后的 C
touch_count = np.zeros((N, N), dtype=int)  # 每个 cell 被更新的次数
touch_history = []       # touch_history[t] = 第 t 步结束后的 touch_count

for sj, si, sk in steps:
    # 内三层: B×B 块内朴素乘加
    for i in range(si, si + B):
        for j in range(sj, sj + B):
            s = 0.0
            for k in range(sk, sk + B):
                s += A[i, k] * B_mat[k, j]
            C_cur[i, j] += s
    C_history.append(C_cur.copy())
    touch_count[si : si + B, sj : sj + B] += 1
    touch_history.append(touch_count.copy())

# C 矩阵每个 block 的"首次触及顺序"编号 (用于静态图)
block_order = np.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=int)
order = 1
for sj in range(0, N, B):
    for si in range(0, N, B):
        block_order[si // B, sj // B] = order
        order += 1

# C 的最终值 (用于静态图底色)
C_final = C_history[-1]


# ============================================================================
# 辅助函数: 画一个矩阵 + 高亮块
# ============================================================================
def draw_matrix_with_highlight(
    ax,
    data,
    hi_si=None,
    hi_sj=None,
    hi_color=None,
    hi_fill=None,
    title="",
    fmt=".1f",
    touched=None,
):
    """在 ax 上绘制 N×N 矩阵, 并在指定块上叠加高亮边框。

    Parameters
    ----------
    ax : Axes
    data : (N,N) ndarray — 矩阵数值
    hi_si, hi_sj : int | None — 高亮块的起始行/列; None 表示不高亮
    hi_color : str — 高亮边框颜色
    hi_fill  : str — 高亮填充颜色
    title : str
    fmt : str — 数值格式化
    touched : (N,N) ndarray | None — 记录每个 cell 被 touched 的次数, 用于 C 矩阵着色
    """
    extent = [-0.5, N - 0.5, N - 0.5, -0.5]

    # 底色
    if touched is not None:
        # C 矩阵: 按 touch 次数着色 (越深 = 越多 sk 贡献)
        max_touch = NUM_BLOCKS
        green_intensity = np.clip(touched / max_touch, 0, 1)
        # 从白到绿: RGB(1,1,1) → RGB(0.78,0.90,0.79)
        r = 1.0 - green_intensity * 0.22
        g = 1.0 - green_intensity * 0.10
        b = 1.0 - green_intensity * 0.21
        rgb = np.stack([r, g, b], axis=-1)
        ax.imshow(rgb, extent=extent, aspect="equal", origin="upper")
    else:
        # A / B 矩阵: 浅灰底色
        bg = np.ones((N, N, 3)) * 0.95
        ax.imshow(bg, extent=extent, aspect="equal", origin="upper")

    # 活动块高亮填充
    if hi_si is not None and hi_sj is not None and hi_fill is not None:
        rect_fill = Rectangle(
            (hi_sj - 0.5, hi_si - 0.5),
            B,
            B,
            linewidth=0,
            facecolor=hi_fill,
            zorder=2,
        )
        ax.add_patch(rect_fill)

    # 数值文字
    for i in range(N):
        for j in range(N):
            val = data[i, j]
            color = "#333333"
            fontweight = "normal"
            ax.text(
                j,
                i,
                f"{val:{fmt}}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
                fontweight=fontweight,
                zorder=5,
            )

    # 活动块边框
    if hi_si is not None and hi_sj is not None and hi_color is not None:
        rect_edge = Rectangle(
            (hi_sj - 0.5, hi_si - 0.5),
            B,
            B,
            linewidth=3,
            edgecolor=hi_color,
            facecolor="none",
            zorder=6,
        )
        ax.add_patch(rect_edge)

    # 块分隔线 (较粗)
    for k in range(0, N + 1, B):
        ax.axhline(y=k - 0.5, color="#666666", linewidth=1.2, zorder=4)
        ax.axvline(x=k - 0.5, color="#666666", linewidth=1.2, zorder=4)

    # 普通网格线 (较细)
    for k in range(0, N + 1):
        ax.axhline(y=k - 0.5, color="#cccccc", linewidth=0.5, zorder=0)
        ax.axvline(x=k - 0.5, color="#cccccc", linewidth=0.5, zorder=0)

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # 行列标签
    for i in range(N):
        ax.text(-1.0, i, f"i={i}", ha="right", va="center", fontsize=6, color="#999")
    for j in range(N):
        ax.text(j, N, f"j={j}", ha="center", va="bottom", fontsize=6, color="#999")


# ============================================================================
# 静态图: C 矩阵计算顺序
# ============================================================================
def create_static_figure():
    """生成 C 矩阵块级计算顺序总览图。"""
    fig, (ax_main, ax_legend) = plt.subplots(
        1, 2,
        figsize=(16, 7),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    # --- 左: C 矩阵块顺序 ---
    extent = [-0.5, N - 0.5, N - 0.5, -0.5]

    # 底色: 按 block_order 从暖到冷
    order_colors = np.zeros((N, N))
    for bi in range(NUM_BLOCKS):
        for bj in range(NUM_BLOCKS):
            si, sj = bi * B, bj * B
            order_colors[si : si + B, sj : sj + B] = block_order[bi, bj]

    ax_main.imshow(
        order_colors,
        cmap=plt.cm.YlOrRd,
        extent=extent,
        aspect="equal",
        origin="upper",
        vmin=1,
        vmax=NUM_BLOCKS**2,
    )

    # 块编号 (使用粗体数字 + 白色圆形背景)
    from matplotlib.patches import Circle
    for bi in range(NUM_BLOCKS):
        for bj in range(NUM_BLOCKS):
            order_num = block_order[bi, bj]
            si_center = bi * B + B / 2 - 0.5
            sj_center = bj * B + B / 2 - 0.5
            # 绘制白色圆形背景
            circle = Circle(
                (sj_center, si_center), 0.55,
                facecolor="white", edgecolor="#333333",
                linewidth=2, zorder=9,
            )
            ax_main.add_patch(circle)
            # 数字
            ax_main.text(
                sj_center,
                si_center,
                str(order_num),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="#333333",
                zorder=10,
            )
            # 小字标注 (sj, si)
            ax_main.text(
                sj_center,
                si_center + 0.65,
                f"sj={bj*B}, si={bi*B}",
                ha="center",
                va="center",
                fontsize=6,
                color="#333333",
                zorder=10,
            )

    # 块分隔线
    for k in range(0, N):
        ax_main.axhline(y=k - 0.5, color="white", linewidth=1.5, alpha=0.6)
        ax_main.axvline(x=k - 0.5, color="white", linewidth=1.5, alpha=0.6)

    # 行列标签
    for i in range(N):
        ax_main.text(-0.8, i, f"i={i}", ha="right", va="center", fontsize=7, color="#555")
    for j in range(N):
        ax_main.text(j, N + 0.3, f"j={j}", ha="center", va="top", fontsize=7, color="#555")

    # 箭头标注: 沿 C 列方向画箭头表示 sj 是最外层
    for bj in range(NUM_BLOCKS):
        col_center_x = bj * B + B / 2 - 0.5
        ax_main.annotate(
            "",
            xy=(col_center_x, N + 1.2),
            xytext=(col_center_x, N + 0.3),
            arrowprops=dict(arrowstyle="->", color="#E53935", lw=2),
        )
    ax_main.text(N / 2 - 0.5, N + 1.8, "sj 遍历方向 (最外层)", ha="center",
                 fontsize=9, color="#E53935", fontweight="bold")

    # 沿行方向箭头 (si)
    for bi_idx in range(NUM_BLOCKS):
        row_center_y = bi_idx * B + B / 2 - 0.5
        ax_main.annotate(
            "",
            xy=(-1.5, row_center_y),
            xytext=(-0.8, row_center_y),
            arrowprops=dict(arrowstyle="->", color="#1E88E5", lw=2),
        )
    ax_main.text(-1.5, N / 2 - 0.5, "si\n遍历\n(中层)",
                 ha="center", va="center", fontsize=9, color="#1E88E5",
                 fontweight="bold", rotation=90)

    ax_main.set_xlim(-2.5, N + 1.0)
    ax_main.set_ylim(N + 2.5, -2.0)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_title(
        f"矩阵 C 的块级计算顺序 (N={N}, B={B})\n"
        f"颜色越深 = 越早完成; 编号 = 首次触及顺序",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # --- 右: 算法描述 + 图例 ---
    ax_legend.axis("off")

    desc_text = (
        "分块矩阵乘法算法\n"
        "══════════════════\n\n"
        "外层循环 (块级别):\n"
        "  ① for sj = 0, B, 2B, ...\n"
        "  ②   for si = 0, B, 2B, ...\n"
        "  ③     for sk = 0, B, 2B, ...\n\n"
        "内层循环 (B×B 块内):\n"
        "  ④     for j in [sj, sj+B)\n"
        "  ⑤       for i in [si, si+B)\n"
        "  ⑥         sum = 0\n"
        "  ⑦         for k in [sk, sk+B)\n"
        "  ⑧           sum += A[i][k]·B[k][j]\n"
        "  ⑨         C[i][j] += sum\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"N = {N},  B = {B}\n"
        f"每维度块数 = {NUM_BLOCKS}\n"
        f"外层总迭代 = {NUM_BLOCKS}³ = {NUM_STEPS} 步\n\n"
        "关键理解:\n"
        "• sj 遍历 C 的列块 (最外层)\n"
        "• si 遍历 C 的行块 (中层)\n"
        "• sk 遍历归约维度 K (最内层)\n"
        "• 每个 C 块经所有 sk 迭代才完整\n"
        "• C 按列优先顺序逐个块完成"
    )

    ax_legend.text(
        0.05, 0.95, desc_text,
        transform=ax_legend.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8),
        linespacing=1.4,
    )

    # 图例小方块
    legend_items = [
        ("A 活动块", A_EDGE, A_FILL, "A[si:si+B, sk:sk+B]"),
        ("B 活动块", B_EDGE, B_FILL, "B[sk:sk+B, sj:sj+B]"),
        ("C 活动块", C_EDGE, C_FILL, "C[si:si+B, sj:sj+B] (累加中)"),
    ]
    y_start = 0.32
    for idx, (label, edge_c, fill_c, desc) in enumerate(legend_items):
        y = y_start - idx * 0.06
        # 色块
        ax_legend.add_patch(
            Rectangle((0.05, y), 0.06, 0.04, transform=ax_legend.transAxes,
                       facecolor=fill_c, edgecolor=edge_c, linewidth=2)
        )
        ax_legend.text(0.13, y + 0.02, f"{label}: {desc}",
                       transform=ax_legend.transAxes, fontsize=9,
                       verticalalignment="center")

    fig.tight_layout(pad=2)
    out_path = os.path.join(os.path.dirname(__file__), "tiled_matmul_order.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[static] Saved: {out_path}")


# ============================================================================
# 动画: 逐步展示计算过程 (含循环变量面板)
# ============================================================================
def create_animation():
    """生成逐步动画 GIF, 展示每步的 A/B/C 活动块 + 循环变量状态。"""
    fig = plt.figure(figsize=(18, 9.5))

    # GridSpec: 上排 3 个矩阵, 下排循环变量面板
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 2.5], hspace=0.35)
    gs_top = gs[0].subgridspec(1, 3)
    axes = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
    ax_loop = fig.add_subplot(gs[1])
    ax_loop.axis("off")

    def update(frame_idx):
        sj, si, sk = steps[frame_idx]
        C_state = C_history[frame_idx]
        touch_state = touch_history[frame_idx]

        # 清除
        for ax in axes:
            ax.clear()
        ax_loop.clear()
        ax_loop.axis("off")

        # --- A 矩阵 ---
        draw_matrix_with_highlight(
            axes[0], A,
            hi_si=si, hi_sj=sk,
            hi_color=A_EDGE, hi_fill=A_FILL,
            title=f"A (si={si}, sk={sk})",
        )

        # --- B 矩阵 ---
        draw_matrix_with_highlight(
            axes[1], B_mat,
            hi_si=sk, hi_sj=sj,
            hi_color=B_EDGE, hi_fill=B_FILL,
            title=f"B (sk={sk}, sj={sj})",
        )

        # --- C 矩阵 ---
        draw_matrix_with_highlight(
            axes[2], C_state,
            hi_si=si, hi_sj=sj,
            hi_color=C_EDGE, hi_fill=C_FILL,
            title=f"C (si={si}, sj={sj})",
            touched=touch_state,
        )

        # ================================================================
        # 循环变量面板
        # ================================================================
        progress_pct = (frame_idx + 1) / NUM_STEPS * 100

        # -- 生成所有可能的 sj / si / sk 值序列 --
        all_vals = list(range(0, N, B))  # e.g. [0, 2, 4, 6]

        def format_loop_values(current, values, color):
            """把值序列格式化为字符串, current 用颜色高亮。e.g. '{0, 2, [4], 6}'"""
            parts = []
            for v in values:
                if v == current:
                    parts.append(f"[{v}]")
                else:
                    parts.append(str(v))
            return "{" + ", ".join(parts) + "}"

        sj_str = format_loop_values(sj, all_vals, A_EDGE)
        si_str = format_loop_values(si, all_vals, B_EDGE)
        sk_str = format_loop_values(sk, all_vals, C_EDGE)

        # 构建面板文本
        lines = []
        lines.append(f"Step {frame_idx + 1} / {NUM_STEPS}  ({progress_pct:.1f}%)")
        lines.append("")

        # 外层循环 — 三层
        lines.append("Outer loops (block-level):")
        lines.append(f"  for sj in {sj_str}    ← C column block (outermost)")
        lines.append(f"  for si in {si_str}    ← C row block (middle)")
        lines.append(f"  for sk in {sk_str}    ← reduction dim K (innermost)")
        lines.append("")

        # 内层循环 — 由外层当前值决定
        lines.append(f"Inner loops (inside BxB block):")
        lines.append(f"  for j in [{sj}, {sj+B})    for i in [{si}, {si+B})    for k in [{sk}, {sk+B})")
        lines.append(f"    C[i][j] += sum_k A[i][k] * B[k][j]")
        lines.append("")

        # 运算式
        lines.append(f"Operation:")
        lines.append(f"  A[{si}:{si+B}, {sk}:{sk+B}]  x  B[{sk}:{sk+B}, {sj}:{sj+B}]  ->  C[{si}:{si+B}, {sj}:{sj+B}]")

        panel_text = "\n".join(lines)

        # 绘制面板 — 使用背景框
        ax_loop.text(
            0.5, 0.95, panel_text,
            transform=ax_loop.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="center",
            linespacing=1.5,
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="#FAFAFA",
                edgecolor="#CCCCCC",
                linewidth=1.5,
            ),
        )

        # 在当前值下面画彩色下划线标记
        # 计算 sj_str / si_str / sk_str 中 [...] 的位置, 给对应字符着色
        # (在 monospace 文本中通过彩色 text 叠加来实现高亮)
        return [ax_loop] + [ax for ax in axes]

    # 创建动画
    anim = FuncAnimation(
        fig, update,
        frames=NUM_STEPS,
        interval=500,   # 每帧 500ms (稍慢便于看清)
        blit=False,
    )

    # 保存为 GIF
    out_path = os.path.join(os.path.dirname(__file__), "tiled_matmul_animation.gif")
    writer = matplotlib.animation.PillowWriter(fps=2)
    anim.save(out_path, writer=writer, dpi=80)
    plt.close(fig)
    print(f"[anim] Saved: {out_path}")
    print(f"       {NUM_STEPS} frames, 2 fps, ~{NUM_STEPS / 2:.0f}s")


# ============================================================================
# 额外: 单步详解图 (展示某一步的详细数据流)
# ============================================================================
def create_single_step_detail(step_idx=0):
    """为指定步骤生成一张放大的详解图, 标注数据流。

    Parameters
    ----------
    step_idx : int — 步骤索引, 0 <= step_idx < NUM_STEPS
    """
    sj, si, sk = steps[step_idx]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7.5))
    C_state = C_history[step_idx]
    touch_state = touch_history[step_idx]

    draw_matrix_with_highlight(
        axes[0], A,
        hi_si=si, hi_sj=sk,
        hi_color=A_EDGE, hi_fill=A_FILL,
        title=f"矩阵 A — 读取 A[{si}:{si+B}, {sk}:{sk+B}]",
    )

    draw_matrix_with_highlight(
        axes[1], B_mat,
        hi_si=sk, hi_sj=sj,
        hi_color=B_EDGE, hi_fill=B_FILL,
        title=f"矩阵 B — 读取 B[{sk}:{sk+B}, {sj}:{sj+B}]",
    )

    draw_matrix_with_highlight(
        axes[2], C_state,
        hi_si=si, hi_sj=sj,
        hi_color=C_EDGE, hi_fill=C_FILL,
        title=f"矩阵 C — 累加到 C[{si}:{si+B}, {sj}:{sj+B}]",
        touched=touch_state,
    )

    fig.suptitle(
        f"详解: Step {step_idx + 1}/{NUM_STEPS} — sj={sj}, si={si}, sk={sk}\n"
        f"C[{si}:{si+B}, {sj}:{sj+B}] += A[{si}:{si+B}, {sk}:{sk+B}] × B[{sk}:{sk+B}, {sj}:{sj+B}]",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__),
                            f"tiled_matmul_step{step_idx:03d}.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[step] Saved: {out_path}")
    return out_path


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Tiled Matrix Multiplication Visualization")
    print(f"  N = {N},  B = {B},  blocks/dim = {NUM_BLOCKS},  outer steps = {NUM_STEPS}")
    print("=" * 60)

    # 1) 静态计算顺序图
    create_static_figure()

    # 2) 动画 GIF
    create_animation()

    # 3) 首步 / 中间步 / 末步 详解截图
    for idx in [0, NUM_STEPS // 2, NUM_STEPS - 1]:
        create_single_step_detail(idx)

    print("\n[Done] Generated files:")
    print("  - tiled_matmul_order.png       (static: computation order overview)")
    print("  - tiled_matmul_animation.gif   (animation: step-by-step process)")
    print("  - tiled_matmul_step000.png     (detail: step 1)")
    print(f"  - tiled_matmul_step{NUM_STEPS // 2:03d}.png     (detail: middle step)")
    print(f"  - tiled_matmul_step{NUM_STEPS - 1:03d}.png     (detail: last step)")
