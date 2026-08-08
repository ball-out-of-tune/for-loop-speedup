"""
Visualize irregular GEMM: M=44, N=44, K=44, TILE=16
Shows how the edge block handles out-of-bounds with zero-padding.

The key ratio: M/N/K are NOT multiples of TILE, so the last block
wraps partially out of bounds — exactly the scenario boundary checks handle.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Small sizes to make the irregular shape visible
M, N, K = 44, 44, 44
TILE = 16
GRID_M = (M + TILE - 1) // TILE  # 3
GRID_N = (N + TILE - 1) // TILE  # 3
NUM_K_TILES = (K + TILE - 1) // TILE  # 3

# Generate fake data for visualization
np.random.seed(42)
A = np.random.randn(M, K).astype(np.float32) * 0.5
B = np.random.randn(K, N).astype(np.float32) * 0.5
C_cpu = A @ B  # reference

# Focus on the LAST block: blockIdx.y=2, blockIdx.x=2
# This block covers rows 32-47 (valid: 32-43), cols 32-47 (valid: 32-43)
BLOCK_Y = 2
BLOCK_X = 2
ROW_START = BLOCK_Y * TILE  # 32
COL_START = BLOCK_X * TILE  # 32

# Valid rows and cols within this block
valid_rows_in_block = [r for r in range(TILE) if ROW_START + r < M]  # 0-11
valid_cols_in_block = [c for c in range(TILE) if COL_START + c < N]  # 0-11

print(f"M={M}, N={N}, K={K}, TILE={TILE}")
print(f"Grid: {GRID_M}x{GRID_N} blocks")
print(f"Focus block: ({BLOCK_Y}, {BLOCK_X})")
print(f"  Global rows: {ROW_START}-{min(ROW_START+TILE-1, M-1)} (valid up to {M-1})")
print(f"  Global cols: {COL_START}-{min(COL_START+TILE-1, N-1)} (valid up to {N-1})")
print(f"  Valid tile rows: {valid_rows_in_block[0]}-{valid_rows_in_block[-1]}")
print(f"  Valid tile cols: {valid_cols_in_block[0]}-{valid_cols_in_block[-1]}")
oob_row_start = valid_rows_in_block[-1] + 1 if valid_rows_in_block else 0
oob_col_start = valid_cols_in_block[-1] + 1 if valid_cols_in_block else 0
print(f"  OOB rows: {oob_row_start}-{TILE-1}")
print(f"  OOB cols: {oob_col_start}-{TILE-1}")

# Set up the figure
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Irregular GEMM: Boundary Check Visualization\n'
             f'M={M}, N={N}, K={K}, TILE={TILE}  |  '
             f'Focus: Edge Block ({BLOCK_Y},{BLOCK_X}) covering rows {ROW_START}-{ROW_START+TILE-1}, cols {COL_START}-{COL_START+TILE-1}',
             fontsize=12, fontweight='bold')

# Create subplot grid: 2 rows, 3 cols
# Row 1: C matrix overview, A shared mem tile, B shared mem tile
# Row 2: K-loop animation of dot product
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1],
                       hspace=0.35, wspace=0.35,
                       left=0.05, right=0.95, top=0.90, bottom=0.08)

ax_c = fig.add_subplot(gs[0, 0])    # C matrix overview
ax_as = fig.add_subplot(gs[0, 1])   # A shared memory tile
ax_bs = fig.add_subplot(gs[0, 2])   # B shared memory tile
ax_dot = fig.add_subplot(gs[1, :])  # Dot product accumulation

# ---- Static plots: C matrix overview ----
def draw_c_overview(ax, highlight_block=True):
    ax.clear()
    # Draw the C matrix as a grid
    cmap = plt.cm.RdYlGn
    for i in range(M):
        for j in range(N):
            in_focus_block = (ROW_START <= i < ROW_START + TILE and
                             COL_START <= j < COL_START + TILE)
            if in_focus_block:
                # Valid cells in focus block
                if i < M and j < N:
                    color = '#4CAF50'  # green: valid
                else:
                    color = '#FF5252'  # red: OOB
            else:
                if i < M and j < N:
                    color = '#BBDEFB'  # light blue: other valid
                else:
                    color = '#EEEEEE'  # grey: OOB other

            rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                           facecolor=color, edgecolor='white', linewidth=0.3)
            ax.add_patch(rect)

    # Draw block boundaries
    for bi in range(GRID_M + 1):
        ax.axhline(y=bi * TILE - 0.5, color='#1565C0', linewidth=1.5, linestyle='--', alpha=0.7)
    for bj in range(GRID_N + 1):
        ax.axvline(x=bj * TILE - 0.5, color='#1565C0', linewidth=1.5, linestyle='--', alpha=0.7)

    # Highlight focus block
    if highlight_block:
        rect = Rectangle((COL_START - 0.5, ROW_START - 0.5), TILE, TILE,
                        facecolor='none', edgecolor='#FF6F00', linewidth=3, zorder=10)
        ax.add_patch(rect)

    # Shade the OOB region (beyond M x N)
    oob_rect = Rectangle((N - 0.5, M - 0.5),
                         GRID_N * TILE - N, GRID_M * TILE - M,
                         facecolor='red', alpha=0.15, edgecolor='red',
                         linewidth=1, linestyle=':', zorder=5)
    ax.add_patch(oob_rect)

    ax.set_xlim(-1, GRID_N * TILE)
    ax.set_ylim(GRID_M * TILE - 1, -1)
    ax.set_aspect('equal')
    ax.set_title(f'C Matrix ({M}×{N})\nGrid: {GRID_M}×{GRID_N} blocks of {TILE}×{TILE}',
                fontsize=10, fontweight='bold')
    ax.set_xlabel('N (columns)')
    ax.set_ylabel('M (rows)')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor='#BBDEFB', edgecolor='white', label='Valid (other blocks)'),
        mpatches.Patch(facecolor='#4CAF50', edgecolor='white', label='Valid (focus block)'),
        mpatches.Patch(facecolor='#FF5252', edgecolor='white', label='OOB (zero-padded)'),
        mpatches.Patch(facecolor='red', alpha=0.15, edgecolor='red', label='Beyond M×N'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
             bbox_to_anchor=(1.35, 1.0))

# ---- Static plots: A shared memory tile ----
def draw_as_tile(ax, k_tile_idx):
    """Draw the A shared memory tile for the focus block at k_tile_idx"""
    ax.clear()
    k_start = k_tile_idx * TILE

    for ti in range(TILE):  # tile row
        for tj in range(TILE):  # tile col (K dimension in tile)
            global_row = ROW_START + ti
            global_k = k_start + tj

            if global_row < M and global_k < K:
                val = A[global_row, global_k]
                # Color by value
                if val > 0:
                    intensity = min(1.0, abs(val))
                    color = plt.cm.Reds(intensity)
                else:
                    intensity = min(1.0, abs(val))
                    color = plt.cm.Blues(intensity)
            else:
                color = '#FF5252'  # Red = OOB, will be zero-padded

            rect = Rectangle((tj - 0.5, ti - 0.5), 1, 1,
                           facecolor=color, edgecolor='white', linewidth=0.3)
            ax.add_patch(rect)

    # Draw boundary between valid and OOB rows
    valid_row_end = M - ROW_START
    if 0 < valid_row_end < TILE:
        ax.axhline(y=valid_row_end - 0.5, color='#FF6F00', linewidth=2.5, linestyle='-')

    # Draw boundary between valid and OOB K values
    valid_k_end = K - k_start
    if 0 < valid_k_end < TILE:
        ax.axvline(x=valid_k_end - 0.5, color='#FF6F00', linewidth=2.5, linestyle='-')

    ax.set_xlim(-1, TILE)
    ax.set_ylim(TILE - 1, -1)
    ax.set_aspect('equal')
    ax.set_title(f'A Shared Memory Tile\nk_tile={k_tile_idx} (K=[{k_start},{k_start+TILE-1}])\n'
                f'Rows {ROW_START}-{ROW_START+TILE-1} (valid to {M-1})',
                fontsize=9, fontweight='bold')
    ax.set_xlabel('K dimension in tile')
    ax.set_ylabel('Row in tile')

# ---- Static plots: B shared memory tile ----
def draw_bs_tile(ax, k_tile_idx):
    """Draw the B shared memory tile for the focus block at k_tile_idx"""
    ax.clear()
    k_start = k_tile_idx * TILE

    for ti in range(TILE):  # tile row (K dimension in tile)
        for tj in range(TILE):  # tile col (N dimension in tile)
            global_k = k_start + ti
            global_col = COL_START + tj

            if global_k < K and global_col < N:
                val = B[global_k, global_col]
                if val > 0:
                    intensity = min(1.0, abs(val))
                    color = plt.cm.Reds(intensity)
                else:
                    intensity = min(1.0, abs(val))
                    color = plt.cm.Blues(intensity)
            else:
                color = '#FF5252'

            rect = Rectangle((tj - 0.5, ti - 0.5), 1, 1,
                           facecolor=color, edgecolor='white', linewidth=0.3)
            ax.add_patch(rect)

    # Draw boundaries
    valid_k_end = K - k_start
    if 0 < valid_k_end < TILE:
        ax.axhline(y=valid_k_end - 0.5, color='#FF6F00', linewidth=2.5, linestyle='-')

    valid_col_end = N - COL_START
    if 0 < valid_col_end < TILE:
        ax.axvline(x=valid_col_end - 0.5, color='#FF6F00', linewidth=2.5, linestyle='-')

    ax.set_xlim(-1, TILE)
    ax.set_ylim(TILE - 1, -1)
    ax.set_aspect('equal')
    ax.set_title(f'B Shared Memory Tile\nk_tile={k_tile_idx} (K=[{k_start},{k_start+TILE-1}])\n'
                f'Cols {COL_START}-{COL_START+TILE-1} (valid to {N-1})',
                fontsize=9, fontweight='bold')
    ax.set_xlabel('N dimension in tile')
    ax.set_ylabel('K dimension in tile')

# ---- Dot product accumulation visualization ----
def draw_dot_product(ax, k_tile_idx, inner_k):
    """Show the dot product accumulation: which elements are being multiplied"""
    ax.clear()

    # Show a schematic of the dot product
    # For one specific output cell (say, thread computing row 0, col 0 within the tile)
    # s00 += As[ty][k] * Bs[k][tx]

    # We'll show the accumulation for 4 output positions (matching 2x2 register blocking)
    # For simplicity, pick thread (ty=2, tx=3) which computes:
    #   C[cur_row][cur_col], C[cur_row][cur_col+TILE/2],
    #   C[cur_row+TILE/2][cur_col], C[cur_row+TILE/2][cur_col+TILE/2]

    ty, tx = 2, 3
    global_row0 = ROW_START + ty
    global_row1 = ROW_START + ty + TILE // 2
    global_col0 = COL_START + tx
    global_col1 = COL_START + tx + TILE // 2

    k_total_start = k_tile_idx * TILE

    ax.text(0.5, 0.95, f'Dot Product Accumulation (k_tile={k_tile_idx}, inner_k={inner_k}/{TILE-1})\n'
            f'Thread ({ty},{tx}): computing 2×2 outputs via register blocking',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            ha='center', va='top')

    # Show As row values for this thread
    ax.text(0.02, 0.82, 'As tile (A values):', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')

    # Draw the two rows being accessed
    n_k_vals = min(8, TILE)  # Show first 8 K values for clarity
    x_positions = np.linspace(0.1, 0.55, n_k_vals)

    for ki in range(n_k_vals):
        global_k = k_total_start + ki

        # Row 0 value
        if global_row0 < M and global_k < K:
            a0_val = A[global_row0, global_k]
            a0_color = '#4CAF50'
            a0_text = f'{a0_val:+.2f}'
        else:
            a0_val = 0.0
            a0_color = '#FF5252'
            a0_text = '0 (OOB)'

        # Row 1 value
        if global_row1 < M and global_k < K:
            a1_val = A[global_row1, global_k]
            a1_color = '#4CAF50'
            a1_text = f'{a1_val:+.2f}'
        else:
            a1_val = 0.0
            a1_color = '#FF5252'
            a1_text = '0 (OOB)'

        # Highlight current k being processed
        is_current = (ki == inner_k % n_k_vals)
        edge_color = '#FF6F00' if is_current else 'none'
        edge_width = 3 if is_current else 1

        rect0 = FancyBboxPatch((x_positions[ki] - 0.03, 0.72), 0.06, 0.06,
                               boxstyle='round,pad=0.01', facecolor=a0_color,
                               edgecolor=edge_color, linewidth=edge_width,
                               transform=ax.transAxes)
        ax.add_patch(rect0)
        ax.text(x_positions[ki], 0.69, a0_text, transform=ax.transAxes,
               fontsize=5, ha='center', va='top', rotation=90)

        rect1 = FancyBboxPatch((x_positions[ki] - 0.03, 0.60), 0.06, 0.06,
                               boxstyle='round,pad=0.01', facecolor=a1_color,
                               edgecolor=edge_color, linewidth=edge_width,
                               transform=ax.transAxes)
        ax.add_patch(rect1)
        ax.text(x_positions[ki], 0.57, a1_text, transform=ax.transAxes,
               fontsize=5, ha='center', va='top', rotation=90)

    ax.text(0.32, 0.82, f'As[{ty}][k]  (row {global_row0})', transform=ax.transAxes, fontsize=8, va='top')
    ax.text(0.32, 0.70, f'As[{ty+TILE//2}][k]  (row {global_row1})', transform=ax.transAxes, fontsize=8, va='top')

    # Show Bs column values
    ax.text(0.02, 0.52, 'Bs tile (B values):', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')

    for ki in range(n_k_vals):
        global_k = k_total_start + ki

        if global_k < K and global_col0 < N:
            b0_val = B[global_k, global_col0]
            b0_color = '#4CAF50'
            b0_text = f'{b0_val:+.2f}'
        else:
            b0_val = 0.0
            b0_color = '#FF5252'
            b0_text = '0 (OOB)'

        if global_k < K and global_col1 < N:
            b1_val = B[global_k, global_col1]
            b1_color = '#4CAF50'
            b1_text = f'{b1_val:+.2f}'
        else:
            b1_val = 0.0
            b1_color = '#FF5252'
            b1_text = '0 (OOB)'

        is_current = (ki == inner_k % n_k_vals)
        edge_color = '#FF6F00' if is_current else 'none'
        edge_width = 3 if is_current else 1

        rect0 = FancyBboxPatch((x_positions[ki] - 0.03, 0.42), 0.06, 0.06,
                               boxstyle='round,pad=0.01', facecolor=b0_color,
                               edgecolor=edge_color, linewidth=edge_width,
                               transform=ax.transAxes)
        ax.add_patch(rect0)
        ax.text(x_positions[ki], 0.39, b0_text, transform=ax.transAxes,
               fontsize=5, ha='center', va='top', rotation=90)

        rect1 = FancyBboxPatch((x_positions[ki] - 0.03, 0.30), 0.06, 0.06,
                               boxstyle='round,pad=0.01', facecolor=b1_color,
                               edgecolor=edge_color, linewidth=edge_width,
                               transform=ax.transAxes)
        ax.add_patch(rect1)
        ax.text(x_positions[ki], 0.27, b1_text, transform=ax.transAxes,
               fontsize=5, ha='center', va='top', rotation=90)

    ax.text(0.32, 0.52, f'Bs[k][{tx}]  (col {global_col0})', transform=ax.transAxes, fontsize=8, va='top')
    ax.text(0.32, 0.40, f'Bs[k][{tx+TILE//2}]  (col {global_col1})', transform=ax.transAxes, fontsize=8, va='top')

    # Show accumulation
    ax.text(0.65, 0.82, 'Accumulating:', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')

    # Compute partial sums
    s00 = s01 = s10 = s11 = 0.0
    k_start = k_tile_idx * TILE
    for k in range(inner_k + 1):
        gk = k_start + k
        if global_row0 < M and gk < K:
            a0 = A[global_row0, gk]
        else:
            a0 = 0.0
        if global_row1 < M and gk < K:
            a1 = A[global_row1, gk]
        else:
            a1 = 0.0
        if gk < K and global_col0 < N:
            b0 = B[gk, global_col0]
        else:
            b0 = 0.0
        if gk < K and global_col1 < N:
            b1 = B[gk, global_col1]
        else:
            b1 = 0.0
        s00 += a0 * b0
        s01 += a0 * b1
        s10 += a1 * b0
        s11 += a1 * b1

    text_lines = [
        f's00 = Σ As[{ty}][k]·Bs[k][{tx}]',
        f'      C[{global_row0}][{global_col0}] = {s00:+.4f}',
        f'',
        f's01 = Σ As[{ty}][k]·Bs[k][{tx+TILE//2}]',
        f'      C[{global_row0}][{global_col1}] = {s01:+.4f}',
        f'',
        f's10 = Σ As[{ty+TILE//2}][k]·Bs[k][{tx}]',
        f'      C[{global_row1}][{global_col0}] = {s10:+.4f}',
        f'',
        f's11 = Σ As[{ty+TILE//2}][k]·Bs[k][{tx+TILE//2}]',
        f'      C[{global_row1}][{global_col1}] = {s11:+.4f}',
    ]

    # Map output names to whether they're OOB
    oob_map = {
        's00': (global_row0 >= M or global_col0 >= N),
        's01': (global_row0 >= M or global_col1 >= N),
        's10': (global_row1 >= M or global_col0 >= N),
        's11': (global_row1 >= M or global_col1 >= N),
    }

    for i, line in enumerate(text_lines):
        y = 0.72 - i * 0.035
        color = 'black'
        if line.startswith('s0'):
            color = '#1565C0'
            name = line.split(' ')[0]  # e.g. 's00'
            if oob_map.get(name, False):
                color = '#FF5252'
        elif 'C[' in line:
            # Line format: "      C[row][col] = value"
            for name, is_oob in oob_map.items():
                if name in line and is_oob:
                    color = '#FF5252'
                    line = line + ' ← SKIP (OOB)'
                    break
        ax.text(0.62, y, line, transform=ax.transAxes, fontsize=7.5,
               va='top', fontfamily='monospace', color=color)

    # Legend
    legend_text = ax.text(0.65, 0.20,
                         '[Green] = valid data (read from global mem)\n'
                         '[Red]   = OOB (zero-padded, no global mem read)\n'
                         '[Orange border] = current k being processed',
                         transform=ax.transAxes, fontsize=8, va='top')

    ax.axis('off')

# ---- Create animation frames ----
# We'll cycle through: overview → k_tile=0,1,2 with inner_k progression
frames = []

# Phase descriptions
phases = [
    "C Matrix Overview:\nEdge block covers rows 32-47 (valid: 32-43)\n& cols 32-47 (valid: 32-43)",
]

# For each k_tile, create frames showing the loading and compute
for k_tile in range(NUM_K_TILES):
    phases.append(f"k_tile={k_tile}: Load A tile\n(K=[{k_tile*TILE},{(k_tile+1)*TILE-1}])")
    phases.append(f"k_tile={k_tile}: Load B tile\n(K=[{k_tile*TILE},{(k_tile+1)*TILE-1}])")
    phases.append(f"k_tile={k_tile}: __syncthreads()\nBoth tiles ready")
    for inner_k in [0, 4, 8, TILE-1]:
        phases.append(f"k_tile={k_tile}, inner_k={inner_k}:\nDot product accumulation")
    phases.append(f"k_tile={k_tile}: __syncthreads()\nReady for next tile")

phases.append("Writeback to C:\nBoundary check before each write")

# Create frames
total_frames = len(phases)

def get_frame_data(frame_idx):
    """Return what to draw for each frame"""
    phase = phases[frame_idx]

    if frame_idx == 0:
        return 'overview', 0, 0

    # Map frame to k_tile and inner_k
    idx = frame_idx - 1  # remove overview

    # Each k_tile has: load_A(1) + load_B(1) + sync(1) + inner_k*4(4) + sync(1) = 8 frames
    frames_per_ktile = 8

    if idx >= NUM_K_TILES * frames_per_ktile:
        return 'writeback', 0, 0

    k_tile = idx // frames_per_ktile
    inner_idx = idx % frames_per_ktile

    if inner_idx == 0:
        return 'load_A', k_tile, 0
    elif inner_idx == 1:
        return 'load_B', k_tile, 0
    elif inner_idx == 2:
        return 'sync', k_tile, 0
    elif inner_idx in [3, 4, 5, 6]:
        inner_k_vals = [0, 4, 8, TILE-1]
        return 'compute', k_tile, inner_k_vals[inner_idx - 3]
    else:  # inner_idx == 7
        return 'sync2', k_tile, 0

# Pre-compute all frames
n_frames = total_frames

def update(frame_idx):
    """Update function for animation"""
    action, k_tile, inner_k = get_frame_data(frame_idx)
    phase_text = phases[frame_idx]

    # Clear all axes
    for ax in [ax_c, ax_as, ax_bs, ax_dot]:
        ax.clear()

    # Always draw C overview (with different highlights)
    if action == 'writeback':
        draw_c_overview(ax_c, highlight_block=True)
        # Highlight which cells get written
        for ti in range(TILE):
            for tj in range(TILE):
                gr = ROW_START + ti
                gc = COL_START + tj
                if gr < M and gc < N:
                    rect = Rectangle((gc - 0.35, gr - 0.35), 0.7, 0.7,
                                   facecolor='#FFD700', edgecolor='#FF6F00',
                                   linewidth=1.5, alpha=0.8, zorder=11)
                    ax_c.add_patch(rect)
    else:
        draw_c_overview(ax_c, highlight_block=True)

    # Draw A tile - show loading state
    if action == 'load_A' or action == 'load_B' or action == 'sync' or action == 'compute' or action == 'sync2':
        draw_as_tile(ax_as, k_tile)
        # Add loading indicator
        if action == 'load_A':
            ax_as.set_title(ax_as.get_title() + '\n[LOADING from global mem...]',
                          fontsize=8, color='#FF6F00', fontweight='bold')
        elif action == 'sync':
            ax_as.set_title(ax_as.get_title() + '\n[__syncthreads() ✓ — tile loaded]',
                          fontsize=8, color='#1565C0', fontweight='bold')
        elif action == 'compute':
            ax_as.set_title(ax_as.get_title() + '\n[READING for dot product]',
                          fontsize=8, color='#2E7D32', fontweight='bold')
    elif action == 'writeback':
        # Show empty tiles to indicate computation is done
        ax_as.text(0.5, 0.5, 'Computation\nComplete', transform=ax_as.transAxes,
                  ha='center', va='center', fontsize=14, fontweight='bold',
                  color='#1565C0')
        ax_as.set_xlim(0, 1)
        ax_as.set_ylim(1, 0)
    else:
        draw_as_tile(ax_as, 0)

    # Draw B tile
    if action == 'load_A' or action == 'load_B' or action == 'sync' or action == 'compute' or action == 'sync2':
        draw_bs_tile(ax_bs, k_tile)
        if action == 'load_B':
            ax_bs.set_title(ax_bs.get_title() + '\n[LOADING from global mem...]',
                          fontsize=8, color='#FF6F00', fontweight='bold')
        elif action == 'sync':
            ax_bs.set_title(ax_bs.get_title() + '\n[__syncthreads() ✓ — tile loaded]',
                          fontsize=8, color='#1565C0', fontweight='bold')
        elif action == 'compute':
            ax_bs.set_title(ax_bs.get_title() + '\n[READING for dot product]',
                          fontsize=8, color='#2E7D32', fontweight='bold')
    elif action == 'writeback':
        ax_bs.text(0.5, 0.5, 'All k_tiles\nprocessed',
                  transform=ax_bs.transAxes, ha='center', va='center',
                  fontsize=14, fontweight='bold', color='#1565C0')
        ax_bs.set_xlim(0, 1)
        ax_bs.set_ylim(1, 0)
    else:
        draw_bs_tile(ax_bs, 0)

    # Draw dot product
    if action == 'compute':
        draw_dot_product(ax_dot, k_tile, inner_k)
    elif action == 'writeback':
        ax_dot.clear()
        ax_dot.axis('off')

        ty, tx = 2, 3
        gr0 = ROW_START + ty
        gr1 = ROW_START + ty + TILE // 2
        gc0 = COL_START + tx
        gc1 = COL_START + tx + TILE // 2

        outputs = [
            (gr0, gc0, 's00'), (gr0, gc1, 's01'),
            (gr1, gc0, 's10'), (gr1, gc1, 's11'),
        ]

        ax_dot.text(0.5, 0.95, 'WRITEBACK PHASE — Boundary Check Before Each Write',
                   transform=ax_dot.transAxes, fontsize=11, fontweight='bold',
                   ha='center', va='top', color='#1565C0')

        for i, (gr, gc, name) in enumerate(outputs):
            y = 0.75 - i * 0.15
            row_ok = gr < M
            col_ok = gc < N
            will_write = row_ok and col_ok

            if will_write:
                status = f'[WRITE] row={gr}<{M} ✓, col={gc}<{N} ✓ → WRITE C[{gr}][{gc}] = {name}'
                color = '#2E7D32'
            else:
                if not row_ok and not col_ok:
                    status = f'[SKIP] row={gr}≥{M}, col={gc}≥{N} → SKIP'
                elif not row_ok:
                    status = f'[SKIP] row={gr}≥{M} → SKIP'
                else:
                    status = f'[SKIP] col={gc}≥{N} → SKIP'
                color = '#FF5252'

            ax_dot.text(0.1, y, status, transform=ax_dot.transAxes,
                       fontsize=10, va='center', fontfamily='monospace',
                       color=color, fontweight='bold')

        ax_dot.text(0.1, 0.1, 'OOB cells: computation happened (value = 0 or correct), but write is SKIPPED.\n'
                   'This prevents corrupting GPU memory beyond the C matrix.',
                   transform=ax_dot.transAxes, fontsize=9, va='center',
                   color='#555555')
    else:
        draw_dot_product(ax_dot, k_tile if action in ['load_A', 'load_B', 'sync', 'compute', 'sync2'] else 0,
                        inner_k if action == 'compute' else 0)
        if action in ['load_A', 'load_B', 'sync', 'sync2']:
            ax_dot.clear()
            ax_dot.axis('off')
            if action == 'load_A':
                msg = 'Loading A tile into shared memory...\nOOB rows → zero-padded (red cells)'
            elif action == 'load_B':
                msg = 'Loading B tile into shared memory...\nOOB cols → zero-padded (red cells)'
            elif action == 'sync':
                msg = '__syncthreads() barrier\nAll threads: tiles fully loaded\nOOB = 0-filled, valid = real data'
            else:
                msg = '__syncthreads() barrier\nAll threads: done with this k_tile\nReady to overwrite tiles for next k_tile'
            ax_dot.text(0.5, 0.5, msg, transform=ax_dot.transAxes,
                       ha='center', va='center', fontsize=12,
                       fontweight='bold', color='#1565C0')

    # Phase title
    fig.suptitle(f'Irregular GEMM: Boundary Check Visualization\n'
                f'M={M}, N={N}, K={K}, TILE={TILE}  |  {phase_text}',
                fontsize=11, fontweight='bold')

# Generate and save key static frames
print(f"Generating {len(phases)} frames, saving key snapshots...")

# Save key static frames only (GIF is too memory-heavy)
key_frames = [
    (0, '00_overview'),
    (1, '01_load_A_ktile0'),
    (3, '02_sync_ktile0'),
    (6, '03_compute_ktile0_k4'),
    (10, '04_compute_ktile1_k8'),
    (14, '05_compute_ktile2_last'),
    (n_frames - 1, '06_writeback'),
]

for frame_idx, name in key_frames:
    update(frame_idx)
    fig.savefig(f'gemm_boundary_{name}.png', dpi=100, bbox_inches='tight')
    print(f"Saved {name}.png (frame {frame_idx})")

print("\nAll frames saved!")
