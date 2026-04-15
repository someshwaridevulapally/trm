"""
Visualisation utilities for maze, 8-puzzle, and ARC tasks.

Provides Matplotlib-based rendering functions:
  - Maze grids with solution paths overlaid
  - 8-puzzle state sequences as tile boards
  - ARC input/output grid pairs shown side-by-side
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from typing import List, Optional, Tuple


# ── ARC colour palette (official 10 colours 0–9) ──────────────────────────────
ARC_COLORS = [
    "#000000",  # 0 – black
    "#0074D9",  # 1 – blue
    "#FF4136",  # 2 – red
    "#2ECC40",  # 3 – green
    "#FFDC00",  # 4 – yellow
    "#AAAAAA",  # 5 – grey
    "#F012BE",  # 6 – magenta
    "#FF851B",  # 7 – orange
    "#7FDBFF",  # 8 – cyan
    "#B10DC9",  # 9 – maroon/purple
]
ARC_CMAP = mcolors.ListedColormap(ARC_COLORS)
ARC_NORM = mcolors.BoundaryNorm(np.arange(-0.5, 10.5, 1), ARC_CMAP.N)


# ── Maze visualisation ────────────────────────────────────────────────────────

def visualise_maze(
    grid: np.ndarray,
    path: Optional[List[Tuple[int, int]]] = None,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    title: str = "Maze",
    save_path: Optional[str] = None,
) -> None:
    """
    Render a maze grid with an optional solution path overlaid.

    Args:
        grid:      2-D numpy array (H, W) with 0 = open, 1 = wall.
        path:      List of (row, col) positions forming the solution.
        start:     (row, col) of the start cell (drawn as a green circle).
        goal:      (row, col) of the goal cell (drawn as a red star).
        title:     Plot title.
        save_path: If given, save figure to this file instead of showing.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # Draw grid: walls black, open white
    ax.imshow(grid, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)

    # Overlay solution path
    if path is not None and len(path) > 1:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        ax.plot(cols, rows, color="#FF4136", linewidth=2.5, alpha=0.85, zorder=2)

    # Mark start and goal
    if start is not None:
        ax.plot(start[1], start[0], "o", color="#2ECC40", markersize=10, zorder=3)
    if goal is not None:
        ax.plot(goal[1], goal[0], "*", color="#FF4136", markersize=14, zorder=3)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── 8-puzzle visualisation ────────────────────────────────────────────────────

def visualise_puzzle_sequence(
    states: List[np.ndarray],
    max_display: int = 12,
    title: str = "8-Puzzle Solution",
    save_path: Optional[str] = None,
) -> None:
    """
    Show a sequence of 3×3 puzzle states as a strip of tile boards.

    Args:
        states:      List of 3×3 numpy arrays (0 = blank).
        max_display: Maximum number of states to render.
        title:       Overall figure title.
        save_path:   If given, save figure to this file.
    """
    states = states[:max_display]
    n = len(states)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.5))
    if n == 1:
        axes = [axes]

    for idx, (ax, state) in enumerate(zip(axes, states)):
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(2.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        for r in range(3):
            for c in range(3):
                val = int(state[r, c])
                color = "#F5F5F5" if val == 0 else "#0074D9"
                text_color = "#0074D9" if val == 0 else "white"
                rect = plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                     facecolor=color, edgecolor="#333", linewidth=1.5,
                                     zorder=1)
                ax.add_patch(rect)
                if val != 0:
                    ax.text(c, r, str(val), ha="center", va="center",
                            fontsize=16, fontweight="bold", color=text_color, zorder=2)

        ax.set_title(f"t={idx}", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── ARC grid visualisation ───────────────────────────────────────────────────

def _draw_arc_grid(ax, grid: np.ndarray, title: str = "") -> None:
    """Helper: draw a single ARC grid on a Matplotlib axis."""
    h, w = grid.shape
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest")
    # Draw cell borders
    for r in range(h + 1):
        ax.axhline(r - 0.5, color="#555", linewidth=0.5)
    for c in range(w + 1):
        ax.axvline(c - 0.5, color="#555", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def visualise_arc_task(
    demo_inputs: List[np.ndarray],
    demo_outputs: List[np.ndarray],
    test_input: np.ndarray,
    test_output: Optional[np.ndarray] = None,
    predicted_output: Optional[np.ndarray] = None,
    title: str = "ARC Task",
    save_path: Optional[str] = None,
) -> None:
    """
    Show ARC demo pairs side-by-side and the test input/output.

    Args:
        demo_inputs:      List of 2-D arrays for demonstration inputs.
        demo_outputs:     List of 2-D arrays for demonstration outputs.
        test_input:       2-D array for the test input.
        test_output:      Ground-truth test output (optional).
        predicted_output: Model's predicted output (optional).
        title:            Figure title.
        save_path:        If given, save figure.
    """
    n_demos = len(demo_inputs)
    # Columns: 2 per demo (in/out) + test_in + optional gt + optional pred
    n_extra = 1 + (1 if test_output is not None else 0) + (1 if predicted_output is not None else 0)
    n_cols = 2 * n_demos + n_extra
    fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 3))
    if n_cols == 1:
        axes = [axes]

    col = 0
    for i in range(n_demos):
        _draw_arc_grid(axes[col], demo_inputs[i], f"Demo {i+1} In")
        col += 1
        _draw_arc_grid(axes[col], demo_outputs[i], f"Demo {i+1} Out")
        col += 1

    _draw_arc_grid(axes[col], test_input, "Test In")
    col += 1

    if test_output is not None:
        _draw_arc_grid(axes[col], test_output, "Test Out (GT)")
        col += 1

    if predicted_output is not None:
        _draw_arc_grid(axes[col], predicted_output, "Predicted")
        col += 1

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
