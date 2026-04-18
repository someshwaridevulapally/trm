"""
CLI entry point for training the TRM (Tiny Recursion Model).

Updated for arXiv:2510.04871v1 "Less is More: Recursive Reasoning with Tiny Networks":
  - T (macro steps) and n (micro steps) replace old max_iters
  - hidden_dim default = 512 (paper setting)
  - AdamW + warmup-cosine LR scheduler
  - EMA on model weights
  - Puzzle: tile-tokenized encoder (9 tokens), ramped deep supervision
  - Sudoku task added

Usage:
    python train.py --task maze    --epochs 20 --hidden_dim 512 --T 3 --n 6
    python train.py --task puzzle  --epochs 40 --hidden_dim 512 --T 5 --n 8
    python train.py --task arc     --epochs 20 --arc_data_dir data/arc
    python train.py --task sudoku  --epochs 30 --difficulty extreme
"""

import argparse
import sys
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the TRM on maze, puzzle, ARC, or Sudoku tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Task selection ─────────────────────────────────────────────────────
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["maze", "puzzle", "arc", "sudoku"],
        help="Which task to train on.",
    )

    # ── TRM model hyperparameters ──────────────────────────────────────────
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Model hidden dimensionality (d_model). Paper uses 512.")
    parser.add_argument("--T", type=int, default=5,
                        help="Number of macro (outer) recursion steps. Default 5 for puzzle.")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of micro (inner) steps per macro step. Default 8 for puzzle.")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Attention heads in the Transformer core.")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Transformer layers per block (paper uses 2).")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout probability.")

    # Legacy alias
    parser.add_argument("--max_iters", type=int, default=None,
                        help="[Deprecated] Use --T instead. Maps to macro steps.")

    # ── Training hyperparameters ───────────────────────────────────────────
    parser.add_argument("--epochs",     type=int,   default=40,  help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int,   default=128, help="Batch size for training.")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate for AdamW.")
    parser.add_argument("--ema_decay",  type=float, default=0.999, help="EMA decay factor.")
    parser.add_argument("--seed",       type=int,   default=42,  help="Random seed.")

    # ── Data parameters ───────────────────────────────────────────────────
    parser.add_argument("--num_samples", type=int, default=10_000,
                        help="Mazes / puzzles / Sudoku boards to generate.")
    parser.add_argument("--arc_data_dir", type=str, default="data/arc",
                        help="Path to ARC-AGI dataset root (for --task arc).")
    parser.add_argument("--arc_max_tasks", type=int, default=None,
                        help="Limit ARC tasks loaded (debugging).")
    parser.add_argument("--difficulty", type=str, default="extreme",
                        choices=["easy", "medium", "hard", "extreme"],
                        help="Sudoku difficulty level (for --task sudoku).")

    # ── Device ────────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu', 'cuda', 'cuda:0'. Auto-detects if unset.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Map legacy --max_iters to T
    T = args.max_iters if (args.max_iters is not None) else args.T

    print(f"Device:     {args.device}")
    print(f"Task:       {args.task}")
    print(f"Config:     hidden_dim={args.hidden_dim}  T={T}  n={args.n}  "
          f"epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}  ema={args.ema_decay}")
    print("=" * 70)

    # ── Dispatch ──────────────────────────────────────────────────────────
    if args.task == "maze":
        from tasks.maze.maze_trainer import train_maze
        results = train_maze(
            hidden_dim=args.hidden_dim,
            T=T, n=args.n,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_mazes=args.num_samples,
            device=args.device,
            seed=args.seed,
            ema_decay=args.ema_decay,
        )

    elif args.task == "puzzle":
        from tasks.puzzle.puzzle_trainer import train_puzzle
        results = train_puzzle(
            hidden_dim=args.hidden_dim,
            T=T, n=args.n,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_puzzles=args.num_samples,
            device=args.device,
            seed=args.seed,
            ema_decay=args.ema_decay,
        )

    elif args.task == "arc":
        from tasks.arc.arc_trainer import train_arc
        results = train_arc(
            hidden_dim=args.hidden_dim,
            T=T, n=args.n,
            epochs=args.epochs,
            batch_size=min(args.batch_size, 16),
            lr=args.lr,
            data_dir=args.arc_data_dir,
            device=args.device,
            max_tasks=args.arc_max_tasks,
            ema_decay=args.ema_decay,
        )

    elif args.task == "sudoku":
        from tasks.sudoku.sudoku_trainer import train_sudoku
        results = train_sudoku(
            hidden_dim=args.hidden_dim,
            T=T, n=args.n,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            difficulty=args.difficulty,
            num_puzzles=args.num_samples,
            device=args.device,
            seed=args.seed,
            ema_decay=args.ema_decay,
        )

    else:
        print(f"Unknown task: {args.task}")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Training complete! Results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nCheckpoint: checkpoints/{args.task}/best_model.pt")
    print(f"Log:        logs/{args.task}_training.csv")


if __name__ == "__main__":
    main()
