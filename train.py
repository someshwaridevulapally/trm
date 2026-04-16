"""
CLI entry point for training the recursive neural network.

Usage:
    python train.py --task maze --epochs 20 --hidden_dim 128 --max_iters 10
    python train.py --task puzzle --epochs 30 --batch_size 64
    python train.py --task arc --epochs 20 --arc_data_dir data/arc
"""

import argparse
import sys
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Recursive Neural Network on maze, puzzle, or ARC tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Task selection ────────────────────────────────────────────────────
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["maze", "puzzle", "arc"],
        help="Which task to train on.",
    )

    # ── Model hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden-state dimensionality for encoder, RecCore, and decoder.")
    parser.add_argument("--max_iters", type=int, default=10,
                        help="Maximum number of RecCore iterations.")
    parser.add_argument("--epsilon", type=float, default=1e-3,
                        help="Convergence threshold for early stopping in RecCore.")

    # ── Training hyperparameters ──────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimiser.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    # ── Data parameters ───────────────────────────────────────────────────
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of mazes or puzzles to generate (ignored for ARC).")
    parser.add_argument("--arc_data_dir", type=str, default="data/arc",
                        help="Path to ARC-AGI dataset root (for --task arc).")
    parser.add_argument("--arc_max_tasks", type=int, default=None,
                        help="Limit number of ARC tasks to load (for debugging).")

    # ── Device ────────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (e.g. 'cpu', 'cuda', 'cuda:0'). "
                             "Auto-detects GPU if not specified.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")
    print(f"Task:   {args.task}")
    print(f"Config: hidden_dim={args.hidden_dim}  max_iters={args.max_iters}  "
          f"epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")
    print("=" * 70)

    # ── Dispatch to task-specific trainer ─────────────────────────────────
    if args.task == "maze":
        from tasks.maze.maze_trainer import train_maze
        results = train_maze(
            hidden_dim=args.hidden_dim,
            max_iters=args.max_iters,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_mazes=args.num_samples,
            device=args.device,
            seed=args.seed,
        )

    elif args.task == "puzzle":
        from tasks.puzzle.puzzle_trainer import train_puzzle
        results = train_puzzle(
            hidden_dim=args.hidden_dim,
            max_iters=args.max_iters,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_puzzles=args.num_samples,
            device=args.device,
            seed=args.seed,
        )

    elif args.task == "arc":
        from tasks.arc.arc_trainer import train_arc
        results = train_arc(
            hidden_dim=args.hidden_dim,
            max_iters=args.max_iters,
            epochs=args.epochs,
            batch_size=min(args.batch_size, 16),  # ARC needs smaller batches
            lr=args.lr,
            data_dir=args.arc_data_dir,
            device=args.device,
            max_tasks=args.arc_max_tasks,
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
    print(f"\nCheckpoint saved to: checkpoints/{args.task}/best_model.pt")
    print(f"Training log saved to: logs/{args.task}_training.csv")


if __name__ == "__main__":
    main()
