"""
Evaluation and visualisation script.

Loads a trained checkpoint and runs evaluation with visualisation:
  - Maze: solves random mazes end-to-end and draws solution paths
  - Puzzle: solves random 8-puzzles and shows state sequences
  - ARC: predicts outputs for test inputs and shows grids side-by-side

Usage:
    python eval.py --task maze --checkpoint checkpoints/maze/best_model.pt
    python eval.py --task puzzle --checkpoint checkpoints/puzzle/best_model.pt
    python eval.py --task arc --checkpoint checkpoints/arc/best_model.pt --arc_data_dir data/arc
"""

import argparse
import os
import torch
import numpy as np

from model.recursive_net import RecursiveNet
from utils.visualise import visualise_maze, visualise_puzzle_sequence, visualise_arc_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and visualise recursive model predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, required=True,
                        choices=["maze", "puzzle", "arc"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint. Defaults to checkpoints/{task}/best_model.pt")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--num_eval", type=int, default=5,
                        help="Number of examples to evaluate and visualise.")
    parser.add_argument("--save_dir", type=str, default="eval_output",
                        help="Directory to save visualisation images.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--arc_data_dir", type=str, default="data/arc")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def eval_maze(args, device):
    """Evaluate and visualise maze solutions."""
    from tasks.maze.maze_env import generate_solvable_maze, ACTION_DELTAS

    model = RecursiveNet(
        in_channels=2, hidden_dim=args.hidden_dim,
        head_sizes={"maze": 4}, max_iters=args.max_iters,
    ).to(device)

    ckpt = args.checkpoint or "checkpoints/maze/best_model.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print(f"⚠ Checkpoint not found: {ckpt}. Using random weights.")

    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)

    solved = 0
    for i in range(args.num_eval):
        grid, start, goal, gt_actions = generate_solvable_maze(
            height=10, width=10, seed=args.seed + i,
        )
        H, W = grid.shape

        # Roll out model predictions
        r, c = start
        pred_path = [(r, c)]
        for step in range(200):
            if (r, c) == goal:
                solved += 1
                break
            ch_maze = grid.astype(np.float32)
            ch_pos = np.zeros((H, W), dtype=np.float32)
            ch_pos[r, c] = 1.0
            tensor = torch.tensor(np.stack([ch_maze, ch_pos])).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, n_iters = model(tensor, task="maze")
            action = logits.argmax(dim=-1).item()
            dr, dc = ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                r, c = nr, nc
            pred_path.append((r, c))

        # Also compute GT path positions for comparison
        gr, gc = start
        gt_path = [(gr, gc)]
        for a in gt_actions:
            dr, dc = ACTION_DELTAS[a]
            gr, gc = gr + dr, gc + dc
            gt_path.append((gr, gc))

        # Visualise
        save_path = os.path.join(args.save_dir, f"maze_{i}_pred.png")
        visualise_maze(grid, path=pred_path, start=start, goal=goal,
                       title=f"Maze {i} – Model ({len(pred_path)-1} steps)", save_path=save_path)

        save_path_gt = os.path.join(args.save_dir, f"maze_{i}_gt.png")
        visualise_maze(grid, path=gt_path, start=start, goal=goal,
                       title=f"Maze {i} – BFS Optimal ({len(gt_actions)} steps)",
                       save_path=save_path_gt)

        print(f"  Maze {i}: model={len(pred_path)-1} steps, optimal={len(gt_actions)} steps, "
              f"reached_goal={(r,c)==goal}")

    print(f"\nSolved: {solved}/{args.num_eval}")
    print(f"Visualisations saved to {args.save_dir}/")


def eval_puzzle(args, device):
    """Evaluate and visualise 8-puzzle solutions."""
    import random
    from tasks.puzzle.puzzle_env import (
        scramble_puzzle, GOAL_STATE, _state_to_tuple, _find_blank,
    )
    from tasks.puzzle.puzzle_dataset import _state_to_onehot

    model = RecursiveNet(
        in_channels=9, hidden_dim=args.hidden_dim,
        head_sizes={"puzzle": 9}, max_iters=args.max_iters,
    ).to(device)

    ckpt = args.checkpoint or "checkpoints/puzzle/best_model.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print(f"⚠ Checkpoint not found: {ckpt}. Using random weights.")

    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)
    goal_tuple = _state_to_tuple(GOAL_STATE)

    solved = 0
    for i in range(args.num_eval):
        random.seed(args.seed + i)
        depth = random.randint(10, 25)
        state, optimal_actions = scramble_puzzle(num_moves=depth, seed=args.seed + i)
        optimal_len = len(optimal_actions)

        states_sequence = [state.copy()]
        visited = set()

        for step in range(80):
            st = _state_to_tuple(state)
            if st == goal_tuple:
                solved += 1
                break
            if st in visited:
                break
            visited.add(st)

            oh = _state_to_onehot(state)
            tensor = torch.tensor(oh).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(tensor, task="puzzle")
            action = logits.argmax(dim=-1).item()

            br, bc = _find_blank(state)
            tr, tc = action // 3, action % 3
            if abs(br - tr) + abs(bc - tc) == 1:
                state[br, bc] = state[tr, tc]
                state[tr, tc] = 0
                states_sequence.append(state.copy())
            else:
                break

        save_path = os.path.join(args.save_dir, f"puzzle_{i}.png")
        visualise_puzzle_sequence(
            states_sequence,
            title=f"Puzzle {i} – {len(states_sequence)-1} steps (optimal: {optimal_len})",
            save_path=save_path,
        )
        print(f"  Puzzle {i}: model={len(states_sequence)-1} steps, "
              f"optimal={optimal_len}, solved={_state_to_tuple(state)==goal_tuple}")

    print(f"\nSolved: {solved}/{args.num_eval}")
    print(f"Visualisations saved to {args.save_dir}/")


def eval_arc(args, device):
    """Evaluate and visualise ARC predictions."""
    from tasks.arc.arc_loader import load_arc_dataset, grid_to_tensor_channels, MAX_H, MAX_W
    from tasks.arc.arc_trainer import ARCModel
    from tasks.arc.arc_dataset import MAX_H, MAX_W

    model = ARCModel(hidden_dim=args.hidden_dim, max_iters=args.max_iters).to(device)

    ckpt = args.checkpoint or "checkpoints/arc/best_model.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print(f"⚠ Checkpoint not found: {ckpt}. Using random weights.")

    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        tasks = load_arc_dataset(args.arc_data_dir, split="training",
                                  max_tasks=args.num_eval)
    except FileNotFoundError as e:
        print(f"⚠ {e}")
        return

    grid_matches = 0
    for idx, task in enumerate(tasks[:args.num_eval]):
        demo_inputs_raw = [d["input"] for d in task["train"]]
        demo_outputs_raw = [d["output"] for d in task["train"]]

        for test_pair in task["test"]:
            test_in_raw = test_pair["input"]
            test_out_raw = test_pair["output"]
            out_h, out_w = test_out_raw.shape

            # Encode demos
            di_tensors = []
            do_tensors = []
            for di, do in zip(demo_inputs_raw, demo_outputs_raw):
                di_tensors.append(torch.tensor(grid_to_tensor_channels(di, MAX_H, MAX_W)))
                do_tensors.append(torch.tensor(grid_to_tensor_channels(do, MAX_H, MAX_W)))

            demo_in = torch.stack(di_tensors).unsqueeze(0).to(device)    # (1, N, 10, H, W)
            demo_out = torch.stack(do_tensors).unsqueeze(0).to(device)   # (1, N, 10, H, W)
            demo_mask = torch.ones(1, len(di_tensors)).to(device)        # (1, N)
            test_in = torch.tensor(
                grid_to_tensor_channels(test_in_raw, MAX_H, MAX_W)
            ).unsqueeze(0).to(device)  # (1, 10, H, W)

            with torch.no_grad():
                logits, n_iters = model(demo_in, demo_out, demo_mask, test_in)
            # logits: (1, MAX_H, MAX_W, 10)
            pred_full = logits[0].argmax(dim=-1).cpu().numpy()  # (MAX_H, MAX_W)
            pred = pred_full[:out_h, :out_w]

            match = np.array_equal(pred, test_out_raw)
            if match:
                grid_matches += 1

            # Visualise
            save_path = os.path.join(args.save_dir, f"arc_{task['task_id']}.png")
            visualise_arc_task(
                demo_inputs=demo_inputs_raw,
                demo_outputs=demo_outputs_raw,
                test_input=test_in_raw,
                test_output=test_out_raw,
                predicted_output=pred,
                title=f"ARC {task['task_id']} ({'✓ Match' if match else '✗ Mismatch'})",
                save_path=save_path,
            )
            print(f"  Task {task['task_id']}: match={match}, iters={n_iters}")

    print(f"\nExact grid matches: {grid_matches}/{min(args.num_eval, len(tasks))}")
    print(f"Visualisations saved to {args.save_dir}/")


def main():
    args = parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")

    if args.task == "maze":
        eval_maze(args, device)
    elif args.task == "puzzle":
        eval_puzzle(args, device)
    elif args.task == "arc":
        eval_arc(args, device)


if __name__ == "__main__":
    main()
