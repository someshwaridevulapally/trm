"""
Evaluation and visualisation script — updated for TRM.

Loads a trained checkpoint and runs evaluation:
  - Maze:   end-to-end rollout, draws solution paths
  - Puzzle: autoregressively solves 8-puzzles, shows state sequences
  - ARC:    predicts outputs for test inputs, shows grids side-by-side
  - Sudoku: predicts complete board from partial clues, shows board

Usage:
    python eval.py --task maze   --checkpoint checkpoints/maze/best_model.pt
    python eval.py --task puzzle --checkpoint checkpoints/puzzle/best_model.pt
    python eval.py --task arc    --checkpoint checkpoints/arc/best_model.pt --arc_data_dir data/arc
    python eval.py --task sudoku --checkpoint checkpoints/sudoku/best_model.pt
"""

import argparse
import os
import torch
import numpy as np

from model.recursive_net import RecursiveNet
from utils.visualise import visualise_maze, visualise_puzzle_sequence, visualise_arc_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and visualise TRM predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task",       type=str, required=True,
                        choices=["maze", "puzzle", "arc", "sudoku"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--T",          type=int, default=3,  help="Macro steps.")
    parser.add_argument("--n",          type=int, default=6,  help="Micro steps.")
    # Legacy alias
    parser.add_argument("--max_iters",  type=int, default=None,
                        help="[Deprecated] Use --T instead.")
    parser.add_argument("--num_eval",   type=int, default=5)
    parser.add_argument("--save_dir",   type=str, default="eval_output")
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--arc_data_dir", type=str, default="data/arc")
    parser.add_argument("--difficulty", type=str, default="extreme",
                        choices=["easy", "medium", "hard", "extreme"])
    parser.add_argument("--seed",       type=int, default=12345)
    return parser.parse_args()


# ── Maze ──────────────────────────────────────────────────────────────────

def eval_maze(args, device):
    from tasks.maze.maze_env import generate_solvable_maze, ACTION_DELTAS

    T = args.max_iters if args.max_iters is not None else args.T
    model = RecursiveNet(
        in_channels=2, hidden_dim=args.hidden_dim,
        head_sizes={"maze": 4}, T=T, n=args.n,
    ).to(device)

    ckpt = args.checkpoint or "checkpoints/maze/best_model.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"Loaded: {ckpt}")
    else:
        print(f"⚠ Checkpoint not found: {ckpt}. Using random weights.")

    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)
    solved = 0

    for i in range(args.num_eval):
        grid, start, goal, gt_actions = generate_solvable_maze(
            height=10, width=10, seed=args.seed + i)
        H, W = grid.shape
        r, c = start
        pred_path = [(r, c)]

        for _ in range(200):
            if (r, c) == goal:
                solved += 1
                break
            ch_maze = grid.astype(np.float32)
            ch_pos  = np.zeros((H, W), dtype=np.float32)
            ch_pos[r, c] = 1.0
            tensor = torch.tensor(np.stack([ch_maze, ch_pos])).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(tensor, task="maze")
            action = logits.argmax(dim=-1).item()
            dr, dc = ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                r, c = nr, nc
            pred_path.append((r, c))

        gr, gc = start
        gt_path = [(gr, gc)]
        for a in gt_actions:
            dr, dc = ACTION_DELTAS[a]
            gr, gc = gr + dr, gc + dc
            gt_path.append((gr, gc))

        visualise_maze(grid, path=pred_path, start=start, goal=goal,
                       title=f"Maze {i} – Model ({len(pred_path)-1} steps)",
                       save_path=os.path.join(args.save_dir, f"maze_{i}_pred.png"))
        visualise_maze(grid, path=gt_path, start=start, goal=goal,
                       title=f"Maze {i} – BFS Optimal ({len(gt_actions)} steps)",
                       save_path=os.path.join(args.save_dir, f"maze_{i}_gt.png"))
        print(f"  Maze {i}: model={len(pred_path)-1} steps, optimal={len(gt_actions)}, "
              f"reached_goal={(r,c)==goal}")

    print(f"\nSolved: {solved}/{args.num_eval}")


# ── 8-Puzzle ───────────────────────────────────────────────────────────────

def eval_puzzle(args, device):
    import random
    from tasks.puzzle.puzzle_env import scramble_puzzle, GOAL_STATE, _state_to_tuple, _find_blank
    from tasks.puzzle.puzzle_dataset import _state_to_onehot

    T = args.max_iters if args.max_iters is not None else args.T
    model = RecursiveNet(
        in_channels=9, hidden_dim=args.hidden_dim,
        head_sizes={"puzzle": 9}, T=T, n=args.n,
    ).to(device)

    ckpt = args.checkpoint or "checkpoints/puzzle/best_model.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"Loaded: {ckpt}")
    else:
        print(f"⚠ Checkpoint not found: {ckpt}. Using random weights.")

    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)
    goal_tuple = _state_to_tuple(GOAL_STATE)
    solved = 0

    for i in range(args.num_eval):
        random.seed(args.seed + i)
        depth = random.randint(10, 25)
        state, opt_actions = scramble_puzzle(num_moves=depth, seed=args.seed + i)
        states_seq = [state.copy()]
        visited    = set()

        for _ in range(80):
            st = _state_to_tuple(state)
            if st == goal_tuple:
                solved += 1
                break
            if st in visited:
                break
            visited.add(st)
            oh     = _state_to_onehot(state)
            tensor = torch.tensor(oh).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(tensor, task="puzzle")
            action = logits.argmax(dim=-1).item()
            br, bc = _find_blank(state)
            tr, tc = action // 3, action % 3
            if abs(br - tr) + abs(bc - tc) == 1:
                state[br, bc] = state[tr, tc]
                state[tr, tc] = 0
                states_seq.append(state.copy())
            else:
                break

        visualise_puzzle_sequence(
            states_seq,
            title=f"Puzzle {i} – {len(states_seq)-1} steps (optimal: {len(opt_actions)})",
            save_path=os.path.join(args.save_dir, f"puzzle_{i}.png"),
        )
        print(f"  Puzzle {i}: model={len(states_seq)-1} steps, optimal={len(opt_actions)}, "
              f"solved={_state_to_tuple(state)==goal_tuple}")

    print(f"\nSolved: {solved}/{args.num_eval}")


# ── ARC ───────────────────────────────────────────────────────────────────

def eval_arc(args, device):
    from tasks.arc.arc_loader import load_arc_dataset, grid_to_tensor_channels
    from tasks.arc.arc_trainer import ARCModel
    from tasks.arc.arc_dataset import MAX_H, MAX_W

    T = args.max_iters if args.max_iters is not None else args.T
    model = ARCModel(hidden_dim=args.hidden_dim, T=T, n=args.n).to(device)

    ckpt = args.checkpoint or "checkpoints/arc/best_model.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"Loaded: {ckpt}")
    else:
        print(f"⚠ Checkpoint not found: {ckpt}. Using random weights.")

    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        tasks = load_arc_dataset(args.arc_data_dir, split="training", max_tasks=args.num_eval)
    except FileNotFoundError as e:
        print(f"⚠ {e}")
        return

    grid_matches = 0
    for idx, task in enumerate(tasks[:args.num_eval]):
        demo_inputs_raw  = [d["input"]  for d in task["train"]]
        demo_outputs_raw = [d["output"] for d in task["train"]]

        for test_pair in task["test"]:
            test_in_raw  = test_pair["input"]
            test_out_raw = test_pair["output"]
            out_h, out_w = test_out_raw.shape

            di_tensors = [torch.tensor(grid_to_tensor_channels(di, MAX_H, MAX_W))
                          for di in demo_inputs_raw]
            do_tensors = [torch.tensor(grid_to_tensor_channels(do, MAX_H, MAX_W))
                          for do in demo_outputs_raw]

            demo_in  = torch.stack(di_tensors).unsqueeze(0).to(device)
            demo_out = torch.stack(do_tensors).unsqueeze(0).to(device)
            demo_mask = torch.ones(1, len(di_tensors)).to(device)
            test_in  = torch.tensor(
                grid_to_tensor_channels(test_in_raw, MAX_H, MAX_W)
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = model(demo_in, demo_out, demo_mask, test_in)

            pred_full = logits[0].argmax(dim=-1).cpu().numpy()
            pred      = pred_full[:out_h, :out_w]
            match     = np.array_equal(pred, test_out_raw)
            if match:
                grid_matches += 1

            visualise_arc_task(
                demo_inputs=demo_inputs_raw, demo_outputs=demo_outputs_raw,
                test_input=test_in_raw, test_output=test_out_raw,
                predicted_output=pred,
                title=f"ARC {task['task_id']} ({'✓' if match else '✗'})",
                save_path=os.path.join(args.save_dir, f"arc_{task['task_id']}.png"),
            )
            print(f"  Task {task['task_id']}: match={match}")

    print(f"\nExact grid matches: {grid_matches}/{min(args.num_eval, len(tasks))}")


# ── Sudoku ────────────────────────────────────────────────────────────────

def eval_sudoku(args, device):
    from tasks.sudoku.sudoku_env import generate_sudoku
    from tasks.sudoku.sudoku_dataset import board_to_onehot, DIFFICULTY_CLUES

    _HEAD_SIZE = 9 * 9 * 10
    T = args.max_iters if args.max_iters is not None else args.T
    model = RecursiveNet(
        in_channels=10, hidden_dim=args.hidden_dim,
        head_sizes={"sudoku": _HEAD_SIZE}, T=T, n=args.n,
    ).to(device)

    ckpt = args.checkpoint or "checkpoints/sudoku/best_model.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"Loaded: {ckpt}")
    else:
        print(f"⚠ Checkpoint not found: {ckpt}. Using random weights.")

    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)

    num_clues = DIFFICULTY_CLUES[args.difficulty]
    solved    = 0

    for i in range(args.num_eval):
        puzzle, solution = generate_sudoku(num_clues=num_clues, seed=args.seed + i)
        tensor = torch.tensor(board_to_onehot(puzzle)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(tensor, task="sudoku")

        pred_flat = logits.view(9, 9, 10).argmax(dim=-1).cpu().numpy()
        correct   = (pred_flat == solution).all()
        if correct:
            solved += 1

        # Simple text visualisation
        print(f"\n  Puzzle {i} ({'✓ SOLVED' if correct else '✗ WRONG'}):")
        print("  Clues:     " + " | ".join(
            " ".join(str(puzzle[r, c]) if puzzle[r, c] else "." for c in range(9))
            for r in range(9)
        ))
        print("  Predicted: " + " | ".join(
            " ".join(str(pred_flat[r, c]) for c in range(9))
            for r in range(9)
        ))
        print("  Solution:  " + " | ".join(
            " ".join(str(solution[r, c]) for c in range(9))
            for r in range(9)
        ))

    print(f"\nSolved: {solved}/{args.num_eval} ({solved/args.num_eval:.1%})")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    dispatch = {
        "maze":   eval_maze,
        "puzzle": eval_puzzle,
        "arc":    eval_arc,
        "sudoku": eval_sudoku,
    }
    dispatch[args.task](args, device)
    print(f"\nVisualisations saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
