"""
Flask API server for the TRM Task Solver UI.

Provides endpoints:
  Maze:
    - GET  /api/generate       -> generates a new random maze
    - POST /api/solve          -> runs the maze model step-by-step
    - POST /api/solve_bfs      -> BFS ground-truth solver

  Puzzle:
    - GET  /api/puzzle/generate -> generates a scrambled 8-puzzle
    - POST /api/puzzle/solve    -> runs the puzzle model step-by-step

  ARC:
    - GET  /api/arc/generate   -> loads a random ARC task with demos
    - POST /api/arc/solve      -> runs the ARC model on the test input
"""

import os
import random
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from tasks.maze.maze_env import generate_solvable_maze, ACTION_DELTAS, bfs_solve
from tasks.puzzle.puzzle_env import (
    scramble_puzzle, solve_puzzle, GOAL_STATE,
    _state_to_tuple, _find_blank
)
from tasks.puzzle.puzzle_dataset import _state_to_onehot
from model.recursive_net import RecursiveNet

app = Flask(__name__, static_folder='ui/dist', static_url_path='')
CORS(app)

# -- Global state --------------------------------------------------------------
current_maze       = None
current_start      = None
current_goal       = None
current_gt_actions = None

current_puzzle_state   = None
current_puzzle_optimal = None

current_arc_task = None   # full task dict with demos + test

maze_model   = None
puzzle_model = None
arc_model    = None
arc_tasks    = None  # loaded ARC task list
device       = None


# -- Model loaders -------------------------------------------------------------

def load_maze_model():
    global maze_model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maze_model = RecursiveNet(
        in_channels=2,
        hidden_dim=256,
        head_sizes={"maze": 4},
        T=3, n=6,  # Maze is simpler, kept at 3,6
    ).to(device)
    ckpt = "checkpoints/maze/best_model.pt"
    if os.path.exists(ckpt):
        try:
            maze_model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            print(f"[OK] Loaded maze checkpoint: {ckpt}")
        except Exception as e:
            print(f"[WARN] Maze checkpoint incompatible: {e}")
            print("[WARN] Using random weights for maze.")
    else:
        print(f"[WARN] Maze checkpoint not found: {ckpt}. Using random weights.")
    maze_model.eval()


def load_puzzle_model():
    global puzzle_model, device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NOTE: must match training config — tile-tokenized encoder (9 tokens), T=5, n=8
    puzzle_model = RecursiveNet(
        in_channels=9,
        hidden_dim=512,
        head_sizes={"puzzle": 9},
        T=5, n=8,
        encoder_mode="tile_embed",
        grid_h=3, grid_w=3,
    ).to(device)
    ckpt = "checkpoints/puzzle/best_model.pt"
    if os.path.exists(ckpt):
        try:
            puzzle_model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            print(f"[OK] Loaded puzzle checkpoint: {ckpt}")
        except Exception as e:
            print(f"[WARN] Puzzle checkpoint incompatible (old architecture?): {e}")
            print("[WARN] Using random weights. Retrain with: python train.py --task puzzle")
    else:
        print(f"[WARN] Puzzle checkpoint not found: {ckpt}. Using random weights.")
        print("[INFO] Run: python train.py --task puzzle")
    puzzle_model.eval()


def load_arc_model():
    """Load ARC model (ARCModel wrapper) and the ARC task dataset for serving."""
    global arc_model, arc_tasks, device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tasks for the generate endpoint
    try:
        from tasks.arc.arc_loader import load_arc_dataset
        # Try common data paths
        for data_dir in ["data/arc/data", "data/arc"]:
            try:
                arc_tasks = load_arc_dataset(data_dir, split="training", max_tasks=None)
                break
            except FileNotFoundError:
                continue
        if arc_tasks is None:
            print("[WARN] ARC data not found. ARC generate will not work.")
            arc_tasks = []
    except Exception as e:
        print(f"[WARN] Could not load ARC tasks: {e}")
        arc_tasks = []

    # Load model
    from tasks.arc.arc_trainer import ARCModel
    arc_model = ARCModel(hidden_dim=512, T=5, n=8).to(device)
    ckpt = "checkpoints/arc/best_model.pt"
    if os.path.exists(ckpt):
        try:
            arc_model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            print(f"[OK] Loaded ARC checkpoint: {ckpt}")
        except Exception as e:
            print(f"[ERROR] ARC checkpoint incompatible (likely due to decoder upgrade): {e}")
            print("[WARN] Using random weights for ARC. Please run: python train.py --task arc")
    else:
        print(f"[WARN] ARC checkpoint not found: {ckpt}. Using random weights.")
    arc_model.eval()


# -- Static routes -------------------------------------------------------------

@app.route('/')
def index():
    return app.send_static_file('index.html')


# -- Maze endpoints ------------------------------------------------------------

@app.route('/api/blank')
def get_blank():
    size = 21
    grid = np.zeros((size, size), dtype=int).tolist()
    return jsonify({"grid": grid, "start": None, "goal": None, "status": "blank"})


@app.route('/api/generate')
def generate():
    global current_maze, current_start, current_goal, current_gt_actions
    seed = request.args.get('seed', random.randint(0, 100000), type=int)
    current_maze, current_start, current_goal, current_gt_actions = \
        generate_solvable_maze(height=10, width=10, seed=seed)
    return jsonify({
        "grid":          current_maze.tolist(),
        "start":         list(current_start),
        "goal":          list(current_goal),
        "optimal_steps": len(current_gt_actions),
        "status":        "generated"
    })


@app.route('/api/solve', methods=['POST'])
def solve():
    global current_maze, current_start, current_goal, maze_model, device
    if current_maze is None:
        return jsonify({"error": "No maze generated yet"}), 400

    grid = current_maze
    H, W = grid.shape
    goal = current_goal
    action_names = {0: "up", 1: "down", 2: "left", 3: "right"}

    start_r, start_c = current_start
    stack = [(start_r, start_c, [], set())]
    visited_states = {(start_r, start_c)}

    final_path = []
    max_steps  = 500
    steps_taken = 0

    while stack and steps_taken < max_steps:
        r, c, path_so_far, tried_actions = stack[-1]
        steps_taken += 1

        if (r, c) == goal:
            final_path = path_so_far
            break

        ch_maze = grid.astype(np.float32)
        ch_pos  = np.zeros((H, W), dtype=np.float32)
        ch_pos[r, c] = 1.0
        tensor = torch.tensor(np.stack([ch_maze, ch_pos])).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, logits_list = maze_model(tensor, task="maze", return_all=True)

        action_probs   = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        sorted_actions = np.argsort(-action_probs)

        # Build per-macro-step thinking trace
        _dir_keys = ["up", "down", "left", "right"]
        macro_steps = []
        for ml in logits_list:
            sp = torch.softmax(ml, dim=-1).squeeze().cpu().numpy()
            macro_steps.append({k: round(float(sp[i]), 4) for i, k in enumerate(_dir_keys)})

        found_next = False
        for action in sorted_actions:
            if action in tried_actions:
                continue
            dr, dc = ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W and
                    grid[nr, nc] == 0 and (nr, nc) not in visited_states):
                tried_actions.add(action)
                visited_states.add((nr, nc))
                new_path = path_so_far + [{
                    "row":         nr,
                    "col":         nc,
                    "action":      action_names[action],
                    "iterations":  len(logits_list),
                    "confidence":  float(action_probs[action]),
                    "macro_steps": macro_steps
                }]
                stack.append((nr, nc, new_path, set()))
                found_next = True
                break
            else:
                tried_actions.add(action)

        if not found_next:
            stack.pop()

    response_path = [{"row": start_r, "col": start_c, "action": None}]
    response_path.extend(final_path)
    solved = (len(final_path) > 0 and
              final_path[-1]["row"] == goal[0] and
              final_path[-1]["col"] == goal[1])

    return jsonify({
        "path":           response_path,
        "solved":         solved,
        "steps":          len(final_path),
        "optimal_steps":  len(current_gt_actions) if current_gt_actions else None,
        "status":         "solved" if solved else "failed",
        "total_explored": steps_taken
    })


@app.route('/api/solve_bfs', methods=['POST'])
def solve_bfs():
    global current_maze, current_start, current_goal
    if current_maze is None:
        return jsonify({"error": "No maze generated yet"}), 400

    actions = bfs_solve(current_maze, current_start, current_goal)
    if actions is None:
        return jsonify({"error": "No solution found"}), 400

    r, c = current_start
    path = [{"row": r, "col": c, "action": None}]
    action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
    for action in actions:
        dr, dc = ACTION_DELTAS[action]
        r, c = r + dr, c + dc
        path.append({"row": r, "col": c, "action": action_names[action]})

    return jsonify({"path": path, "solved": True, "steps": len(actions), "status": "solved"})


# -- Puzzle endpoints ----------------------------------------------------------

@app.route('/api/puzzle/generate')
def puzzle_generate():
    """Generate a scrambled 8-puzzle and its A* optimal solution."""
    global current_puzzle_state, current_puzzle_optimal
    seed       = request.args.get('seed', random.randint(0, 100000), type=int)
    num_moves  = request.args.get('moves', random.randint(10, 25), type=int)

    state, optimal_actions = scramble_puzzle(num_moves=num_moves, seed=seed)
    current_puzzle_state   = state
    current_puzzle_optimal = optimal_actions

    return jsonify({
        "state":         state.tolist(),
        "goal":          GOAL_STATE.tolist(),
        "optimal_steps": len(optimal_actions),
        "status":        "generated"
    })


@app.route('/api/puzzle/solve', methods=['POST'])
def puzzle_solve():
    """Run the puzzle model autoregressively to solve the current puzzle."""
    global current_puzzle_state, current_puzzle_optimal, puzzle_model, device

    if current_puzzle_state is None:
        return jsonify({"error": "No puzzle generated yet"}), 400

    goal_tuple = _state_to_tuple(GOAL_STATE)
    max_steps  = 150

    state  = current_puzzle_state.copy()
    states = [state.tolist()]
    actions_taken = []
    visited = set()

    for step in range(max_steps):
        st = _state_to_tuple(state)
        if st == goal_tuple:
            break
        if st in visited:
            break
        visited.add(st)

        oh     = _state_to_onehot(state)
        tensor = torch.tensor(oh).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, logits_list = puzzle_model(tensor, task="puzzle", return_all=True)

        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        sorted_actions = np.argsort(-probs)

        br, bc = _find_blank(state)
        moved  = False
        for action in sorted_actions:
            tr, tc = action // 3, action % 3
            if abs(br - tr) + abs(bc - tc) == 1:
                # Build per-macro-step thinking trace for this move
                macro_steps = []
                for ml in logits_list:
                    sp = torch.softmax(ml, dim=-1).squeeze().cpu().numpy()
                    macro_steps.append({
                        "probs":             [round(float(v), 4) for v in sp.tolist()],
                        "chosen_confidence": round(float(sp[action]), 4)
                    })
                new_state        = state.copy()
                new_state[br, bc] = new_state[tr, tc]
                new_state[tr, tc] = 0
                state = new_state
                states.append(state.tolist())
                actions_taken.append({
                    "action":      int(action),
                    "confidence":  float(probs[action]),
                    "macro_steps": macro_steps
                })
                moved = True
                break

        if not moved:
            break

    final_tuple = _state_to_tuple(state)
    solved = (final_tuple == goal_tuple)

    return jsonify({
        "states":        states,
        "actions":       actions_taken,
        "solved":        solved,
        "steps":         len(actions_taken),
        "optimal_steps": len(current_puzzle_optimal) if current_puzzle_optimal else None,
        "status":        "solved" if solved else "failed"
    })


# -- ARC endpoints -------------------------------------------------------------

@app.route('/api/arc/generate')
def arc_generate():
    """Load a random ARC task with its demonstration pairs and test input."""
    global current_arc_task, arc_tasks

    if not arc_tasks:
        return jsonify({"error": "No ARC tasks loaded"}), 400

    idx = request.args.get('index', random.randint(0, len(arc_tasks) - 1), type=int)
    idx = idx % len(arc_tasks)
    task = arc_tasks[idx]
    current_arc_task = task

    # Format demos as plain lists for JSON
    demos = []
    for pair in task["train"]:
        demos.append({
            "input":  pair["input"].tolist(),
            "output": pair["output"].tolist()
        })

    # Use first test pair
    test_pair = task["test"][0]

    return jsonify({
        "task_id":     task["task_id"],
        "demos":       demos,
        "test_input":  test_pair["input"].tolist(),
        "test_output": test_pair["output"].tolist(),
        "num_demos":   len(demos),
        "input_size":  list(test_pair["input"].shape),
        "output_size": list(test_pair["output"].shape),
        "status":      "generated"
    })


@app.route('/api/arc/solve', methods=['POST'])
def arc_solve():
    """Run the ARC model on the current task's test input."""
    global current_arc_task, arc_model, device

    if current_arc_task is None:
        return jsonify({"error": "No ARC task loaded yet"}), 400

    from tasks.arc.arc_loader import grid_to_tensor_channels

    MAX_H, MAX_W = 30, 30
    task = current_arc_task
    test_pair = task["test"][0]

    # Encode demonstration pairs
    demo_input_list = []
    demo_output_list = []
    for pair in task["train"]:
        di = grid_to_tensor_channels(pair["input"], MAX_H, MAX_W)
        do = grid_to_tensor_channels(pair["output"], MAX_H, MAX_W)
        demo_input_list.append(torch.tensor(di))
        demo_output_list.append(torch.tensor(do))

    demo_inputs  = torch.stack(demo_input_list).unsqueeze(0).to(device)   # (1, N, 10, H, W)
    demo_outputs = torch.stack(demo_output_list).unsqueeze(0).to(device)  # (1, N, 10, H, W)
    demo_mask    = torch.ones(1, len(task["train"])).to(device)           # (1, N)

    # Encode test input
    test_in = torch.tensor(
        grid_to_tensor_channels(test_pair["input"], MAX_H, MAX_W)
    ).unsqueeze(0).to(device)  # (1, 10, H, W)

    # Run model
    with torch.no_grad():
        logits, logits_list = arc_model(demo_inputs, demo_outputs, demo_mask, test_in, return_all=True)

    # logits shape: (1, MAX_H, MAX_W, 10) -> predicted grid
    pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # (MAX_H, MAX_W)

    # Crop prediction to actual output size
    out_h, out_w = test_pair["output"].shape
    pred_cropped = pred[:out_h, :out_w].tolist()

    # Ground truth
    gt = test_pair["output"].tolist()

    # Calculate accuracy
    correct = 0
    total   = out_h * out_w
    for r in range(out_h):
        for c in range(out_w):
            if pred_cropped[r][c] == gt[r][c]:
                correct += 1

    cell_acc = correct / max(total, 1)
    grid_match = (correct == total)

    # Build per-macro-step grid snapshots for the thinking panel
    macro_grids = []
    for ml in logits_list:
        step_pred = ml.argmax(dim=-1).squeeze(0).cpu().numpy()
        macro_grids.append(step_pred[:out_h, :out_w].tolist())

    return jsonify({
        "prediction":    pred_cropped,
        "ground_truth":  gt,
        "cell_accuracy": round(cell_acc, 4),
        "grid_match":    grid_match,
        "output_size":   [out_h, out_w],
        "macro_grids":   macro_grids,
        "solved":        grid_match,
        "status":        "solved" if grid_match else "failed"
    })


# -- Entry point ---------------------------------------------------------------

if __name__ == '__main__':
    load_maze_model()
    load_puzzle_model()
    load_arc_model()
    print("Starting TRM Task Solver UI server...")
    print("Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False)
