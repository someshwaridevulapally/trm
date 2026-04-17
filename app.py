"""
Flask API server for the Maze Solver UI.

Provides endpoints:
  - GET /api/blank       → returns a blank grid
  - GET /api/generate    → generates a new random maze
  - POST /api/solve      → runs the model to solve the maze step-by-step
"""

import os
import random
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from tasks.maze.maze_env import generate_solvable_maze, ACTION_DELTAS, bfs_solve
from model.recursive_net import RecursiveNet

app = Flask(__name__, static_folder='ui/dist', static_url_path='')
CORS(app)

# Global state
current_maze = None
current_start = None
current_goal = None
current_gt_actions = None
model = None
device = None


def load_model():
    """Load the trained maze model."""
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecursiveNet(
        in_channels=2,
        hidden_dim=256,
        head_sizes={"maze": 4},
        T=3,
        n=6,
    ).to(device)
    
    ckpt_path = "checkpoints/maze/best_model.pt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"✓ Loaded checkpoint: {ckpt_path}")
    else:
        print(f"⚠ Checkpoint not found: {ckpt_path}. Using random weights.")
    model.eval()


@app.route('/')
def index():
    """Serve the frontend."""
    return app.send_static_file('index.html')


@app.route('/api/blank')
def get_blank():
    """Return a blank 21x21 grid (all zeros)."""
    size = 21  # 2*10+1 for a 10x10 logical maze
    grid = np.zeros((size, size), dtype=int).tolist()
    return jsonify({
        "grid": grid,
        "start": None,
        "goal": None,
        "status": "blank"
    })


@app.route('/api/generate')
def generate():
    """Generate a new random maze."""
    global current_maze, current_start, current_goal, current_gt_actions
    
    seed = request.args.get('seed', random.randint(0, 100000), type=int)
    current_maze, current_start, current_goal, current_gt_actions = generate_solvable_maze(
        height=10, width=10, seed=seed
    )
    
    return jsonify({
        "grid": current_maze.tolist(),
        "start": list(current_start),
        "goal": list(current_goal),
        "optimal_steps": len(current_gt_actions),
        "status": "generated"
    })


@app.route('/api/solve', methods=['POST'])
def solve():
    """
    Solve the current maze using the model with backtracking.
    When stuck, tries alternative actions ranked by model confidence.
    """
    global current_maze, current_start, current_goal, model, device
    
    if current_maze is None:
        return jsonify({"error": "No maze generated yet"}), 400
    
    grid = current_maze
    H, W = grid.shape
    goal = current_goal
    action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
    
    # DFS with model-guided action ordering
    # Stack: (row, col, path_so_far, tried_actions_at_this_pos)
    start_r, start_c = current_start
    stack = [(start_r, start_c, [], set())]
    visited_states = set()
    visited_states.add((start_r, start_c))
    
    final_path = []
    max_steps = 500
    steps_taken = 0
    
    while stack and steps_taken < max_steps:
        r, c, path_so_far, tried_actions = stack[-1]
        steps_taken += 1
        
        if (r, c) == goal:
            final_path = path_so_far
            break
        
        # Get model's action preferences
        ch_maze = grid.astype(np.float32)
        ch_pos = np.zeros((H, W), dtype=np.float32)
        ch_pos[r, c] = 1.0
        tensor = torch.tensor(np.stack([ch_maze, ch_pos])).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, logits_list = model(tensor, task="maze")
        
        # Sort actions by model confidence (highest first)
        action_probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        sorted_actions = np.argsort(-action_probs)  # descending order
        
        # Try actions in order of model preference
        found_next = False
        for action in sorted_actions:
            if action in tried_actions:
                continue
            
            dr, dc = ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc
            
            # Check if valid and not visited
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0 and (nr, nc) not in visited_states:
                tried_actions.add(action)
                visited_states.add((nr, nc))
                
                new_path = path_so_far + [{
                    "row": nr,
                    "col": nc,
                    "action": action_names[action],
                    "iterations": len(logits_list),
                    "confidence": float(action_probs[action])
                }]
                
                stack.append((nr, nc, new_path, set()))
                found_next = True
                break
            else:
                tried_actions.add(action)
        
        if not found_next:
            # Backtrack - no valid untried actions from this position
            stack.pop()
    
    # Build final response path
    response_path = [{"row": start_r, "col": start_c, "action": None}]
    response_path.extend(final_path)
    
    solved = len(final_path) > 0 and final_path[-1]["row"] == goal[0] and final_path[-1]["col"] == goal[1]
    
    return jsonify({
        "path": response_path,
        "solved": solved,
        "steps": len(final_path),
        "optimal_steps": len(current_gt_actions) if current_gt_actions else None,
        "status": "solved" if solved else "failed",
        "total_explored": steps_taken
    })


@app.route('/api/solve_bfs', methods=['POST'])
def solve_bfs():
    """Solve using BFS (ground truth) for comparison."""
    global current_maze, current_start, current_goal
    
    if current_maze is None:
        return jsonify({"error": "No maze generated yet"}), 400
    
    actions = bfs_solve(current_maze, current_start, current_goal)
    
    if actions is None:
        return jsonify({"error": "No solution found"}), 400
    
    # Build path
    r, c = current_start
    path = [{"row": r, "col": c, "action": None}]
    action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
    
    for action in actions:
        dr, dc = ACTION_DELTAS[action]
        r, c = r + dr, c + dc
        path.append({
            "row": r,
            "col": c,
            "action": action_names[action]
        })
    
    return jsonify({
        "path": path,
        "solved": True,
        "steps": len(actions),
        "status": "solved"
    })


if __name__ == '__main__':
    load_model()
    print("Starting Maze Solver UI server...")
    print("Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False)
