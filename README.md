# Recursive Neural Network

A from-scratch recursive neural network that learns to solve structured reasoning tasks through iterative computation. The model uses a GRU-based recursive core that runs for multiple iterations (with convergence-based early stopping), enabling it to perform variable-depth reasoning.

## Architecture

```
Input Grid (B, C, H, W)
      │
      ▼
┌─────────────┐
│  CNN Encoder │  2-layer Conv2d → AdaptivePool → Linear
└──────┬──────┘
       │  (B, hidden_dim)
       ▼
┌──────────────────┐
│  Recursive Core  │  GRUCell × max_iters (with convergence check)
│  (+ optional     │  Optional: task embedding from MetaEncoder
│   ARC context)   │  concatenated at every iteration
└──────┬───────────┘
       │  (B, hidden_dim)
       ▼
┌─────────────────┐
│    Decoder       │  Task-switchable linear heads
│  maze:    → 4   │  (up/down/left/right)
│  puzzle:  → 9   │  (cell swap index)
│  arc:     → H×W×10│ (per-cell 10-class colour)
└─────────────────┘
```

## Tasks

### 🟢 Maze Solving
- **Grid**: 10×10 mazes generated via recursive backtracking
- **Solver**: BFS for shortest-path ground truth
- **Training**: Cross-entropy on per-step action predictions
- **Eval**: End-to-end solve rate, average steps vs optimal

### 🔵 8-Puzzle
- **Board**: 3×3 sliding tile puzzle (blank = 0)
- **Solver**: A* with Manhattan distance heuristic
- **Curriculum**: Scramble depth increases from 5 to 30 over training
- **Eval**: Solve rate, average solution length vs optimal

### 🟣 ARC-AGI
- **Format**: Few-shot learning from (input, output) demonstration pairs
- **MetaEncoder**: Encodes demo pairs → task embedding via shared CNN + mean pooling
- **Output**: Per-cell 10-class colour prediction
- **Eval**: Exact grid match accuracy

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train

```bash
# Train on maze task (fastest)
python train.py --task maze --epochs 20 --hidden_dim 128 --max_iters 10

# Train on 8-puzzle with curriculum learning
python train.py --task puzzle --epochs 30 --hidden_dim 128

# Train on ARC (requires ARC-AGI dataset)
git clone https://github.com/fchollet/ARC-AGI.git data/arc
python train.py --task arc --epochs 20 --arc_data_dir data/arc/data
```

### Evaluate & Visualise

```bash
python eval.py --task maze --num_eval 10
python eval.py --task puzzle --num_eval 5
python eval.py --task arc --num_eval 5 --arc_data_dir data/arc/data
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | (required) | `maze`, `puzzle`, or `arc` |
| `--hidden_dim` | 128 | Hidden state size |
| `--max_iters` | 10 | Max RecCore iterations |
| `--epochs` | 20 | Training epochs |
| `--batch_size` | 128 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--num_samples` | 10000 | Mazes/puzzles to generate |
| `--device` | auto | `cpu` or `cuda` |
| `--seed` | 42 | Random seed |

## Project Structure

```
recursive-model/
├── model/
│   ├── encoder.py        # CNN encoder for 2D grid inputs
│   ├── rec_core.py       # GRU-based recursive core with iteration loop
│   ├── decoder.py        # Task-switchable output head
│   └── recursive_net.py  # Main model: encoder → RecCore → decoder
├── tasks/
│   ├── maze/
│   │   ├── maze_env.py       # Random maze generator + BFS solver
│   │   ├── maze_dataset.py   # PyTorch Dataset from BFS paths
│   │   └── maze_trainer.py   # Training + eval loop
│   ├── puzzle/
│   │   ├── puzzle_env.py     # 8-puzzle environment + A* solver
│   │   ├── puzzle_dataset.py # Dataset from A* solutions
│   │   └── puzzle_trainer.py # Training + eval loop with curriculum
│   └── arc/
│       ├── arc_loader.py     # Load ARC-AGI tasks from JSON
│       ├── arc_dataset.py    # Few-shot dataset with demo pairs
│       ├── meta_encoder.py   # Demo pair encoder → task embedding
│       └── arc_trainer.py    # Training + eval loop
├── utils/
│   ├── convergence.py    # ||h_t - h_{t-1}|| < ε convergence check
│   └── visualise.py      # Matplotlib visualisations
├── train.py              # CLI training entry point
├── eval.py               # Evaluation + visualisation script
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Outputs

- **Checkpoints**: `checkpoints/{task}/best_model.pt`
- **Training logs**: `logs/{task}_training.csv` (epoch, loss, accuracy per epoch)
- **Visualisations**: `eval_output/` (PNG images from eval.py)

## Requirements

- Python 3.10+
- PyTorch 2.x
- NumPy, Matplotlib, tqdm
