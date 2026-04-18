"""
Grid Encoder for TRM — Tile-Tokenized.

Maps a 2-D grid tensor (B, C, H, W) to a sequence of token embeddings
(B, num_cells, hidden_dim) where each grid cell gets its own token.

For the 8-puzzle (3×3 grid) this produces 9 tokens — one per cell —
enabling the TinyTransformer to attend across tile positions and
reason spatially about the board state.

Key insight: The old single-token encoder collapsed all spatial info
into one vector, making Transformer self-attention a no-op. With 9 tokens,
the micro/macro recursion loops can now model inter-tile relationships.

Two encoder modes:
  - 'tile_embed'  (default, puzzle/sudoku): tile-value embedding + 2D pos embed.
  - 'grid_patch'  (ARC/maze): lightweight CNN patch encoder per spatial cell.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Tile-tokenized grid encoder: one token per grid cell.

    For a 3×3 puzzle board with one-hot input (9_channels, 3, 3):
      - Reads the tile value at each of the 9 cell positions.
      - Embeds tile value (0–8) via a learnable embedding table.
      - Adds a learnable 2-D position embedding for the cell's (row, col).
      - Projects to hidden_dim via a linear layer.
    Output shape: (B, 9, hidden_dim)

    For non-square or larger grids (ARC, maze), falls back to a CNN-patch
    encoder that extracts per-cell features with a shared small CNN.

    Args:
        in_channels (int):  Number of input channels (one-hot classes for puzzle).
        hidden_dim  (int):  Output embedding dimensionality. Default 512.
        grid_h      (int):  Grid height. Default 3 (8-puzzle).
        grid_w      (int):  Grid width.  Default 3 (8-puzzle).
        mode        (str):  'tile_embed' or 'grid_patch'.
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_dim: int = 512,
        grid_h: int = 3,
        grid_w: int = 3,
        mode: str = "tile_embed",
    ):
        super().__init__()
        self.mode = mode
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_cells = grid_h * grid_w   # 9 for 3×3
        self.hidden_dim = hidden_dim

        if mode == "tile_embed":
            # Learnable embedding for each tile value (0 = blank, 1–8 = tiles)
            # in_channels == number of possible tile values (9 for 8-puzzle)
            self.tile_embed = nn.Embedding(in_channels, hidden_dim // 2)
            # Learnable 2-D position embedding: one vector per (row, col) cell
            self.pos_embed  = nn.Embedding(self.num_cells, hidden_dim // 2)
            self.proj       = nn.Linear(hidden_dim, hidden_dim)
            self.norm       = nn.LayerNorm(hidden_dim)
        else:
            # grid_patch: shared tiny CNN applied per cell patch
            patch_flat = in_channels * grid_h * grid_w
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),
            )
            self.fc   = nn.Linear(64 * 4 * 4, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Grid input.
               - tile_embed mode: one-hot (B, C, H, W) where C = num tile values.
               - grid_patch mode: arbitrary (B, C, H, W).

        Returns:
            Token embeddings, shape (B, num_cells, hidden_dim).
            For tile_embed: num_cells = H*W = 9 for 3×3 puzzle.
        """
        if self.mode == "tile_embed":
            return self._tile_embed_forward(x)
        else:
            return self._grid_patch_forward(x)

    def _tile_embed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert one-hot board (B, C, H, W) → tile-ID tensor (B, H*W)
        then embed each position separately.

        For the 8-puzzle with one-hot input (B, 9, 3, 3):
          x.argmax(dim=1)  →  (B, 3, 3) integer tile IDs (0–8)
          flatten          →  (B, 9)    one ID per cell position
        """
        B = x.shape[0]
        # Tile ID at each cell: argmax over the channel (one-hot) dimension
        tile_ids = x.argmax(dim=1)                      # (B, H, W)
        tile_ids = tile_ids.view(B, -1)                 # (B, 9)  — flattened cell index

        # Position indices: 0,1,...,8 for cells left→right, top→bottom
        pos_ids = torch.arange(self.num_cells, device=x.device)   # (9,)
        pos_ids = pos_ids.unsqueeze(0).expand(B, -1)               # (B, 9)

        # Embed tile values and positions, then concatenate
        t_emb = self.tile_embed(tile_ids)               # (B, 9, hidden_dim//2)
        p_emb = self.pos_embed(pos_ids)                 # (B, 9, hidden_dim//2)
        tokens = torch.cat([t_emb, p_emb], dim=-1)      # (B, 9, hidden_dim)

        # Project + normalise
        tokens = self.proj(tokens)                      # (B, 9, hidden_dim)
        tokens = self.norm(tokens)                      # (B, 9, hidden_dim)
        return tokens

    def _grid_patch_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Legacy CNN path for non-tile tasks (maze, ARC).
        Returns a single-token summary (B, 1, hidden_dim) for backward compat.
        """
        feat = self.cnn(x)                              # (B, 64, 4, 4)
        feat = feat.view(feat.size(0), -1)              # (B, 1024)
        feat = self.fc(feat)                            # (B, hidden_dim)
        feat = self.norm(feat)                          # (B, hidden_dim)
        return feat.unsqueeze(1)                        # (B, 1, hidden_dim)
