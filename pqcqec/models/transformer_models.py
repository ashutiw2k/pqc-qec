import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

# --- Configuration ---
MAX_BASE_LEN = 1024
PREV_K = 1
HID_DIM = 256
DROP = 0.1
N_HEADS = 8
FF_DIM = 1024
N_LAYERS = 6


class ZZRingAnglePredictor(nn.Module):
    """Transformer model that predicts 7*n_qubits angles per PQC block. """
    
    def __init__(self, gate_blocks: int, n_qubits: int):
        super().__init__()
        self.gate_blocks = int(gate_blocks)
        self.n_qubits = int(n_qubits)
        self.angles_per_block = 7 * n_qubits
        
        self.max_blocks = math.ceil(MAX_BASE_LEN / max(1, gate_blocks))
        
        # Input features: [gate_count, cumulative_count, block_index, prev_angles_flattened]
        self.feat_dim = 3 + self.angles_per_block * PREV_K
        
        self.in_proj = nn.Sequential(
            nn.Linear(self.feat_dim, HID_DIM),
            nn.GELU(),
            nn.Dropout(DROP),
            nn.LayerNorm(HID_DIM)
        )
        
        self.pos_emb = nn.Embedding(self.max_blocks, HID_DIM)
        
        # Causal transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            HID_DIM, N_HEADS, FF_DIM, DROP,
            batch_first=True, norm_first=True
        )
        try:
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_layers=N_LAYERS,
                enable_nested_tensor=False
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        
        # Output head: 2 values (x,y on unit circle) per angle
        self.head_ln = nn.LayerNorm(HID_DIM)
        self.head = nn.Linear(HID_DIM, 2 * self.angles_per_block)
        
        # Cache for attention masks
        self._attn_mask_cache: dict[Tuple[int, torch.device], torch.Tensor] = {}
        
        # Initialize to predict identity (all angles = 0)
        self._init_head()

    def _init_head(self) -> None:
        """Initialize the output head to predict angle 0 (x=1, y=0)."""
        with torch.no_grad():
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            if self.head.bias.numel() >= 2 * self.angles_per_block:
                b = self.head.bias.view(self.angles_per_block, 2)
                b[:, 0] = 1.0  # x=1 -> angle=0
                b[:, 1] = 0.0  # y=0
    
    def _get_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Get or create cached causal attention mask.
        
        Args:
            size: Sequence length
            device: torch device
            
        Returns:
            mask: [size, size] boolean mask
        """
        cache_key = (size, device)
        if cache_key not in self._attn_mask_cache:
            mask = torch.triu(
                torch.ones((size, size), dtype=torch.bool, device=device),
                diagonal=1
            )
            self._attn_mask_cache[cache_key] = mask
        return self._attn_mask_cache[cache_key]
    
    def _angles_from_s1(self, logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Convert S¹ representation to angles [-π, π] with improved stability.
        
        Args:
            logits: [..., 2*angles_per_block] raw outputs
            eps: numerical stability epsilon
            
        Returns:
            angles: [..., angles_per_block] in [-π, π]
        """
        in_shape = logits.shape
        all_dims = in_shape[:-1]
        
        xy = logits.view(*all_dims, self.angles_per_block, 2)  # [..., A, 2]
        x, y = xy[..., 0], xy[..., 1]
        
        # Use hypot for better numerical stability
        r = torch.hypot(x, y).clamp(min=eps)
        x_norm = x / r
        y_norm = y / r
        
        # Convert to angle
        theta = torch.atan2(y_norm, x_norm)  # Already in [-π, π]
        
        # Handle any numerical issues
        theta = torch.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
        
        return theta

    def _get_vectorized_features(
        self, 
        batch, 
        max_blocks: int, 
        target_angles: torch.Tensor,  # [B, T, A]
        device: torch.device
    ) -> torch.Tensor:  # [B, T, feat_dim]
        """Compute all features in a vectorized (parallel) way."""
        B = batch.base_g.size(0)
        
        # 1. Vectorized gate counts [B, T]
        block_indices = torch.arange(max_blocks, device=device).unsqueeze(0)  # [1, T]
        block_starts = block_indices * self.gate_blocks
        block_ends = (block_indices + 1) * self.gate_blocks
        
        Lb_expanded = batch.base_len.unsqueeze(1).float()  # [B, 1]
        clipped_ends = torch.min(block_ends, Lb_expanded)
        counts = torch.clamp(clipped_ends - block_starts, min=0.0)  # [B, T]
        
        # 2. Cumulative counts [B, T]
        cum = counts.cumsum(dim=1)
        
        # 3. Block indices [B, T]
        idx_seq = block_indices.expand(B, -1).float()
        
        # 4. Previous angles (teacher-forced) [B, T, K*A]
        # Pad time dimension with K zeros at the beginning
        padded_angles = F.pad(
            target_angles, (0, 0, PREV_K, 0), 'constant', 0.0
        )  # [B, K+T, A]
        
        # Sliding window of size K
        unfolded_prev = padded_angles.unfold(dimension=1, size=PREV_K, step=1)
        # Shape: [B, T, A, K]
        
        # Reshape to [B, T, K*A]
        prev_seq = unfolded_prev.permute(0, 1, 3, 2).reshape(
            B, max_blocks, PREV_K * self.angles_per_block
        )
        
        # 5. Combine all features
        feats = torch.cat([
            counts.unsqueeze(-1),      # [B, T, 1]
            cum.unsqueeze(-1),         # [B, T, 1]
            idx_seq.unsqueeze(-1),     # [B, T, 1]
            prev_seq,                  # [B, T, K*A]
        ], dim=-1)
        
        return feats  # [B, T, feat_dim]

    def forward(self, batch, device: torch.device) -> torch.Tensor:
        """
        Efficient, parallel, teacher-forced forward pass for TRAINING.
        
        Args:
            batch: Batch of circuits with target_angles
            device: torch device
                
        Returns:
            predicted_angles: [B, T*A, 1] where T=max_blocks, A=angles_per_block
        """
        B = batch.base_g.size(0)
        Lb_max = int(batch.base_len.max().item())
        max_blocks = math.ceil(Lb_max / max(1, self.gate_blocks))
        
        # Reshape targets from [B, T*A, 1] to [B, T, A]
        try:
            target_angles_T_A = batch.target_angles.view(
                B, max_blocks, self.angles_per_block
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"target_angles shape mismatch. Expected shape for "
                f"B={B}, T={max_blocks}, A={self.angles_per_block} "
                f"but got {batch.target_angles.shape}. "
                "Ensure batch.target_angles is shaped correctly."
            ) from e

        # 1. Get all features in parallel
        feats = self._get_vectorized_features(
            batch, max_blocks, target_angles_T_A, device
        )  # [B, T, feat_dim]
        
        # 2. Get cached causal attention mask
        attn_mask = self._get_causal_mask(max_blocks, device)
        
        # 3. Single, efficient transformer pass
        x = self.in_proj(feats)
        x = x + self.pos_emb(torch.arange(max_blocks, device=device)).unsqueeze(0)
        
        h = self.encoder(x, mask=attn_mask)  # [B, T, HID_DIM]
        
        # 4. Get logits for ALL time steps
        logits = self.head(self.head_ln(h))  # [B, T, 2*A]
        
        # 5. Convert all logits to all angles
        Y = self._angles_from_s1(logits)  # [B, T, A]
        
        # 6. Reshape to match expected format
        return Y.reshape(B, max_blocks * self.angles_per_block, 1)

    @torch.no_grad()
    def generate(self, batch, device: torch.device) -> torch.Tensor:
        """
        Autoregressive generation loop for INFERENCE.
        
        NOTE: O(T³) without KV caching. Consider implementing KV cache
        for O(T²) inference.
        
        Args:
            batch: Batch of circuits
            device: torch device
            
        Returns:
            predicted_angles: [B, T*A, 1] where T=max_blocks, A=angles_per_block
        """
        self.eval()
        
        B = batch.base_g.size(0)
        Lb_max = int(batch.base_len.max().item())
        max_blocks = math.ceil(Lb_max / max(1, self.gate_blocks))
        
        # Vectorized gate counts [B, T]
        block_indices = torch.arange(max_blocks, device=device).unsqueeze(0)
        block_starts = block_indices * self.gate_blocks
        block_ends = (block_indices + 1) * self.gate_blocks
        Lb_expanded = batch.base_len.unsqueeze(1).float()
        clipped_ends = torch.min(block_ends, Lb_expanded)
        counts = torch.clamp(clipped_ends - block_starts, min=0.0)  # [B, T]
        
        # Cumulative counts
        cum = counts.cumsum(dim=1)
        
        # Block indices
        idx_seq = block_indices.expand(B, -1).float()
        
        # Previous angles buffer (autoregressive)
        prev_buf = torch.zeros(B, PREV_K, self.angles_per_block, device=device)
        prev_seq = torch.zeros(B, max_blocks, self.angles_per_block * PREV_K, device=device)
        
        # Outputs
        Y = torch.zeros(B, max_blocks, self.angles_per_block, device=device)
        
        # Pre-calculate position embeddings
        pos_embeddings = self.pos_emb(torch.arange(max_blocks, device=device))
        
        # Autoregressive loop over blocks
        for t in range(max_blocks):
            L = t + 1  # Current sequence length
            
            # Store current prev window
            prev_seq[:, t, :] = prev_buf.reshape(B, self.angles_per_block * PREV_K)
            
            # Build features for blocks [0, t]
            feats = torch.cat([
                counts[:, :L].unsqueeze(-1),      # [B, L, 1]
                cum[:, :L].unsqueeze(-1),         # [B, L, 1]
                idx_seq[:, :L].unsqueeze(-1),     # [B, L, 1]
                prev_seq[:, :L, :],               # [B, L, K*A]
            ], dim=-1)
            
            # Project and encode
            x = self.in_proj(feats)
            x = x + pos_embeddings[:L].unsqueeze(0)
            
            # Apply causal transformer with cached mask
            attn_mask = self._get_causal_mask(L, device)
            h = self.encoder(x, mask=attn_mask)  # [B, L, HID_DIM]
            
            # Predict angles from last position only
            h_last = self.head_ln(h[:, -1, :])  # [B, HID_DIM]
            logits_t = self.head(h_last)  # [B, 2*A]
            
            # Convert to angles (already handles NaN/inf)
            y_t = self._angles_from_s1(logits_t)  # [B, A]
            
            # Store
            Y[:, t, :] = y_t
            
            # Update prev buffer for next step
            prev_buf = torch.roll(prev_buf, shifts=-1, dims=1)
            prev_buf[:, -1, :] = y_t
        
        # Reshape to [B, T*A, 1]
        return Y.reshape(B, max_blocks * self.angles_per_block, 1)