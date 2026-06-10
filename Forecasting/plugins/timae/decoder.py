"""
Ti-MAE Decoder — lightweight Transformer decoder for masked patch reconstruction.

Architecture follows the original MAE (He et al., 2022) design:
  - Mask token is a learned vector broadcast to all masked positions.
  - Positional embeddings are added before the decoder.
  - A single linear layer projects decoder output back to patch dimension.

Reference
---------
Li, Z. et al. "Ti-MAE: Self-Supervised Masked Time Series Autoencoders",
arXiv:2301.08871, 2023.
"""

import torch
import torch.nn as nn


class TiMAEDecoder(nn.Module):
    """
    Transformer-based decoder that reconstructs masked patches.

    Args:
        encoder_dim  : Dimensionality of encoder output embeddings (d).
        decoder_dim  : Internal width of the decoder Transformer.
        patch_size   : Number of raw time-steps per patch (reconstruction target dim).
        num_patches  : Total number of patches N_p.
        depth        : Number of Transformer decoder layers.
        num_heads    : Number of attention heads.
        mlp_ratio    : MLP hidden-dim ratio inside each Transformer block.
        dropout      : Dropout probability.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        patch_size: int,
        num_patches: int,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Project encoder embeddings into decoder space
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim, bias=False)

        # Learnable mask token (broadcast to all masked patch positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Fixed sinusoidal positional embedding (no gradient)
        self.register_buffer(
            "pos_embed",
            self._sinusoidal_embed(num_patches, decoder_dim),
        )

        # Transformer decoder blocks
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LayerNorm (more stable)
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)

        # Final projection back to patch space (reconstruction head)
        self.pred_head = nn.Linear(decoder_dim, patch_size)

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sinusoidal_embed(n: int, d: int) -> torch.Tensor:
        """Return [1, n, d] sinusoidal positional embeddings."""
        position = torch.arange(n).unsqueeze(1).float()          # [n, 1]
        div_term = torch.exp(
            torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(10000.0)) / d)
        )                                                         # [d/2]
        pe = torch.zeros(n, d)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)                                    # [1, n, d]

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        visible_tokens: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct all patches (visible + masked) from visible encoder tokens.

        Args:
            visible_tokens : [BM, len_keep, encoder_dim]  — kept patch embeddings
            ids_restore    : [BM, N_p]                    — indices to unshuffle

        Returns:
            pred : [BM, N_p, patch_size]  — reconstructed patches for all positions
        """
        BM, len_keep, _ = visible_tokens.shape
        N_p = ids_restore.shape[1]
        n_masked = N_p - len_keep

        # Project visible tokens to decoder dimension
        x = self.encoder_to_decoder(visible_tokens)              # [BM, len_keep, D_dec]

        # Expand mask token and concatenate
        mask_tokens = self.mask_token.expand(BM, n_masked, -1)   # [BM, n_masked, D_dec]
        x_full = torch.cat([x, mask_tokens], dim=1)              # [BM, N_p, D_dec]

        # Unshuffle to original patch order
        idx = ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1])
        x_full = torch.gather(x_full, dim=1, index=idx)          # [BM, N_p, D_dec]

        # Add positional embeddings
        x_full = x_full + self.pos_embed                         # [BM, N_p, D_dec]

        # Transformer decoder
        x_full = self.decoder(x_full)                            # [BM, N_p, D_dec]

        # Predict raw patch values
        pred = self.pred_head(x_full)                            # [BM, N_p, patch_size]
        return pred
