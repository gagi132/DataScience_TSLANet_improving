"""
plugins/timae/linear_decoder.py
================================

Strategy A: Lightweight Linear Decoder for Ti-MAE
---------------------------------------------------

보고서 Section III.A 에서 제안한 구조입니다.

Transformer 디코더가 CNN backbone 에 비해 과도하게 복잡하여
보간(interpolation) 에 과적합된다는 문제를 해결하기 위해
디코더를 단일 선형 레이어로 교체합니다.

L_recon = (1 / |M|) * sum_{i in M} ||W * h_i + b - x_i||^2

여기서:
  - M         : 마스킹된 패치 인덱스 집합
  - h_i       : 인코더가 출력한 패치 임베딩 (dim = emb_dim)
  - W, b      : 선형 레이어 가중치
  - x_i       : 원본 패치 값 (dim = patch_size)

특성:
  1. 인코더 용량 불균형 해소: 선형 디코더는 local reconstruction 에
     과적합할 능력이 없으므로, 모든 representational burden 이
     인코더에 집중됩니다.
  2. Visible-only encoding 유지: shortcut 학습을 구조적으로 방지.
  3. 파라미터 수 대폭 감소: Transformer 디코더 대비 ~10x 절감.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple, Dict
from utils import random_masking_3D


class TiMAELinearPlugin(nn.Module):
    """
    Ti-MAE with lightweight single-layer linear decoder.

    Encoder : TSLANet (visible patches only, same as TiMAEPlugin)
    Decoder : Linear(emb_dim → patch_size)  ← Strategy A 핵심 변경점

    Args:
        encoder       : TSLANet model instance (shared reference)
        patch_size    : P — raw timesteps per patch
        stride        : patch stride (= patch_size // 2)
        seq_len       : input sequence length L
        emb_dim       : encoder embedding dimension d
        mask_ratio    : fraction of patches to mask (default 0.75)
        norm_pix_loss : normalize each patch before MSE (He et al. 2022)
    """

    def __init__(
        self,
        encoder: nn.Module,
        patch_size: int,
        stride: int,
        seq_len: int,
        emb_dim: int,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.encoder       = encoder
        self.mask_ratio    = mask_ratio
        self.patch_size    = patch_size
        self.stride        = stride
        self.norm_pix_loss = norm_pix_loss

        self.num_patches = int((seq_len - patch_size) / stride + 1)

        # ── Strategy A: single linear layer instead of Transformer decoder ──
        self.decoder = nn.Linear(emb_dim, patch_size)

    # ─────────────────────────────────────────────────────────────────────────

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, M] → [B*M, N_p, P]  with RevIN normalisation."""
        B, L, M = x.shape
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = x / stdev

        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        return x

    def _encode_visible(
        self,
        x_patched: torch.Tensor,
        x_kept: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode ONLY the visible (unmasked) patches.

        Bug fix vs. original TiMAEPlugin:
          random_masking_3D already returns x_kept (correctly gathered).
          We use it directly, avoiding the ids_keep reconstruction bug.

        Args:
            x_patched : [BM, N_p, P]  — all patches (unused here, kept for API compat)
            x_kept    : [BM, len_keep, P]  — visible patches from random_masking_3D

        Returns:
            [BM, len_keep, d]
        """
        emb = self.encoder.input_layer(x_kept)   # [BM, len_keep, d]
        for blk in self.encoder.tsla_blocks:
            emb = blk(emb)
        return emb                                # [BM, len_keep, d]

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Strategy A pretraining loss.

        Args:
            x : [B, L, M] raw batch

        Returns:
            loss      : scalar (masked MSE with linear decoder)
            loss_dict : {'loss_recon': ..., 'mask_ratio': ...}
        """
        x_patched = self._patchify(x)               # [BM, N_p, P]
        BM, N_p, P = x_patched.shape

        # ── random masking (uses x_kept directly — bug fix) ───────────────
        _, x_kept, mask, ids_restore = random_masking_3D(
            x_patched, mask_ratio=self.mask_ratio
        )
        # mask : [BM, N_p]  (0=keep, 1=masked)

        # ── encode visible patches ────────────────────────────────────────
        vis_emb = self._encode_visible(x_patched, x_kept)  # [BM, len_keep, d]

        # ── linear decoder: project each visible token to patch space ─────
        #    We need to reconstruct ALL patches including masked ones.
        #    For masked positions we insert the mean encoder output as
        #    a lightweight stand-in (no learned mask token needed).
        BM2, len_keep, d = vis_emb.shape
        n_masked = N_p - len_keep

        # mean-pool visible tokens → surrogate for masked positions
        surrogate = vis_emb.mean(dim=1, keepdim=True).expand(-1, n_masked, -1)
        full_emb  = torch.cat([vis_emb, surrogate], dim=1)     # [BM, N_p, d]

        # unshuffle to original patch order
        idx_restore = ids_restore.unsqueeze(-1).expand(-1, -1, d)
        full_emb    = torch.gather(full_emb, dim=1, index=idx_restore)

        # linear projection
        pred   = self.decoder(full_emb)              # [BM, N_p, P]

        # ── reconstruction target ─────────────────────────────────────────
        target = x_patched.clone()
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var( dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / (var + 1e-6).sqrt()

        # MSE on masked positions only
        mask_bool   = mask.bool()
        loss_per    = ((pred - target) ** 2).mean(dim=-1)       # [BM, N_p]
        loss        = (loss_per * mask_bool.float()).sum() \
                    / mask_bool.float().sum()

        loss_dict = {
            "loss_recon": loss.detach(),
            "mask_ratio": torch.tensor(self.mask_ratio, device=x.device),
            "decoder":    "linear",
        }
        return loss, loss_dict
