import torch
import torch.nn as nn
from einops import rearrange

from utils import random_masking_3D
from plugins.timae.decoder import TiMAEDecoder


class TiMAEPlugin(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        patch_size: int,
        stride: int,
        seq_len: int,
        emb_dim: int,
        mask_ratio: float = 0.4,
        decoder_dim: int = 64,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        decoder_mlp_ratio: float = 4.0,
        decoder_dropout: float = 0.0,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.stride = stride
        self.norm_pix_loss = norm_pix_loss

        self.num_patches = int((seq_len - patch_size) / stride + 1)

        self.decoder = TiMAEDecoder(
            encoder_dim=emb_dim,
            decoder_dim=decoder_dim,
            patch_size=patch_size,
            num_patches=self.num_patches,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=decoder_mlp_ratio,
            dropout=decoder_dropout,
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
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

    def _encode_visible(self, visible: torch.Tensor) -> torch.Tensor:
        # visible: [BM, len_keep, P]
        x = self.encoder.input_layer(visible)

        for blk in self.encoder.tsla_blocks:
            x = blk(x)

        return x

    def forward(self, x: torch.Tensor):
        x_patched = self._patchify(x)  # [BM, N_p, P]

        _, x_kept, mask, ids_restore = random_masking_3D(
            x_patched, mask_ratio=self.mask_ratio
        )

        # ── encoder ─────────────────────────────
        visible_emb = self._encode_visible(x_kept)

        # ── decoder ─────────────────────────────
        pred = self.decoder(visible_emb, ids_restore)

        # ── target ─────────────────────────────
        target = x_patched

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / (var + 1e-6).sqrt()

        # ── loss ───────────────────────────────
        mask = mask.bool()
        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss_per_patch * mask.float()).sum() / mask.float().sum()

        return loss, {
            "loss_recon": loss.detach(),
            "mask_ratio": torch.tensor(self.mask_ratio, device=x.device),
        }