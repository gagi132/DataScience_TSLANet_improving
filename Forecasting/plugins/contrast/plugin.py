"""
Multi-task Contrastive Pretraining Plugin for TSLANet.

Implements three contrastive objectives on top of the TSLANet encoder:

  1. Contextual  – two random sub-window crops of the same series
  2. Temporal    – original series vs. a time-shifted version
  3. Augmentation– two independently augmented copies (jitter, scaling, …)

수정 내역:
----------
BUG 8 수정: temperature warmup 연동
  - ContrastivePlugin 이 TemperatureScheduler 를 내장.
  - 매 forward() 호출마다 temperature 를 갱신.
  - training=True 일 때만 step 진행, eval 시에는 final_temp 고정 사용.

BUG aggregator 수정:
  - ContrastiveLossAggregator 의 alpha/beta/gamma 가 nn.Parameter 로 변경됨.
  - configure_optimizers 에서 aggregator.parameters() 도 등록 필요.
  - → TSLANet_Forecasting.py 의 configure_optimizers 수정 필요 (아래 설명 참고).

TSLANet_Forecasting.py configure_optimizers 수정 사항:
  기존:
    params += list(self.contrast_plugin.proj_head.parameters())
  수정:
    params += list(self.contrast_plugin.proj_head.parameters())
    params += list(self.contrast_plugin.aggregator.parameters())  ← 추가

Performance note (encode() 호출 횟수):
  - z_orig 는 temporal task 와 공유 → 5회 (기존 6회에서 유지)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from plugins.contrast.augmentations import (
    time_shift, contextual_crop, apply_augmentation
)
from plugins.contrast.proj import ProjectionHead
from plugins.contrast.losses import (
    info_nce_loss, ContrastiveLossAggregator, TemperatureScheduler
)


class ContrastivePlugin(nn.Module):
    """
    Plug-and-play contrastive pretraining wrapper for TSLANet.

    Args:
        encoder          : TSLANet instance (must expose an `encode()` method).
        emb_dim          : Embedding dimension output by the encoder.
        proj_dim         : Output dimension of the projection head (default 128).
        temperature      : InfoNCE 목표 temperature τ (default 0.2).
        init_temperature : Warmup 시작 temperature (default 0.5, 높을수록 안전).
        warmup_steps     : Warmup 에 걸리는 step 수 (default 500).
        context_weight   : Weight α for contextual contrastive loss.
        temporal_weight  : Weight β for temporal contrastive loss.
        aug_weight       : Weight γ for augmentation contrastive loss.
        aug_list         : List of augmentation names applied in the aug task.
        crop_ratio       : Fraction of the sequence kept in contextual crops.
        shift_ratio      : Fraction of sequence length used as time-shift delta.
    """

    def __init__(
        self,
        encoder: nn.Module,
        emb_dim: int,
        proj_dim: int = 128,
        temperature: float = 0.2,
        init_temperature: float = 0.5,
        warmup_steps: int = 500,
        context_weight: float = 1.0,
        temporal_weight: float = 1.0,
        aug_weight: float = 1.0,
        aug_list: list[str] | None = None,
        crop_ratio: float = 0.75,
        shift_ratio: float = 0.1,
    ):
        super().__init__()
        self.encoder     = encoder
        self.proj_head   = ProjectionHead(input_dim=emb_dim, proj_dim=proj_dim)
        self.aggregator  = ContrastiveLossAggregator(
            context_weight=context_weight,
            temporal_weight=temporal_weight,
            aug_weight=aug_weight,
        )
        # BUG 8 수정: temperature warmup 스케줄러
        self.temp_scheduler = TemperatureScheduler(
            init_temp=init_temperature,
            final_temp=temperature,
            warmup_steps=warmup_steps,
        )
        self.aug_list   = aug_list if aug_list is not None else ["jitter", "scaling"]
        self.crop_ratio = crop_ratio
        self.shift_ratio = shift_ratio

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Encode x → project to contrastive space → [B, proj_dim]."""
        h = self.encoder.encode(x)   # [B, emb_dim]
        z = self.proj_head(h)        # [B, proj_dim]
        return z

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the combined multi-task contrastive loss.

        Args:
            x        : [B, L, M] float tensor (raw time-series batch)
            training : True 면 temperature warmup step 진행,
                       False (val/test) 면 final_temp 고정 사용.

        Returns:
            total_loss : scalar tensor (back-propagatable)
            loss_dict  : {'context': ..., 'temporal': ..., 'aug': ...,
                          'temperature': ...}
        """
        # BUG 8 수정: temperature 갱신
        if training:
            tau = self.temp_scheduler.step()
        else:
            tau = self.temp_scheduler.final_temp

        # ── shared original embedding (temporal task 와 공유) ─────────────────
        z_orig = self._embed(x)

        # ── contextual: two independent random crops ──────────────────────────
        z_c1 = self._embed(contextual_crop(x, crop_ratio=self.crop_ratio))
        z_c2 = self._embed(contextual_crop(x, crop_ratio=self.crop_ratio))
        l_context = info_nce_loss(z_c1, z_c2, tau)

        # ── temporal: original vs. time-shifted ───────────────────────────────
        z_shift    = self._embed(time_shift(x, delta_ratio=self.shift_ratio))
        l_temporal = info_nce_loss(z_orig, z_shift, tau)

        # ── augmentation: two independently augmented copies ──────────────────
        z_a1  = self._embed(apply_augmentation(x.clone(), self.aug_list))
        z_a2  = self._embed(apply_augmentation(x.clone(), self.aug_list))
        l_aug = info_nce_loss(z_a1, z_a2, tau)

        total_loss = self.aggregator(l_context, l_temporal, l_aug)

        loss_dict = {
            "context":     l_context.detach(),
            "temporal":    l_temporal.detach(),
            "aug":         l_aug.detach(),
            "temperature": torch.tensor(tau),   # 로깅용
        }
        return total_loss, loss_dict
