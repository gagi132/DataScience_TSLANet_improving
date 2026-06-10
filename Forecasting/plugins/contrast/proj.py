"""
Projection Head for Contrastive Learning.

Maps encoder output embeddings into a lower-dimensional space where
the InfoNCE loss is applied.

수정 내역:
----------
BUG 7 수정: BatchNorm1d → LayerNorm
  BatchNorm1d 는 B=1 일 때 running_var 계산에서 NaN 발생.
  val/test 의 마지막 배치가 B=1 이면 crash 또는 NaN loss.
  LayerNorm 은 배치 크기에 무관하게 안정적으로 동작.

  추가 효과:
    - 모델을 eval() 로 전환해도 동작 안정 (BN running stats 불안정 문제 제거)
    - pretrain 초기 val loss 가 의미있는 값을 가져 ModelCheckpoint 정상 동작
"""

import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head.

    Architecture: Linear → LayerNorm → GELU → Linear
                  (L2-norm 은 losses.py 의 info_nce_loss 에서 수행)

    Args:
        input_dim  : Dimensionality of the encoder output.
        hidden_dim : Width of the hidden layer (defaults to input_dim).
        proj_dim   : Output projection dimensionality (e.g. 128).
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 128,
        hidden_dim: int = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # BUG 7 수정: BatchNorm1d → LayerNorm (B=1 NaN 방지)
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        """
        Args:
            x : [B, input_dim]
        Returns:
            z : [B, proj_dim]  (NOT L2-normalised)
        """
        return self.net(x)
