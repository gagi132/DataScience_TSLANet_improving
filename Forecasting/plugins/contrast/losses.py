"""
InfoNCE (NT-Xent) loss and weighted loss aggregator.

Reference: Oord et al., "Representation Learning with Contrastive
Predictive Coding", 2018.

수정 내역:
----------
BUG 8 수정: info_nce_loss
  - F.normalize 에 eps guard 추가 (zero vector → NaN 방지)
  - temperature warmup 지원: init_temperature + warmup_steps
    초기에 높은 temperature(0.5)에서 시작해 목표값(0.2)까지 선형 감소.
    이유: 학습 초반 encoder 가 아직 discriminative 하지 않아
          모든 embedding 이 유사 → low temperature 에서 gradient vanish.
          warmup 으로 초반 gradient 를 크게 유지하여 학습 시동.

BUG 추가 수정: ContrastiveLossAggregator
  - register_buffer → nn.Parameter 로 변경 (학습 가능한 가중치)
    고정 가중치를 원하면 requires_grad=False 로 설정.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Core InfoNCE loss
# ─────────────────────────────────────────────────────────────────────────────

def info_nce_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss between query and key embedding batches.

    Positive pairs : (q[i], k[i])  for every i in the batch.
    Negative pairs : (q[i], k[j])  for all j ≠ i (in-batch negatives).

    Args:
        q           : [B, D] query embeddings (raw, before normalisation)
        k           : [B, D] key embeddings  (raw, before normalisation)
        temperature : scalar τ

    Returns:
        Scalar loss value.
    """
    # BUG 8 수정: eps guard → zero vector 입력 시 NaN 방지
    q = F.normalize(q, dim=-1, eps=1e-8)
    k = F.normalize(k, dim=-1, eps=1e-8)

    B = q.shape[0]
    labels = torch.arange(B, device=q.device)

    sim_qk = torch.mm(q, k.T) / temperature     # [B, B]
    sim_kq = torch.mm(k, q.T) / temperature     # [B, B]

    loss = (F.cross_entropy(sim_qk, labels) + F.cross_entropy(sim_kq, labels)) / 2.0
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scheduler (warmup)
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScheduler:
    """
    BUG 8 수정: temperature linear warmup.

    학습 초반에 높은 temperature 로 시작해 목표값까지 선형 감소.

    이유:
      초기 encoder 는 discriminative 하지 않아 모든 embedding 이 유사.
      temperature 가 낮으면(0.2) softmax 가 포화되어 gradient 가 소멸.
      초반에 temperature 를 높게(0.5) 유지하면 gradient 가 살아있어
      encoder 가 학습을 시작할 수 있다.

    Args:
        init_temp   : 시작 temperature (높을수록 안전, 권장 0.5)
        final_temp  : 목표 temperature (논문 기본값 0.2)
        warmup_steps: warmup 에 걸리는 optimizer step 수
    """

    def __init__(
        self,
        init_temp: float = 0.5,
        final_temp: float = 0.2,
        warmup_steps: int = 500,
    ):
        self.init_temp    = init_temp
        self.final_temp   = final_temp
        self.warmup_steps = warmup_steps
        self._step        = 0

    def step(self) -> float:
        """한 step 진행하고 현재 temperature 반환."""
        self._step += 1
        if self._step >= self.warmup_steps:
            return self.final_temp
        ratio = self._step / self.warmup_steps
        return self.init_temp + (self.final_temp - self.init_temp) * ratio

    @property
    def current(self) -> float:
        if self._step >= self.warmup_steps:
            return self.final_temp
        ratio = self._step / self.warmup_steps
        return self.init_temp + (self.final_temp - self.init_temp) * ratio

    def reset(self):
        self._step = 0


# ─────────────────────────────────────────────────────────────────────────────
# Weighted aggregator
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveLossAggregator(nn.Module):
    """
    Combines three contrastive losses with learnable weights:

        L_total = α · L_context + β · L_temporal + γ · L_aug

    수정: register_buffer → nn.Parameter
      buffer 는 optimizer 에 등록되지 않아 학습되지 않는다.
      nn.Parameter 로 변경하면 가중치가 학습되어 태스크별 중요도를 자동 조정.
      고정 가중치를 원하면 초기화 후 requires_grad_(False) 호출.

    Args:
        context_weight  : α 초기값 (default 1.0)
        temporal_weight : β 초기값 (default 1.0)
        aug_weight      : γ 초기값 (default 1.0)
    """

    def __init__(
        self,
        context_weight: float = 1.0,
        temporal_weight: float = 1.0,
        aug_weight: float = 1.0,
    ):
        super().__init__()
        # nn.Parameter → optimizer 에 자동 등록, 학습 가능
        self.alpha = nn.Parameter(torch.tensor(context_weight))
        self.beta  = nn.Parameter(torch.tensor(temporal_weight))
        self.gamma = nn.Parameter(torch.tensor(aug_weight))

    def forward(
        self,
        l_context: torch.Tensor,
        l_temporal: torch.Tensor,
        l_aug: torch.Tensor,
    ) -> torch.Tensor:
        # softplus 로 가중치를 양수로 유지 (음수 가중치 방지)
        alpha = F.softplus(self.alpha)
        beta  = F.softplus(self.beta)
        gamma = F.softplus(self.gamma)
        return alpha * l_context + beta * l_temporal + gamma * l_aug
