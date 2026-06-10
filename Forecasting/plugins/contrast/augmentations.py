"""
Time series augmentation functions for contrastive pretraining.

Each function takes x: [B, L, M] (batch, time, variates) and returns
an augmented tensor of the same shape.

v2 수정사항
-----------
contextual_crop: F.interpolate 대신 단순 슬라이싱 + 반복 패딩으로 교체.
  기존: permute → (B, M, crop_len) → F.interpolate(mode='linear') → permute
        METRLA(M=207)에서 (16, 207, 384) 텐서를 CPU linear interpolation
        → M 채널이 많을수록 선형 비례로 느려짐 (ETTh1 대비 ~29×)

  수정: 슬라이싱만 사용, 길이를 맞추기 위해 repeat + 잘라내기 방식
        CPU 연산 없이 index 연산만 → GPU에서 처리 가능
        shape 유지: (B, L, M) 동일
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Individual augmentations
# ─────────────────────────────────────────────────────────────────────────────

def jitter(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    """Add zero-mean Gaussian noise (σ controls magnitude)."""
    return x + torch.randn_like(x) * sigma


def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Per-sample random amplitude scaling."""
    factor = 1.0 + torch.randn(x.shape[0], 1, 1, device=x.device) * sigma
    return x * factor


def time_shift(x: torch.Tensor, delta_ratio: float = 0.1) -> torch.Tensor:
    """
    Circularly shift the time axis by a random fraction of the sequence length.
    """
    L     = x.shape[1]
    delta = max(1, int(L * delta_ratio))
    shifted = torch.roll(x, shifts=delta, dims=1)
    shifted[:, :delta, :] = x[:, :1, :].expand(-1, delta, -1)
    return shifted


def contextual_crop(x: torch.Tensor, crop_ratio: float = 0.75) -> torch.Tensor:
    """
    Randomly crop a contiguous sub-sequence and tile-repeat back to length L.

    기존 F.interpolate(mode='linear') 방식 대체:
      - 기존: permute(B,M,L) → CPU linear interp → permute (M 채널수에 비례해 느림)
      - 수정: slicing + repeat → index 연산만 사용 (GPU에서 처리, 채널 수 무관)

    Args:
        x          : [B, L, M]
        crop_ratio : fraction of L to keep (0 < crop_ratio < 1)

    Returns:
        [B, L, M] — same shape as input
    """
    B, L, M   = x.shape
    crop_len  = max(2, int(L * crop_ratio))
    max_start = L - crop_len
    start     = torch.randint(0, max_start + 1, (1,)).item()

    cropped = x[:, start: start + crop_len, :]          # [B, crop_len, M]

    # tile-repeat: crop을 반복해 L 이상 만든 뒤 잘라냄
    # ceil(L / crop_len) 번 반복 → [B, reps*crop_len, M] → [:, :L, :]
    reps   = -(-L // crop_len)                          # ceil division
    tiled  = cropped.repeat(1, reps, 1)                 # [B, reps*crop_len, M]
    return tiled[:, :L, :].contiguous()                 # [B, L, M]


def dropout_timesteps(x: torch.Tensor, drop_ratio: float = 0.1) -> torch.Tensor:
    """Randomly zero-out a fraction of time steps."""
    B, L, M = x.shape
    mask    = (torch.rand(B, L, 1, device=x.device) > drop_ratio).float()
    return x * mask


# ─────────────────────────────────────────────────────────────────────────────
# Registry & apply helper
# ─────────────────────────────────────────────────────────────────────────────

AUGMENTATION_DICT = {
    "jitter":           jitter,
    "scaling":          scaling,
    "time_shift":       time_shift,
    "contextual_crop":  contextual_crop,
    "dropout":          dropout_timesteps,
}


def apply_augmentation(x: torch.Tensor, aug_list) -> torch.Tensor:
    """
    Sequentially apply a list of named augmentations to x.

    Args:
        x        : [B, L, M] float tensor
        aug_list : list of strings matching keys in AUGMENTATION_DICT
    Returns:
        augmented tensor, same shape as x
    """
    for name in aug_list:
        fn = AUGMENTATION_DICT.get(name)
        if fn is None:
            raise ValueError(f"Unknown augmentation '{name}'. "
                             f"Available: {list(AUGMENTATION_DICT.keys())}")
        x = fn(x)
    return x
