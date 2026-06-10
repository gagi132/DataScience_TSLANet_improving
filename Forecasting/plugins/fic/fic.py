"""
FIC-TSC: Fisher Information Constraint for Time Series Forecasting
Reference: Chen et al., "FIC-TSC: Learning Time Series Classification with Fisher Information Constraint", ICML 2025.

This module enforces a flat-minima constraint by bounding the trace of the 
empirical Fisher Information Matrix (FIM). To circumvent the prohibitive 
O(N^2) memory and compute overhead of double backpropagation, this implementation 
strictly follows Equation (5) of the FIC-TSC paper:

    g <- g * sqrt(epsilon / ||F||_1)   if ||F||_1 >= epsilon

where ||F||_1 is the 1-norm of the FIM diagonal (i.e., squared L2 norm of gradients).
"""

import torch
import torch.nn as nn
import math

class FisherInformationConstraint(nn.Module):
    def __init__(self, model, lambda_fic=0.001, param_names=None):
        super().__init__()
        self.model = model
        # FIC-TSC 논문의 threshold (Upper bound) 인 epsilon
        self.epsilon = float(lambda_fic)
        self.param_names = param_names

        # 최적화 연산 속도를 높이기 위해 적용할 파라미터 리스트를 미리 캐싱
        self.target_params = []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if self.param_names is None or any(pn in name for pn in self.param_names):
                    self.target_params.append(p)

    def forward(self, task_loss):
        """
        Loss 값은 변경하지 않습니다 (Additive penalty 방식이 아님).
        """
        return task_loss, {"fic_epsilon": self.epsilon}

    def apply_after_backward(self):
        """
        단일 역전파(loss.backward()) 직후 호출되어 FIC-TSC 수식을 적용합니다.
        """
        if self.epsilon <= 0 or not self.target_params:
            return

        # 1. ||F||_1 (Fisher Trace) 계산: 기존 역전파된 1차 미분값만 사용 (O(1) overhead)
        fisher_trace = 0.0
        for p in self.target_params:
            if p.grad is not None:
                fisher_trace += p.grad.pow(2).sum()

        # 2. FIC-TSC 논문 Eq.(5): Trace 가 epsilon 을 초과할 경우 정규화(Constraint) 적용
        if fisher_trace >= self.epsilon:
            # scaling_factor = sqrt(epsilon / ||F||_1)
            scaling_factor = math.sqrt(self.epsilon / (fisher_trace.item() + 1e-12))
            
            # 3. 모델 그래디언트에 In-place 스케일링
            for p in self.target_params:
                if p.grad is not None:
                    p.grad.mul_(scaling_factor)