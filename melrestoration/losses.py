from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import ssim_metric


def _frequency_weights(reference: torch.Tensor, boost: float) -> torch.Tensor:
    freq_bins = reference.shape[-2]
    weights = torch.linspace(
        1.0,
        boost,
        steps=freq_bins,
        device=reference.device,
        dtype=reference.dtype,
    )
    return weights.view(1, 1, freq_bins, 1)


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor, detail_freq_boost: float) -> torch.Tensor:
    weights = _frequency_weights(prediction, detail_freq_boost)

    pred_time = prediction[..., 1:] - prediction[..., :-1]
    target_time = target[..., 1:] - target[..., :-1]
    time_loss = torch.mean(torch.abs(pred_time - target_time) * weights)

    pred_freq = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_freq = target[:, :, 1:, :] - target[:, :, :-1, :]
    freq_weights = 0.5 * (weights[:, :, 1:, :] + weights[:, :, :-1, :])
    freq_loss = torch.mean(torch.abs(pred_freq - target_freq) * freq_weights)

    return 0.5 * (time_loss + freq_loss)


def high_frequency_loss(prediction: torch.Tensor, target: torch.Tensor, detail_freq_boost: float) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=prediction.dtype,
        device=prediction.device,
    ).view(1, 1, 3, 3)
    pred_hf = F.conv2d(prediction, kernel, padding=1)
    target_hf = F.conv2d(target, kernel, padding=1)
    weights = _frequency_weights(prediction, detail_freq_boost)
    return torch.mean(torch.abs(pred_hf - target_hf) * weights)


class CompositeMelLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        grad_weight: float = 0.3,
        hf_weight: float = 0.2,
        ssim_weight: float = 0.1,
        detail_freq_boost: float = 1.5,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
        self.hf_weight = hf_weight
        self.ssim_weight = ssim_weight
        self.detail_freq_boost = detail_freq_boost

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l1 = F.l1_loss(prediction, target)
        grad = gradient_loss(prediction, target, self.detail_freq_boost)
        hf = high_frequency_loss(prediction, target, self.detail_freq_boost)
        ssim = 1.0 - ssim_metric(prediction, target)

        total = (
            self.l1_weight * l1
            + self.grad_weight * grad
            + self.hf_weight * hf
            + self.ssim_weight * ssim
        )
        components = {
            "l1": l1.detach(),
            "grad": grad.detach(),
            "hf": hf.detach(),
            "ssim_loss": ssim.detach(),
            "total": total.detach(),
        }
        return total, components
