from __future__ import annotations

import torch
from torch.nn import functional as F


def _ssim_map(x: torch.Tensor, y: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    padding = window_size // 2

    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=padding)

    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=padding) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=padding) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=padding) - (mu_x * mu_y)

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    return numerator / (denominator + 1e-8)


def ssim_metric(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _ssim_map(prediction, target).mean()


def mae_metric(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target))


def lsd_metric(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    squared_error = torch.square(prediction - target)
    per_frame = torch.sqrt(torch.mean(squared_error, dim=-2) + 1e-8)
    return per_frame.mean()
