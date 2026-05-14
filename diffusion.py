from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _extract(values: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size) -> torch.Tensor:
    gathered = values.gather(0, timesteps)
    return gathered.view(timesteps.shape[0], *((1,) * (len(broadcast_shape) - 1)))


def make_beta_schedule(
    timesteps: int,
    schedule: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    if timesteps < 2:
        raise ValueError("--timesteps must be at least 2.")

    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    if schedule == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5).pow(2)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-5, 0.999).float()

    raise ValueError(f"Unsupported beta schedule: {schedule}")


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        betas = make_beta_schedule(timesteps, beta_schedule, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt((1.0 / alphas_cumprod) - 1.0))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def config(self) -> dict[str, float | int | str]:
        return {
            "timesteps": self.timesteps,
            "beta_schedule": self.beta_schedule,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
        }

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_x_start_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            _extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
        )

    def predict_noise_from_x_start(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        x_start: torch.Tensor,
    ) -> torch.Tensor:
        return (
            x_t - _extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape) * x_start
        ) / _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)

    def q_sample_from_optional_start(
        self,
        initial_x: torch.Tensor | None,
        shape: tuple[int, int, int, int],
        start_timestep: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if initial_x is None:
            return torch.randn(shape, device=device, dtype=dtype)
        if initial_x.shape != shape:
            raise ValueError(f"Expected initial_x shape {shape}, got {tuple(initial_x.shape)}.")
        x_start = initial_x.to(device=device, dtype=dtype)
        if start_timestep <= 0:
            return x_start
        timesteps = torch.full((shape[0],), start_timestep, device=device, dtype=torch.long)
        return self.q_sample(x_start, timesteps, torch.randn_like(x_start))

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
        x0_clip: float | None = None,
    ) -> torch.Tensor:
        predicted_noise = model(x_t, condition, timesteps)
        predicted_x0 = self.predict_x_start_from_noise(x_t, timesteps, predicted_noise)
        if x0_clip is not None and x0_clip > 0:
            predicted_x0 = predicted_x0.clamp(-x0_clip, x0_clip)
        model_mean = (
            _extract(self.posterior_mean_coef1, timesteps, x_t.shape) * predicted_x0
            + _extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        posterior_variance_t = _extract(self.posterior_variance, timesteps, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = (timesteps != 0).float().view(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_ddpm(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        shape: tuple[int, int, int, int],
        initial_x: torch.Tensor | None = None,
        start_timestep: int | None = None,
        x0_clip: float | None = None,
    ) -> torch.Tensor:
        start_timestep = self.timesteps - 1 if start_timestep is None else int(start_timestep)
        start_timestep = max(0, min(start_timestep, self.timesteps - 1))
        x_t = self.q_sample_from_optional_start(
            initial_x,
            shape,
            start_timestep,
            device=condition.device,
            dtype=condition.dtype,
        )
        for step in reversed(range(start_timestep + 1)):
            timesteps = torch.full((shape[0],), step, device=condition.device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, condition, timesteps, x0_clip=x0_clip)
        return x_t

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        shape: tuple[int, int, int, int],
        steps: int = 50,
        eta: float = 0.0,
        initial_x: torch.Tensor | None = None,
        start_timestep: int | None = None,
        x0_clip: float | None = None,
    ) -> torch.Tensor:
        if steps < 2:
            raise ValueError("--sample-steps must be at least 2.")

        start_timestep = self.timesteps - 1 if start_timestep is None else int(start_timestep)
        start_timestep = max(0, min(start_timestep, self.timesteps - 1))
        steps = min(steps, start_timestep + 1)
        x_t = self.q_sample_from_optional_start(
            initial_x,
            shape,
            start_timestep,
            device=condition.device,
            dtype=condition.dtype,
        )
        if start_timestep == 0:
            return x_t

        sequence = torch.linspace(start_timestep, 0, steps, device=condition.device).long().unique_consecutive()

        for index, step in enumerate(sequence):
            timesteps = torch.full((shape[0],), int(step.item()), device=condition.device, dtype=torch.long)
            predicted_noise = model(x_t, condition, timesteps)
            predicted_x0 = self.predict_x_start_from_noise(x_t, timesteps, predicted_noise)
            if x0_clip is not None and x0_clip > 0:
                predicted_x0 = predicted_x0.clamp(-x0_clip, x0_clip)
                predicted_noise = self.predict_noise_from_x_start(x_t, timesteps, predicted_x0)

            if index == len(sequence) - 1:
                x_t = predicted_x0
                continue

            next_step = int(sequence[index + 1].item())
            alpha_t = self.alphas_cumprod[step].view(1, 1, 1, 1)
            alpha_next = self.alphas_cumprod[next_step].view(1, 1, 1, 1)
            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
            direction = torch.sqrt(torch.clamp(1 - alpha_next - sigma.pow(2), min=0.0)) * predicted_noise
            x_t = torch.sqrt(alpha_next) * predicted_x0 + direction
            if eta > 0:
                x_t = x_t + sigma * torch.randn_like(x_t)

        return x_t


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.channels // 2
        if half == 0:
            return timesteps.float().view(-1, 1)

        exponent = -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        exponent = exponent / max(half - 1, 1)
        frequencies = torch.exp(exponent)
        embedding = timesteps.float().view(-1, 1) * frequencies.view(1, -1)
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
        if self.channels % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class TimeResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_channels, out_channels * 2)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_projection(F.silu(time_embedding)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + residual


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(batch, channels, height * width).transpose(1, 2)
        k = k.reshape(batch, channels, height * width)
        v = v.reshape(batch, channels, height * width).transpose(1, 2)

        attention = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        h = torch.bmm(attention, v).transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class ConditionalDiffusionUNet(nn.Module):
    def __init__(
        self,
        cond_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 4),
        time_channels: int | None = None,
        dropout: float = 0.05,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if not channel_mults:
            raise ValueError("channel_mults must contain at least one multiplier.")

        widths = [base_channels * int(multiplier) for multiplier in channel_mults]
        time_channels = time_channels or base_channels * 4

        self.cond_channels = cond_channels
        self.base_channels = base_channels
        self.channel_mults = tuple(int(multiplier) for multiplier in channel_mults)
        self.time_channels = time_channels
        self.dropout = dropout
        self.use_attention = use_attention

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )

        self.input_projection = nn.Conv2d(cond_channels + 1, widths[0], kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = widths[0]
        for index, width in enumerate(widths):
            self.encoders.append(
                nn.ModuleList(
                    [
                        TimeResidualBlock(current_channels, width, time_channels, dropout),
                        TimeResidualBlock(width, width, time_channels, dropout),
                    ]
                )
            )
            current_channels = width
            if index != len(widths) - 1:
                self.downsamples.append(Downsample(current_channels))

        self.mid1 = TimeResidualBlock(widths[-1], widths[-1], time_channels, dropout)
        self.mid_attention = SpatialSelfAttention(widths[-1]) if use_attention else nn.Identity()
        self.mid2 = TimeResidualBlock(widths[-1], widths[-1], time_channels, dropout)

        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        current_channels = widths[-1]
        for level in reversed(range(len(widths))):
            if level != len(widths) - 1:
                self.upsamples.append(Upsample(current_channels))
            skip_channels = widths[level]
            self.decoders.append(
                nn.ModuleList(
                    [
                        TimeResidualBlock(current_channels + skip_channels, widths[level], time_channels, dropout),
                        TimeResidualBlock(widths[level], widths[level], time_channels, dropout),
                    ]
                )
            )
            current_channels = widths[level]

        self.output_norm = nn.GroupNorm(_group_count(widths[0]), widths[0])
        self.output_projection = nn.Conv2d(widths[0], 1, kernel_size=3, padding=1)

    def config(self) -> dict[str, object]:
        return {
            "cond_channels": self.cond_channels,
            "base_channels": self.base_channels,
            "channel_mults": list(self.channel_mults),
            "time_channels": self.time_channels,
            "dropout": self.dropout,
            "use_attention": self.use_attention,
        }

    def forward(
        self,
        noisy: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)
        x = self.input_projection(torch.cat([noisy, condition], dim=1))

        skips: list[torch.Tensor] = []
        for index, blocks in enumerate(self.encoders):
            for block in blocks:
                x = block(x, time_embedding)
            skips.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)

        x = self.mid1(x, time_embedding)
        x = self.mid_attention(x)
        x = self.mid2(x, time_embedding)

        upsample_index = 0
        for level, blocks in zip(reversed(range(len(self.channel_mults))), self.decoders):
            if level != len(self.channel_mults) - 1:
                x = self.upsamples[upsample_index](x)
                upsample_index += 1

            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, time_embedding)

        return self.output_projection(F.silu(self.output_norm(x)))
