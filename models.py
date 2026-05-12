from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.activation(x + residual)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool) -> None:
        super().__init__()
        if downsample:
            self.proj = ConvNormAct(in_channels, out_channels, stride=2)
        else:
            self.proj = ConvNormAct(in_channels, out_channels)
        self.blocks = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.blocks(x)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = ConvNormAct(in_channels + skip_channels, out_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        return self.blocks(x)


class SubBandRefinement(nn.Module):
    def __init__(self, channels: int, num_bands: int = 4, blocks_per_band: int = 2) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.band_blocks = nn.ModuleList(
            [
                nn.Sequential(*[ResidualBlock(channels, channels) for _ in range(blocks_per_band)])
                for _ in range(num_bands)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(x, self.num_bands, dim=2)
        refined_chunks = [block(chunk) for block, chunk in zip(self.band_blocks, chunks)]
        return torch.cat(refined_chunks, dim=2)


class UNetRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 64,
        num_subbands: int = 4,
        use_subband_branch: bool = True,
    ) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        self.enc1 = EncoderStage(in_channels, widths[0], downsample=False)
        self.enc2 = EncoderStage(widths[0], widths[1], downsample=True)
        self.enc3 = EncoderStage(widths[1], widths[2], downsample=True)
        self.enc4 = EncoderStage(widths[2], widths[3], downsample=True)
        self.bottleneck = nn.Sequential(
            ResidualBlock(widths[3], widths[3]),
            ResidualBlock(widths[3], widths[3]),
        )

        self.dec3 = DecoderStage(widths[3], widths[2], widths[2])
        self.dec2 = DecoderStage(widths[2], widths[1], widths[1])
        self.dec1 = DecoderStage(widths[1], widths[0], widths[0])

        self.use_subband_branch = use_subband_branch
        if use_subband_branch:
            self.subband = SubBandRefinement(widths[0], num_bands=num_subbands)
            self.fuse = nn.Sequential(
                ConvNormAct(widths[0] * 2, widths[0], kernel_size=1),
                ResidualBlock(widths[0], widths[0]),
            )

        self.head = nn.Sequential(
            ResidualBlock(widths[0], widths[0]),
            nn.Conv2d(widths[0], out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        x = self.enc4(skip3)
        x = self.bottleneck(x)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        if self.use_subband_branch:
            local_detail = self.subband(x)
            x = self.fuse(torch.cat([x, local_detail], dim=1))

        return self.head(x)


class ProgressiveMelRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_subbands: int = 4,
    ) -> None:
        super().__init__()
        self.stage1 = UNetRefiner(
            in_channels=in_channels,
            out_channels=1,
            base_channels=max(base_channels // 2, 32),
            num_subbands=num_subbands,
            use_subband_branch=False,
        )
        self.stage2 = UNetRefiner(
            in_channels=in_channels + 1,
            out_channels=1,
            base_channels=base_channels,
            num_subbands=num_subbands,
            use_subband_branch=True,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        low = x[:, :1]
        x_half = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        coarse_residual_half = self.stage1(x_half)
        coarse_residual = F.interpolate(
            coarse_residual_half,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        coarse_prediction = low + coarse_residual

        stage2_input = torch.cat([x, coarse_prediction], dim=1)
        detail_residual = self.stage2(stage2_input)
        final_residual = coarse_residual + detail_residual
        prediction = low + final_residual

        return {
            "coarse_prediction": coarse_prediction,
            "coarse_residual": coarse_residual,
            "detail_residual": detail_residual,
            "residual": final_residual,
            "prediction": prediction,
        }
