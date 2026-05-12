# melrestoration

Residual mel-spectrogram restoration on top of an RVQ-VAE output.

This repository contains second-stage PyTorch refiners that learn to restore missing detail from a coarse VAE/RVQ-VAE mel spectrogram.

The deterministic refiner learns a residual:

`x_hat = x_low + f(x_low)`

where `x_low` is the coarse RVQ-VAE mel and `f(.)` predicts the residual needed to recover the high-detail target.

The conditional diffusion refiner learns the distribution of the missing residual detail and samples it while conditioning on the coarse mel.

## What is included

- A paired `.npy` dataset loader for `128 x 128` mel spectrograms
- Shared normalization stats computed from the training split
- Optional 3-channel input: `[mel, delta_time, delta_freq]`
- A progressive residual U-Net refiner with:
  - a coarse stage
  - a full-resolution stage
  - a mel-specific sub-band branch
- A conditional DDPM/DDIM diffusion refiner with:
  - residual or direct-mel diffusion targets
  - sinusoidal timestep conditioning
  - U-Net denoiser with optional bottleneck self-attention
- Composite loss:
  - `L1`
  - gradient loss
  - high-frequency Laplacian loss
  - SSIM loss
- Validation metrics:
  - MAE
  - SSIM
  - LSD
- Checkpointed inference that reuses the saved preprocessing settings

## Repository layout

```text
melrestoration/
  __init__.py
  data.py
  losses.py
  metrics.py
  models.py
  diffusion.py
  train.py
  train_diffusion.py
  infer.py
  infer_diffusion.py
```

## Data layout

The training code expects matching relative file paths in two folders:

```text
data/
  low/
    clip_0001.npy
    clip_0002.npy
  high/
    clip_0001.npy
    clip_0002.npy
```

Nested directories are also supported as long as the relative paths match between `low` and `high`.

If the filenames match but the subfolder layout is different, add `--pairing-mode basename`. Basename pairing requires every `.npy` filename to be unique in both folders.

## Install

```bash
python -m pip install -r requirements.txt
```

## Train

Deterministic residual U-Net example:

Example:

```bash
python -m melrestoration.train ^
  --low-dir data/low ^
  --high-dir data/high ^
  --output-dir runs/refiner_v1 ^
  --pairing-mode basename ^
  --group-separator "_" ^
  --batch-size 32 ^
  --epochs 120 ^
  --lr 2e-4 ^
  --base-channels 64 ^
  --num-subbands 4
```

Important:

- Use `--group-separator` so windows from the same original clip stay in the same split. For filenames like `clip123_00045.npy`, use `_`.
- Use `--pairing-mode basename` if files such as `blues.00000ConcertHallLA_02_mel.npy` exist in both folders but not at the same relative subfolder path.
- If your files are already grouped by folder, you can omit `--group-separator` and the parent directory will be used as the grouping key.
- On Windows, start with `--num-workers 0`. Increase it only after confirming the data loader is stable in your environment.

Useful flags:

- `--no-use-deltas` to use a single mel channel instead of `[mel, dt, df]`
- `--scheduler plateau` to switch from cosine decay to `ReduceLROnPlateau`
- `--detail-freq-boost 2.0` to emphasize upper mel bins more strongly in detail losses
- `--norm none` if your mel files are already normalized consistently

Training outputs:

- `best.pt`: best validation checkpoint
- `last.pt`: most recent checkpoint
- `metrics.csv`: one row per epoch with `train_loss`, `val_loss`, `val_mae`, `val_ssim`, `val_lsd`, and `lr`
- `stats.json`: shared normalization stats used for both input and target

Resume training from the last checkpoint:

```bash
python -m melrestoration.train ^
  --low-dir data/low ^
  --high-dir data/high ^
  --output-dir runs/refiner_v1 ^
  --resume runs/refiner_v1/last.pt ^
  --group-separator "_" ^
  --batch-size 32 ^
  --epochs 120
```

When resuming, `--epochs` is the total target epoch count, not the number of extra epochs. For example, if the checkpoint was saved at epoch `30` and you want to continue to epoch `120`, use `--epochs 120`.

## Train Conditional Diffusion

The diffusion model uses the same paired data layout. By default it diffuses the residual `target - low`, then adds the sampled residual back to the input mel during inference.

Example:

```bash
python -m melrestoration.train_diffusion ^
  --low-dir data/low ^
  --high-dir data/high ^
  --output-dir runs/diffusion_v1 ^
  --pairing-mode basename ^
  --group-separator "_" ^
  --batch-size 16 ^
  --epochs 200 ^
  --lr 1e-4 ^
  --base-channels 64 ^
  --timesteps 1000 ^
  --target-mode residual
```

Useful diffusion flags:

- `--batch-size 8` or `--grad-accum-steps 2` if GPU memory is tight
- `--channel-mults 1,2,4,4` to control U-Net depth/width
- `--target-mode mel` to generate the full high-quality mel instead of the residual
- `--beta-schedule linear` to use a linear DDPM schedule instead of cosine
- `--no-use-attention` to reduce memory use

Training outputs:

- `best_diffusion.pt`: best validation checkpoint
- `last_diffusion.pt`: most recent checkpoint
- `diffusion_metrics.csv`: one row per epoch
- `stats.json`: shared normalization stats, when z-score normalization is enabled

## Inference

Single file:

```bash
python -m melrestoration.infer ^
  --checkpoint runs/refiner_v1/best.pt ^
  --input samples/r1.npy ^
  --output samples/r1_refined.npy
```

Directory:

```bash
python -m melrestoration.infer ^
  --checkpoint runs/refiner_v1/best.pt ^
  --input data/test_low ^
  --output data/test_refined
```

Diffusion inference:

```bash
python -m melrestoration.infer_diffusion ^
  --checkpoint runs/diffusion_v1/best_diffusion.pt ^
  --input data/test_low ^
  --output data/test_diffusion_refined ^
  --sample-steps 50 ^
  --eta 0 ^
  --clip-min -12 ^
  --clip-max 1
```

`--sampler ddim` is the default and is much faster than full DDPM sampling. Increase `--sample-steps` for quality; decrease it for speed. Change `--seed` if you want a different stochastic restoration.

## Notes

- The model predicts a residual, not a full spectrogram from scratch.
- The diffusion model also defaults to residual prediction. Use `--target-mode mel` only if residual sampling is not stable for your data.
- The code assumes each `.npy` file contains a single `2D` mel array. A shape like `(1, 128, 128)` is also accepted.
- `LSD` is computed directly in mel space, which is most meaningful if your stored mel values are already in a log domain.
