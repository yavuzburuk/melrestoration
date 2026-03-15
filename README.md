# melrestoration

Residual mel-spectrogram restoration on top of an RVQ-VAE output.

This repository now contains a second-stage PyTorch refiner that learns the missing detail as a residual:

`x_hat = x_low + f(x_low)`

where `x_low` is the coarse RVQ-VAE mel and `f(.)` predicts the residual needed to recover the high-detail target.

## What is included

- A paired `.npy` dataset loader for `128 x 128` mel spectrograms
- Shared normalization stats computed from the training split
- Optional 3-channel input: `[mel, delta_time, delta_freq]`
- A progressive residual U-Net refiner with:
  - a coarse stage
  - a full-resolution stage
  - a mel-specific sub-band branch
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
  train.py
  infer.py
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

## Install

```bash
python -m pip install -r requirements.txt
```

## Train

Example:

```bash
python -m melrestoration.train ^
  --low-dir data/low ^
  --high-dir data/high ^
  --output-dir runs/refiner_v1 ^
  --group-separator "_" ^
  --batch-size 32 ^
  --epochs 120 ^
  --lr 2e-4 ^
  --base-channels 64 ^
  --num-subbands 4
```

Important:

- Use `--group-separator` so windows from the same original clip stay in the same split. For filenames like `clip123_00045.npy`, use `_`.
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

## Notes

- The model predicts a residual, not a full spectrogram from scratch.
- The code assumes each `.npy` file contains a single `2D` mel array. A shape like `(1, 128, 128)` is also accepted.
- `LSD` is computed directly in mel space, which is most meaningful if your stored mel values are already in a log domain.
