"""PyTorch CNN autoencoder for A/E WDM patches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .wdm_pixels import load_dataset


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, random_split
    except Exception as exc:  # pragma: no cover - depends on caller environment.
        raise ModuleNotFoundError(
            "Autoencoder training requires PyTorch. "
            "Update the lisa-tf environment from env.yml or install pytorch."
        ) from exc
    return torch, nn, DataLoader, TensorDataset, random_split


def _device(torch):
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(input_shape: tuple[int, int, int], latent_dim: int):
    torch, nn, _, _, _ = _require_torch()
    channels, height, width = input_shape

    class ConvAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(4, 16),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(16, 128),
                nn.GELU(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, channels, height, width)
                encoded = self.encoder_conv(dummy)
            self.encoded_shape = tuple(encoded.shape[1:])
            flat_dim = int(np.prod(self.encoded_shape))
            self.to_latent = nn.Linear(flat_dim, latent_dim)
            self.from_latent = nn.Linear(latent_dim, flat_dim)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(4, 16),
                nn.GELU(),
                nn.ConvTranspose2d(16, channels, kernel_size=4, stride=2, padding=1),
            )

        def encode(self, x):
            z = self.encoder_conv(x)
            z = z.reshape(z.shape[0], -1)
            return self.to_latent(z)

        def forward(self, x):
            z = self.encode(x)
            y = self.from_latent(z).reshape(x.shape[0], *self.encoded_shape)
            y = self.decoder(y)
            return y[..., : x.shape[-2], : x.shape[-1]]

    return ConvAutoencoder()


def train_autoencoder(
    dataset_npz: str | Path,
    *,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    validation_fraction: float,
    seed: int,
    model_path: str | Path,
    embeddings_npz: str | Path,
    history_csv: str | Path,
) -> tuple[Path, Path, Path]:
    """Train the CNN autoencoder and save embeddings."""

    torch, nn, DataLoader, TensorDataset, random_split = _require_torch()
    torch.manual_seed(seed)
    x, source_ids = load_dataset(dataset_npz)
    if x.ndim != 4:
        raise ValueError("Dataset x must have shape [n, channels, freq, time].")
    tensor = torch.as_tensor(x, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    n_val = int(round(len(dataset) * validation_fraction))
    n_val = min(max(n_val, 0), max(0, len(dataset) - 1))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if n_val else None

    device = _device(torch)
    model = build_model(tuple(x.shape[1:]), latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = np.nan
        if val_loader is not None:
            model.eval()
            losses = []
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    losses.append(float(loss_fn(model(batch), batch).detach().cpu()))
            val_loss = float(np.mean(losses)) if losses else np.nan
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})

    model_path = Path(model_path)
    embeddings_path = Path(embeddings_npz)
    history_path = Path(history_csv)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_shape": tuple(x.shape[1:]), "latent_dim": latent_dim}, model_path)

    embeddings = encode_dataset(model, x, batch_size=batch_size)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embeddings_path, embeddings=embeddings, source_id=source_ids)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return model_path, embeddings_path, history_path


def encode_dataset(model, x: np.ndarray, *, batch_size: int = 64) -> np.ndarray:
    torch, _, DataLoader, TensorDataset, _ = _require_torch()
    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(torch.as_tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    model.eval()
    out = []
    with torch.no_grad():
        for (batch,) in loader:
            out.append(model.encode(batch.to(device)).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
