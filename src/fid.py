"""Fréchet Inception Distance helpers for the Fashion-MNIST arms.

No model is retrained here: 5000 samples are drawn from the final generator of an arm and
compared against a fixed 5000-image slice of the same dataset test split. InceptionV3 comes
from the installed pytorch_fid package, but only as a feature extractor — the Fréchet distance
itself is computed here, because pytorch_fid's own path/CLI API calls scipy.linalg.sqrtm with
the removed "disp" argument and breaks on scipy 1.18.

Statistics are cached in artifacts/metrics/fid_stats/, so re-runs do not repeat Inception.

This module is a library: the notebook imports it for the live FID of the two arms it trains,
src/fid_fashion_factorial.py uses it for the archived arms.
"""
from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ganlib as G

N_SAMPLES = 5000
BATCH = 50
FEATURE_DIM = 2048
LATENT_SEED = 1234
STATS_DIR = G._CODE_ARTIFACTS / "metrics" / "fid_stats"
OUT = G._CODE_ARTIFACTS / "metrics" / "fid.json"


# ---------------------------------------------------------------- Fréchet distance
def _sym(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root of a symmetric PSD matrix via eigendecomposition (no scipy)."""
    matrix = _sym(np.asarray(matrix, dtype=np.float64))
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """|mu1 - mu2|^2 + Tr(S1) + Tr(S2) - 2 Tr((S1 S2)^(1/2)).

    The last term is the sum of the square roots of the eigenvalues of the symmetric form
    S1^(1/2) S2 S1^(1/2); negative eigenvalues (numerical noise, rank deficiency) clamp to 0.
    """
    mu1, mu2 = np.asarray(mu1, np.float64).ravel(), np.asarray(mu2, np.float64).ravel()
    sigma1, sigma2 = np.asarray(sigma1, np.float64), np.asarray(sigma2, np.float64)
    diff = mu1 - mu2
    root1 = _sqrtm_psd(sigma1)
    covmean = _sym(root1 @ sigma2 @ root1)
    eigenvalues = np.linalg.eigvalsh(covmean)
    trace_sqrt = float(np.sum(np.sqrt(np.clip(eigenvalues, 0.0, None))))
    tr1, tr2 = float(np.trace(sigma1)), float(np.trace(sigma2))
    return float(diff @ diff) + tr1 + tr2 - 2.0 * trace_sqrt


def stats_of(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    return features.mean(axis=0), np.cov(features, rowvar=False)


# ---------------------------------------------------------------- inception features
def make_inception(device: torch.device):
    from pytorch_fid.inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIM]
    model = InceptionV3([block_idx], resize_input=True, normalize_input=True,
                        requires_grad=False)
    return model.to(device).eval()


@torch.no_grad()
def inception_features(images: torch.Tensor, model, device: torch.device) -> np.ndarray:
    """images: (N, 1, 28, 28) in [-1, 1] -> the same tensors in the ImageNet-normalised
    3-channel 299x299 space Inception expects."""
    images = images.to(device, dtype=torch.float32)
    x = (images + 1.0) / 2.0            # [-1, 1] -> [0, 1]
    x = x.repeat(1, 3, 1, 1)            # grayscale -> RGB
    out = model(x)[0]                   # (N, 2048, 1, 1)
    return out.squeeze(-1).squeeze(-1).cpu().numpy()


def features_of_batches(batches, model, device: torch.device) -> np.ndarray:
    chunks = [inception_features(b, model, device) for b in batches]
    return np.concatenate(chunks, axis=0)


def cached_stats(key: str, n: int, produce, force: bool = False):
    """Cache (mu, sigma) per image set so a stage-1b arm costs one Inception pass, not all."""
    path = STATS_DIR / (key + "_n" + str(n) + "_" + str(FEATURE_DIM) + "d.npz")
    if path.exists() and not force:
        blob = np.load(path)
        return blob["mu"], blob["sigma"].astype(np.float64), True
    mu, sigma = produce()
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mu=mu, sigma=sigma.astype(np.float32))
    return mu, sigma, False


# ---------------------------------------------------------------- data and generators
def test_images(dataset: str) -> torch.Tensor:
    """The whole test split as (N, 1, 28, 28) in [-1, 1] — the same transform as training."""
    ds = G.get_test_dataset(dataset=dataset)
    return torch.stack([ds[i][0] for i in range(len(ds))])


def latent(n: int, z_dim: int) -> torch.Tensor:
    """Same latent convention as ganlib: (N, z_dim, 1, 1), fixed seed for every arm so the
    arms are compared on identical latent codes."""
    generator = torch.Generator().manual_seed(LATENT_SEED)
    return torch.randn(n, z_dim, 1, 1, generator=generator)


def load_generator(ckpt_path: str):
    """Rebuild the generator from the config stored in the checkpoint. Returns the model,
    its config dict and an error string (None on success)."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = blob.get("gen")
    if state is None:
        state = blob.get("generator")
    if state is None:
        return None, {}, "checkpoint has no generator (keys: " + str(list(blob)) + ")"
    known = {f.name for f in fields(G.Config)}
    cfg_dict = {k: v for k, v in (blob.get("config") or {}).items() if k in known}
    cfg = G.Config(**cfg_dict)
    gen = G.make_generator(int(cfg.z_dim), tuple(cfg.g_widths))
    state = state.state_dict() if hasattr(state, "state_dict") else state
    try:
        gen.load_state_dict(state)
    except RuntimeError as exc:
        return None, cfg_dict, "state dict does not fit the current constructor: " + str(exc)
    return gen.eval(), cfg_dict, None


def generated_stats(gen, cfg: dict, n: int, model, device):
    z_dim = int(cfg.get("z_dim", 100))
    z = latent(n, z_dim)

    def produce():
        batches = []
        with torch.no_grad():
            for start in range(0, n, BATCH):
                batches.append(gen(z[start:start + BATCH]))
        return stats_of(features_of_batches(batches, model, device))

    return produce


# ---------------------------------------------------------------- main
