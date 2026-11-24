"""Shared utilities for the mode-collapse experiments (Fashion-MNIST and MNIST).

Vanilla GAN (BCE) vs WGAN-GP (Wasserstein + gradient penalty), same generator.
An "epoch" is one pass of the training set through the discriminator/critic.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
# data caches must not move when a run re-points ARTIFACTS at its own directory
_CODE_ARTIFACTS = ARTIFACTS
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
DIGIT_NAMES = [str(i) for i in range(10)]
DATASET_CLASS_NAMES = {"fashion": CLASS_NAMES, "mnist": DIGIT_NAMES}
# A class counts as represented in a generated batch when it holds at least
# SHARE_EPS of the predicted labels; this kills single-sample noise.
SHARE_EPS = 0.01


@dataclass
class Config:
    seed: int = 42
    z_dim: int = 100
    batch_size: int = 128
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    epochs: int = 40
    n_critic: int = 5   # steps of D (or the critic) per one step of G, for both variants
    lambda_gp: float = 10.0
    n_eval: int = 2048
    n_clusters: int = 20
    threads: int = 4
    device: str = "cpu"
    preload: bool = True
    label: str = "vanilla"
    dataset: str = "fashion"      # "fashion" | "mnist"
    model: str = ""               # "" = infer from label; "vanilla" | "wgan"
    g_widths: tuple = (128, 64)   # generator ConvTranspose channel widths
    d_widths: tuple = (32, 64)    # discriminator/critic Conv2d channel widths
    d_bn: bool = True             # BatchNorm inside D, from the second block on
    d_extra: int = 0              # extra 3x3 stride-1 conv blocks at 7x7 in D


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def resolve_device(prefer: str | None = None) -> torch.device:
    """Pick the training device: CUDA/ROCm, then MPS, then CPU.

    ROCm builds of PyTorch expose the HIP runtime through the ``torch.cuda`` API and set
    ``torch.version.hip``, so "cuda" is the right name on an AMD GPU too. ``prefer`` (or
    the ``GAN_DEVICE`` environment variable) overrides auto-detection: ``cuda``/``rocm``,
    ``mps``, ``cpu``.
    """
    if isinstance(prefer, torch.device):
        return prefer
    prefer = str(prefer or os.environ.get("GAN_DEVICE") or "").strip().lower()
    if prefer in ("", "auto"):
        prefer = ""
    if prefer in ("cuda", "rocm", "hip"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm requested, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested, but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer:
        raise ValueError(f"unknown device {prefer!r}; use cuda, mps or cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report(device: torch.device) -> str:
    """One line for logs and notebooks: what will actually train the models."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        memory = torch.cuda.get_device_properties(index).total_memory / 2 ** 30
        if torch.version.hip:
            backend = f"ROCm/HIP {torch.version.hip}"
        else:
            backend = f"CUDA {torch.version.cuda}"
        return f"{name} ({backend}, {memory:.0f} GiB, torch {torch.__version__})"
    if device.type == "mps":
        return f"Apple MPS (torch {torch.__version__})"
    return f"CPU, {torch.get_num_threads()} threads (torch {torch.__version__})"


def configure_threads(threads: int) -> None:
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


class FastConv2d(nn.Conv2d):
    """Conv2d expressed as unfold + GEMM: the same function, the same parameters.

    WGAN-GP differentiates the critic twice (the gradient penalty), and on ROCm the MIOpen
    conv double backward costs ~80 ms per critic step while the equivalent unfold+linear
    graph takes ~12 ms. Parameters, output and initialisation are identical to nn.Conv2d,
    so this is an implementation detail, not a change of the experiment. Used through
    ``conv2d_cls`` only on ROCm; everywhere else the stock layer is kept.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cols = F.unfold(x, self.kernel_size, dilation=self.dilation,
                        padding=self.padding, stride=self.stride)
        weight = self.weight.reshape(self.out_channels, -1)
        out = torch.einsum("oc,bcl->bol", weight, cols)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)
        height = (x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1)
                  - 1) // self.stride[0] + 1
        width = (x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1)
                 - 1) // self.stride[1] + 1
        return out.view(x.shape[0], self.out_channels, height, width)


def conv2d_cls():
    """FastConv2d on ROCm, stock Conv2d on CUDA/MPS/CPU."""
    return FastConv2d if torch.version.hip else nn.Conv2d


def _dataset_cls(dataset: str):
    if dataset == "fashion":
        return datasets.FashionMNIST
    if dataset == "mnist":
        return datasets.MNIST
    raise ValueError(f"unknown dataset {dataset!r}")


def _dataset_dir(root: Path, dataset: str) -> Path:
    return root / ("FashionMNIST" if dataset == "fashion" else "MNIST")


def resolve_data_root(dataset: str = "fashion") -> Path:
    """Local copy first (../seminars/data, then reports/data), otherwise download into
    artifacts/data. Both datasets are already unpacked locally, so no network is needed."""
    for cand in (PROJECT_ROOT.parent.parent / "seminars" / "data",
                 PROJECT_ROOT.parent / "data"):
        if (_dataset_dir(cand, dataset) / "raw" / "train-images-idx3-ubyte").exists():
            return cand
    return _CODE_ARTIFACTS / "data"


def get_train_loader(cfg: Config, root: Path | None = None) -> DataLoader:
    root = root or resolve_data_root(cfg.dataset)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = _dataset_cls(cfg.dataset)(root=str(root), train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)


def _train_tensor_cache(dataset: str) -> Path:
    name = "train_tensor.pt" if dataset == "fashion" else f"train_tensor_{dataset}.pt"
    return _CODE_ARTIFACTS / "data" / name


def load_train_tensor(cfg: Config | None = None) -> torch.Tensor:
    """The whole training split as one CPU tensor: 60k images is 188 MB, so keeping it in
    memory removes the per-batch transform cost that otherwise stalls the GPU."""
    cfg = cfg or Config()
    cache = _train_tensor_cache(cfg.dataset)
    if cache.exists():
        return torch.load(cache, weights_only=True)
    root = resolve_data_root(cfg.dataset)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = _dataset_cls(cfg.dataset)(root=str(root), train=True, download=True, transform=tf)
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(xs, cache)
    return xs


def get_test_dataset(root: Path | None = None, dataset: str = "fashion"):
    root = root or resolve_data_root(dataset)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return _dataset_cls(dataset)(root=str(root), train=False, download=True, transform=tf)


# ---------------------------------------------------------------- networks
def make_generator(z_dim: int = 100, g_widths: tuple = (128, 64)) -> nn.Module:
    """DCGAN generator: a 7x7 seed upsampled twice to 28x28. The default widths are the
    original recipe, so existing runs stay bit-identical."""
    w0, w1 = g_widths
    return nn.Sequential(
        nn.ConvTranspose2d(z_dim, w0, 7, 1, 0), nn.BatchNorm2d(w0), nn.ReLU(True),
        nn.ConvTranspose2d(w0, w1, 4, 2, 1), nn.BatchNorm2d(w1), nn.ReLU(True),
        nn.ConvTranspose2d(w1, 1, 4, 2, 1), nn.Tanh(),
    )


def make_discriminator(d_widths: tuple = (32, 64), d_bn: bool = True, d_extra: int = 0,
                       img_size: int = 28) -> nn.Module:
    """Vanilla-GAN discriminator: sigmoid output over one logit. BatchNorm sits on every
    stride-2 block after the first one (default DCGAN recipe); d_bn=False drops it, and
    d_extra adds stride-1 conv blocks at the 7x7 resolution."""
    layers: list[nn.Module] = []
    channels = 1
    for i, w in enumerate(d_widths):
        layers += [nn.Conv2d(channels, w, 4, 2, 1)]
        if d_bn and i > 0:
            layers.append(nn.BatchNorm2d(w))
        layers.append(nn.LeakyReLU(0.2, True))
        channels = w
    size = img_size // (2 ** len(d_widths))
    for _ in range(d_extra):
        layers += [nn.Conv2d(channels, channels, 3, 1, 1)]
        if d_bn:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.LeakyReLU(0.2, True))
    layers += [nn.Flatten(), nn.Linear(channels * size * size, 1)]
    return nn.Sequential(*layers)


def make_critic(d_widths: tuple = (32, 64), d_extra: int = 0,
                img_size: int = 28) -> nn.Module:
    """WGAN-GP critic: linear output, no BatchNorm (batch norm breaks the per-sample
    gradient penalty, as noted in Gulrajani et al. 2017)."""
    return make_discriminator(d_widths=d_widths, d_bn=False, d_extra=d_extra,
                              img_size=img_size)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------- classifier
class ModeClassifier(nn.Module):
    """Small CNN labelling generated images, plus a 64-d feature head used for
    cluster-coverage measurement."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.embed = nn.Linear(32 * 7 * 7, 64)
        self.head = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.embed(self.features(x))))

    def embed_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.embed(self.features(x)))


def train_classifier(cfg: Config, epochs: int = 4, out_path: Path | None = None,
                     device: str = "cpu") -> tuple[ModeClassifier, float]:
    """Train (or load) the frozen labeller used to read modes off generated samples.
    Each dataset needs its own labeller: the Fashion-MNIST one labels digits as clothes."""
    suffix = "" if cfg.dataset == "fashion" else f"_{cfg.dataset}"
    out_path = out_path or (ARTIFACTS / "models" / f"mode_classifier{suffix}.pt")
    clf = ModeClassifier()
    if out_path.exists():
        blob = torch.load(out_path, map_location="cpu", weights_only=True)
        clf.load_state_dict(blob["state_dict"])
        clf.eval()
        return clf, float(blob["accuracy"])

    dev = torch.device(device)
    clf.to(dev)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    root = str(resolve_data_root(cfg.dataset))
    cls = _dataset_cls(cfg.dataset)
    train_ds = cls(root, train=True, download=True, transform=tf)
    test_ds = cls(root, train=False, download=True, transform=tf)
    train_ld = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=512, num_workers=0)

    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for ep in range(epochs):
        clf.train()
        running, seen = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
            seen += y.numel()
        print(f"[classifier:{cfg.dataset}] epoch {ep + 1}/{epochs} loss={running / seen:.4f}",
              flush=True)

    clf.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_ld:
            x, y = x.to(dev), y.to(dev)
            correct += (clf(x).argmax(1) == y).sum().item()
            total += y.numel()
    accuracy = correct / total
    clf.cpu()  # mode metrics run on CPU tensors, see measure_modes
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": clf.state_dict(), "accuracy": accuracy}, out_path)
    return clf, accuracy


# ---------------------------------------------------------------- mode metrics
def kmeans_np(x: np.ndarray, k: int, iters: int = 100, seed: int = 0,
              tol: float = 1e-4) -> np.ndarray:
    """Lloyd's algorithm with empty-cluster restarts; stops when the centres stop moving."""
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(1)
        updated = centers.copy()
        for j in range(k):
            members = x[assign == j]
            updated[j] = members.mean(0) if len(members) else x[rng.integers(len(x))]
        shift = float(np.abs(updated - centers).max())
        centers = updated
        if shift < tol:
            break
    return centers


def _standardize(feats: np.ndarray, ref: dict) -> np.ndarray:
    """Raw 64-d ReLU features have wildly different scales per dimension; without this
    the nearest-centre search is dominated by the sharpest channels."""
    return (feats - ref["feature_mean"]) / ref["feature_std"]


def build_real_reference(clf: ModeClassifier, cfg: Config) -> dict:
    """Cluster real data in classifier feature space: the set of modes we expect a
    healthy generator to cover."""
    ds = get_test_dataset(dataset=cfg.dataset)
    loader = DataLoader(ds, batch_size=512, num_workers=0)
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            feats.append(clf.embed_features(x))
    feats = torch.cat(feats).numpy()
    ref = {"feature_mean": feats.mean(0), "feature_std": feats.std(0) + 1e-6}
    z = _standardize(feats, ref)
    sub = np.random.default_rng(cfg.seed).choice(len(z), 8000, replace=False)
    centers = kmeans_np(z[sub], cfg.n_clusters, seed=cfg.seed)
    ref["centers"] = centers
    # a genuine image sits close to its own cluster centre; that distance sets the
    # acceptance radius, so a generator emitting noise cannot claim coverage
    d_real = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(-1).min(1)
    ref["dist_threshold"] = float(np.percentile(d_real, 99))
    return ref


@torch.no_grad()
def generate_samples(gen: nn.Module, z: torch.Tensor, batch: int = 512) -> torch.Tensor:
    out = []
    for i in range(0, len(z), batch):
        out.append(gen(z[i:i + batch]))
    return torch.cat(out)


def _mean_pairwise(flat: np.ndarray, max_n: int = 512) -> float:
    """Mean pairwise L2 distance, accumulated row by row to keep memory linear.
    Oversized groups are subsampled at evenly spaced indices — no RNG, so the training
    stream stays bit-reproducible."""
    n_all = len(flat)
    if n_all < 2:
        return 0.0
    if n_all > max_n:
        flat = flat[np.linspace(0, n_all - 1, max_n).round().astype(int)]
    total = 0.0
    n = len(flat)
    for i in range(n - 1):
        total += float(np.linalg.norm(flat[i + 1:] - flat[i], axis=1).sum())
    return total / (n * (n - 1) / 2)


@torch.no_grad()
def measure_modes(gen: nn.Module, z_eval: torch.Tensor, clf: ModeClassifier, ref: dict,
                  n_clusters: int) -> dict:
    """Read the mode distribution off a fixed latent bank."""
    gen.eval()
    dev = next(gen.parameters()).device
    imgs = generate_samples(gen, z_eval.to(dev)).cpu()
    logits = clf(imgs)
    feats = clf.embed_features(imgs).numpy()
    pred = logits.argmax(1).numpy()

    feats = _standardize(feats, ref)
    share = np.bincount(pred, minlength=10) / len(pred)
    ent = -(share[share > 0] * np.log(share[share > 0])).sum() / np.log(10)
    d = ((feats[:, None, :] - ref["centers"][None, :, :]) ** 2).sum(-1)
    nearest, dist = d.argmin(1), d.min(1)
    inside = dist <= ref["dist_threshold"]

    counts = np.bincount(nearest[inside], minlength=n_clusters)
    covered = int((counts >= max(5, int(0.005 * len(pred)))).sum())

    flat = imgs.reshape(len(imgs), -1)
    centroid = flat.mean(0, keepdim=True)
    spread = float((flat - centroid).pow(2).sum(1).mean().sqrt())
    ordered = np.sort(share)[::-1]
    # mode-collapse signature: one class takes everything *and* varies little inside it
    dominant = int(share.argmax())
    dom_mask = pred == dominant

    return {
        "dominant_class": dominant,
        "intra_class_n": int(dom_mask.sum()),
        "intra_class_spread": _mean_pairwise(flat.numpy()[dom_mask]),
        "class_share": share.tolist(),
        # floor of this counter is the labeller's confusion rate, so it is kept raw only
        "n_classes_raw": int((share >= SHARE_EPS).sum()),
        "n_classes_10": int((share >= 0.10).sum()),
        "class_entropy": float(ent),
        "max_class_share": float(share.max()),
        "top3_share": float(ordered[:3].sum()),
        "modes_covered": covered,
        "mode_coverage": covered / n_clusters,
        "coverage_share": float(inside.mean()),
        "pixel_spread": spread,
        "confidence": float(logits.softmax(1).max(1).values.mean()),
    }


# ---------------------------------------------------------------- losses
def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor,
                     lambda_gp: float) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(interp)
    grads = torch.autograd.grad(score, interp, torch.ones_like(score), create_graph=True)[0]
    return lambda_gp * ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def _empty_history() -> dict:
    return {k: [] for k in (
        "epoch", "loss_g", "loss_d", "gp", "d_real", "d_fake", "d_margin",
        "epoch_seconds", "cumulative_seconds", "gen_updates",
    )}


def _record(history: dict, metrics: dict, epoch: int, seconds: float, cumulative: float,
            gen_updates: int) -> None:
    agg = {k: float(np.mean(metrics[k])) if metrics[k] else float("nan")
           for k in ("loss_g", "loss_d", "gp", "d_real", "d_fake", "d_margin")}
    history["epoch"].append(epoch)
    for k, v in agg.items():
        history[k].append(v)
    history["epoch_seconds"].append(seconds)
    history["cumulative_seconds"].append(cumulative)
    history["gen_updates"].append(gen_updates)
    for k, v in metrics["modes"].items():
        history.setdefault(k, [])
        history[k].append([float(x) for x in v] if isinstance(v, list) else v)


def train(cfg: Config, clf: ModeClassifier, ref: dict, out_dir: Path | None = None,
          log=print) -> dict:
    """Train one GAN variant and return its history + wall-clock cost."""
    out_dir = out_dir or (ARTIFACTS / "metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "models").mkdir(parents=True, exist_ok=True)
    configure_threads(cfg.threads)
    set_seed(cfg.seed)
    cpu_start = time.process_time()
    wall_start = time.time()

    dev = torch.device(cfg.device)
    # "model" is explicit; empty means the historical label-based inference
    variant = cfg.model or ("wgan" if "wgan" in cfg.label else "vanilla")
    X = load_train_tensor(cfg).to(dev)
    bs = cfg.batch_size
    iters = X.size(0) // bs
    gen = make_generator(cfg.z_dim, tuple(cfg.g_widths)).to(dev)
    if variant == "vanilla":
        disc = make_discriminator(tuple(cfg.d_widths), cfg.d_bn, cfg.d_extra).to(dev)
    else:
        disc = make_critic(tuple(cfg.d_widths), cfg.d_extra).to(dev)

    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, 0.999))
    bce = nn.BCEWithLogitsLoss()

    g_eval = torch.Generator(device=dev).manual_seed(cfg.seed + 1)
    z_eval = torch.randn(cfg.n_eval, cfg.z_dim, 1, 1, generator=g_eval, device=dev)
    history = _empty_history()
    history["config"] = asdict(cfg)
    history["params_g"] = count_parameters(gen)
    history["params_d"] = count_parameters(disc)

    cumulative = 0.0
    gen_updates = 0
    snapshots: dict[str, np.ndarray] = {}
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        gen.train()
        disc.train()
        log_ep = {"loss_g": [], "loss_d": [], "gp": [], "d_real": [], "d_fake": [],
                  "d_margin": []}

        perm = torch.randperm(X.size(0), device=dev)
        for step in range(iters):
            real = X[perm[step * bs:(step + 1) * bs]]
            z = torch.randn(bs, cfg.z_dim, 1, 1, device=dev)
            if variant == "vanilla":
                fake = gen(z).detach()
                out_real = disc(real)
                out_fake = disc(fake)
                loss_d = bce(out_real, torch.ones_like(out_real)) + \
                    bce(out_fake, torch.zeros_like(out_fake))
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                log_ep["d_real"].append(out_real.mean().item())
                log_ep["d_fake"].append(out_fake.mean().item())
                log_ep["loss_d"].append(loss_d.item())
                log_ep["d_margin"].append(
                    out_real.mean().item() - out_fake.mean().item())
                if (step + 1) % cfg.n_critic == 0:
                    out_g = disc(gen(torch.randn(real.size(0), cfg.z_dim, 1, 1, device=dev)))
                    loss_g = bce(out_g, torch.ones_like(out_g))
                    opt_g.zero_grad()
                    loss_g.backward()
                    opt_g.step()
                    log_ep["loss_g"].append(loss_g.item())
                    gen_updates += 1
            else:
                real = real.requires_grad_(False)
                with torch.no_grad():
                    fake = gen(z)
                d_real = disc(real)
                d_fake = disc(fake)
                gp = gradient_penalty(disc, real, fake, cfg.lambda_gp)
                loss_d = d_fake.mean() - d_real.mean() + gp
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                log_ep["loss_d"].append((d_fake.mean() - d_real.mean()).item())
                log_ep["gp"].append(gp.item())
                log_ep["d_real"].append(d_real.mean().item())
                log_ep["d_fake"].append(d_fake.mean().item())
                log_ep["d_margin"].append(d_real.mean().item() - d_fake.mean().item())
                # one generator step every n_critic critic steps
                if (step + 1) % cfg.n_critic == 0:
                    loss_g = -disc(gen(torch.randn(real.size(0), cfg.z_dim, 1, 1, device=dev))).mean()
                    opt_g.zero_grad()
                    loss_g.backward()
                    opt_g.step()
                    log_ep["loss_g"].append(loss_g.item())
                    gen_updates += 1

            if (step + 1) % 200 == 0:
                log(f"[{cfg.label}] epoch {epoch} batch {step + 1}: "
                    f"d={np.mean(log_ep['loss_d']):.3f} g={np.mean(log_ep['loss_g']):.3f} "
                    f"({time.time() - t0:.0f}s)", flush=True)

        seconds = time.time() - t0
        cumulative += seconds
        modes = measure_modes(gen, z_eval, clf, ref, cfg.n_clusters)
        with torch.no_grad():
            gen.eval()
            _snap = generate_samples(gen, z_eval[:64])
        snapshots[str(epoch)] = ((_snap.cpu() * 0.5 + 0.5).clamp(0, 1).mul(255).byte()
                                 .squeeze(1).numpy())
        np.savez_compressed(out_dir / f"{cfg.label}_snapshots.npz", **snapshots)
        gen.train()
        _record(history, {**log_ep, "modes": modes}, epoch, seconds, cumulative, gen_updates)
        log(f"[{cfg.label}] epoch {epoch:02d} done: {seconds:.1f}s | "
            f"classes10={modes['n_classes_10']} cov={modes['mode_coverage']:.2f} "
            f"H={modes['class_entropy']:.2f} max={modes['max_class_share']:.2f} "
            f"dom={modes['dominant_class']} intra={modes['intra_class_spread']:.2f} | "
            f"loss_d={np.mean(log_ep['loss_d']):.3f} "
            f"margin={np.mean(log_ep['d_margin']):.2f} "
            f"loss_g={np.mean(log_ep['loss_g']):.3f}", flush=True)

        blob = dict(history)
        blob["cpu_seconds"] = time.process_time() - cpu_start
        blob["wall_seconds"] = time.time() - wall_start
        (out_dir / f"{cfg.label}_history.json").write_text(json.dumps(blob, indent=1))
        torch.save({"gen": gen.state_dict(), "disc": disc.state_dict(), "config": asdict(cfg)},
                   ARTIFACTS / "models" / f"{cfg.label}_checkpoint.pt")

    dom = history.get("dominant_class", [])
    history["n_switches"] = int(sum(1 for a, b in zip(dom, dom[1:]) if a != b))
    history["cpu_seconds"] = time.process_time() - cpu_start
    history["wall_seconds"] = time.time() - wall_start
    history["epochs"] = cfg.epochs
    history["threads"] = cfg.threads
    history["n_params_total"] = history["params_g"] + history["params_d"]
    (out_dir / f"{cfg.label}_history.json").write_text(json.dumps(history, indent=1))
    torch.save({"gen": gen.state_dict(), "disc": disc.state_dict(), "config": asdict(cfg)},
               ARTIFACTS / "models" / f"{cfg.label}_checkpoint.pt")
    return history
