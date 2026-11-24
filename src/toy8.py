"""8 Gaussians on a ring: the textbook mode-collapse picture, BCE against WGAN-GP.

Both models are identical three-layer 256-wide MLPs; only the objective (and therefore the
number of critic steps per generator step) differs. Everything is seeded, and the artefacts
this module writes are the only numbers the short notebook quotes.

Usage:
  python src/toy8.py                 # train, write artifacts/toy8/, render the PNGs
  python src/toy8.py --device cpu    # same on CPU
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import ganlib as G

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "toy8"
OUT_SEEDS = ROOT / "artifacts" / "toy8_seeds"

N_MODES = 8
RADIUS = 2.0
STD = 0.1            # spread of every mode, as in the WGAN illustration
HIT_RADIUS = 0.5     # a sample "covers" a mode when it lands this close to its centre
BIG_MODE_SHARE = 0.05

Z_DIM = 8
WIDTH = 256
DEPTH = 3            # hidden layers per network
BATCH = 256
LR = 1e-4
BETAS = (0.5, 0.9)
STEPS = 5000         # generator steps for both arms
N_CRITIC_GP = 5
LAMBDA_GP = 10.0
N_EVAL = 4000
EVAL_EVERY = 250
SEED = 0
Z_SEED = 1234

ARMS = {"bce": "BCE", "wgan_gp": "WGAN-GP"}


# ---------------------------------------------------------------- the data
def centers() -> np.ndarray:
    """Eight centres evenly spaced on a circle of radius RADIUS."""
    angles = np.arange(N_MODES) * 2.0 * np.pi / N_MODES
    return np.stack([RADIUS * np.cos(angles), RADIUS * np.sin(angles)], axis=1)


def sample_real(n: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    centre = torch.tensor(centers(), dtype=torch.float32)
    index = torch.randint(0, N_MODES, (n,), generator=generator)
    noise = torch.randn(n, 2, generator=generator) * STD
    return (centre[index] + noise).to(device)


# ---------------------------------------------------------------- the models
def _mlp(n_in: int, n_out: int, activation: type[nn.Module]) -> nn.Sequential:
    """DEPTH hidden layers of WIDTH units: the same shape for the generator and the critic."""
    widths = [n_in] + [WIDTH] * DEPTH
    layers: list[nn.Module] = []
    for fan_in, fan_out in zip(widths[:-1], widths[1:]):
        layers += [nn.Linear(fan_in, fan_out), activation()]
    return nn.Sequential(*layers, nn.Linear(WIDTH, n_out))


def make_g() -> nn.Module:
    return _mlp(Z_DIM, 2, nn.ReLU)


def make_d() -> nn.Module:
    def leaky() -> nn.Module:
        return nn.LeakyReLU(0.2, True)

    return _mlp(2, 1, leaky)


# ---------------------------------------------------------------- the one metric that matters
def mode_stats(points: np.ndarray) -> dict:
    """How many of the eight modes received samples, and how lopsided the split is."""
    distance = np.linalg.norm(points[:, None, :] - centers()[None, :, :], axis=2)
    nearest = distance.argmin(axis=1)
    closest = distance.min(axis=1)
    share = np.bincount(nearest, minlength=N_MODES) / len(points)
    covered = (distance.min(axis=0) <= HIT_RADIUS)
    return {
        "modes_covered": int(covered.sum()),
        "modes_big": int((share > BIG_MODE_SHARE).sum()),
        "max_mode_share": float(share.max()),
        "outside_share": float((closest > HIT_RADIUS).mean()),
        "share": [round(float(value), 4) for value in share],
    }


# ---------------------------------------------------------------- training
def _gradient_penalty(disc, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, device=real.device)
    mixed = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    score = disc(mixed)
    gradient = torch.autograd.grad(score, mixed, torch.ones_like(score), create_graph=True)[0]
    return ((gradient.norm(2, dim=1) - 1.0) ** 2).mean()


def train(arm: str, device: torch.device, steps: int = STEPS, verbose: bool = True,
          seed: int = SEED) -> dict:
    """Train one arm from scratch. Returns its evaluation samples, history and timing.

    ``seed`` sets the model init and the stream of real batches; the evaluation latent bank
    stays fixed (Z_SEED), so seeds differ in the fit, not in the points they are scored on.
    """
    torch.manual_seed(seed)
    data_rng = torch.Generator().manual_seed(seed)
    gen, disc = make_g().to(device), make_d().to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=LR, betas=BETAS)
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR, betas=BETAS)
    bce = nn.functional.binary_cross_entropy_with_logits
    z_eval = torch.randn(N_EVAL, Z_DIM,
                         generator=torch.Generator().manual_seed(Z_SEED)).to(device)
    n_critic = 1 if arm == "bce" else N_CRITIC_GP
    history: dict[str, list] = {"step": [], "max_mode_share": [], "modes_covered": []}

    started = time.time()
    for step in range(1, steps + 1):
        for _ in range(n_critic):
            real = sample_real(BATCH, data_rng, device)
            fake = gen(torch.randn(BATCH, Z_DIM, device=device)).detach()
            if arm == "bce":
                loss_d = (bce(disc(real), torch.ones(BATCH, 1, device=device))
                          + bce(disc(fake), torch.zeros(BATCH, 1, device=device)))
            else:
                loss_d = (disc(fake).mean() - disc(real).mean()
                          + LAMBDA_GP * _gradient_penalty(disc, real, fake))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
        z = torch.randn(BATCH, Z_DIM, device=device)
        if arm == "bce":
            loss_g = bce(disc(gen(z)), torch.ones(BATCH, 1, device=device))
        else:
            loss_g = -disc(gen(z)).mean()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        if step % EVAL_EVERY == 0 or step == steps:
            stats = mode_stats(gen(z_eval).detach().cpu().numpy())
            history["step"].append(step)
            history["max_mode_share"].append(round(stats["max_mode_share"], 4))
            history["modes_covered"].append(stats["modes_covered"])
            if verbose:
                print(f"[{arm}] step {step:5d}  modes {stats['modes_covered']}/8  "
                      f"top share {stats['max_mode_share']:.2f}", flush=True)

    if device.type == "mps":
        torch.mps.synchronize()
    seconds = time.time() - started
    with torch.no_grad():
        samples = gen(z_eval).cpu().numpy()
    final = mode_stats(samples)
    return {"arm": arm, "display": ARMS[arm], "steps": steps, "n_critic": n_critic,
            "seconds": round(seconds, 1), "final": final, "history": history,
            "samples": samples, "loss_g": float(loss_g.detach()), "loss_d": float(loss_d.detach())}


def train_all(device: str | None = None, steps: int = STEPS, verbose: bool = True) -> dict:
    """Both arms plus the real reference, written to artifacts/toy8/."""
    device = G.resolve_device(device)
    print(f"toy8: {G.device_report(device)}", flush=True)
    runs = {arm: train(arm, device, steps=steps, verbose=verbose) for arm in ARMS}
    torch.manual_seed(0)
    real = sample_real(N_EVAL, torch.Generator().manual_seed(7), torch.device("cpu")).numpy()

    art = {
        "protocol": {
            "modes": N_MODES, "radius": RADIUS, "std": STD, "hit_radius": HIT_RADIUS,
            "big_mode_share": BIG_MODE_SHARE, "z_dim": Z_DIM, "width": WIDTH,
            "hidden_layers": DEPTH, "batch_size": BATCH, "lr": LR, "betas": list(BETAS),
            "steps": steps, "n_critic_gp": N_CRITIC_GP, "lambda_gp": LAMBDA_GP,
            "n_eval": N_EVAL, "seed": SEED, "z_seed": Z_SEED, "device": device.type,
        },
        "real": {"stats": mode_stats(real)},
        "models": {arm: {key: value for key, value in run.items()
                         if key not in ("samples", "history")} | {"history": run["history"]}
                   for arm, run in runs.items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "toy8_samples.npz", real=real,
                        bce=runs["bce"]["samples"], wgan_gp=runs["wgan_gp"]["samples"])
    (OUT / "toy8.json").write_text(json.dumps(art, indent=1, ensure_ascii=False))
    art_with_samples = load()
    plot_samples(art_with_samples)
    plot_curves(art_with_samples)
    return art_with_samples


# ---------------------------------------------------------------- several seeds
COLORS = {"bce": "#c53030", "wgan_gp": "#2b6cb0"}


def _agg(values: list[float]) -> dict:
    """Mean, spread and per-seed values; std is the sample std (ddof=1)."""
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "min": float(arr.min()),
            "max": float(arr.max()), "per_seed": [float(v) for v in arr]}


def train_seeds(seeds, device: str | None = None, steps: int = STEPS,
                verbose: bool = True) -> dict:
    """Both arms on every seed, aggregated over seeds. Writes artifacts/toy8_seeds/.

    The evaluation latent bank is common to all seeds (Z_SEED), so a seed-to-seed spread is
    a spread of the learned distribution, not of the points it is measured on.
    """
    dev = G.resolve_device(device)
    seeds = [int(seed) for seed in seeds]
    print(f"toy8 seeds={seeds}: {G.device_report(dev)}", flush=True)
    runs = {}
    for seed in seeds:
        for arm in ARMS:
            print(f"\n=== seed {seed}: {ARMS[arm]} ===", flush=True)
            runs[(seed, arm)] = train(arm, dev, steps=steps, verbose=verbose, seed=seed)
    real = sample_real(N_EVAL, torch.Generator().manual_seed(7), torch.device("cpu")).numpy()

    art = {
        "protocol": {
            "modes": N_MODES, "radius": RADIUS, "std": STD, "hit_radius": HIT_RADIUS,
            "big_mode_share": BIG_MODE_SHARE, "z_dim": Z_DIM, "width": WIDTH,
            "hidden_layers": DEPTH, "batch_size": BATCH, "lr": LR, "betas": list(BETAS),
            "steps": steps, "n_critic_gp": N_CRITIC_GP, "lambda_gp": LAMBDA_GP,
            "n_eval": N_EVAL, "seeds": seeds, "z_seed": Z_SEED,
            "device": dev.type, "device_report": G.device_report(dev),
        },
        "real": {"stats": mode_stats(real)},
        "arms": {},
    }
    for arm in ARMS:
        per_seed = {}
        for seed in seeds:
            run = runs[(seed, arm)]
            per_seed[str(seed)] = {"final": run["final"], "history": run["history"],
                                   "seconds": run["seconds"], "loss_g": run["loss_g"],
                                   "loss_d": run["loss_d"]}
        art["arms"][arm] = {
            "display": ARMS[arm],
            "seeds": per_seed,
            "agg": {
                "max_mode_share": _agg([per_seed[str(s)]["final"]["max_mode_share"] for s in seeds]),
                "outside_share": _agg([per_seed[str(s)]["final"]["outside_share"] for s in seeds]),
                "modes_covered": _agg([per_seed[str(s)]["final"]["modes_covered"] for s in seeds]),
                "modes_big": _agg([per_seed[str(s)]["final"]["modes_big"] for s in seeds]),
                "seconds": _agg([per_seed[str(s)]["seconds"] for s in seeds]),
                "share": [_agg([per_seed[str(s)]["final"]["share"][i] for s in seeds])
                          for i in range(N_MODES)],
            },
        }

    OUT_SEEDS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_SEEDS / "toy8_seeds_samples.npz", real=real,
                        **{f"{arm}_{seed}": runs[(seed, arm)]["samples"]
                           for seed in seeds for arm in ARMS})
    (OUT_SEEDS / "toy8_seeds.json").write_text(
        json.dumps(art, indent=1, ensure_ascii=False))
    return load_seeds()


def load_seeds() -> dict:
    """Artefacts of train_seeds plus the sample arrays needed for the figure."""
    art = json.loads((OUT_SEEDS / "toy8_seeds.json").read_text())
    blob = np.load(OUT_SEEDS / "toy8_seeds_samples.npz")
    art["samples"] = {key: blob[key] for key in blob.files}
    return art


def plot_seed_samples(art: dict, path: Path | None = None):
    """One panel per (arm, seed), with the eight true centres marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = art["protocol"]["seeds"]
    limit = RADIUS + 3 * STD + 0.9
    fig, axes = plt.subplots(2, len(seeds), figsize=(3.5 * len(seeds), 6.8),
                             sharex=True, sharey=True, squeeze=False)
    for row, arm in enumerate(("bce", "wgan_gp")):
        for col, seed in enumerate(seeds):
            ax = axes[row][col]
            points = art["samples"][f"{arm}_{seed}"]
            stats = art["arms"][arm]["seeds"][str(seed)]["final"]
            ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.5, color=COLORS[arm])
            ax.scatter(centers()[:, 0], centers()[:, 1], s=60, facecolor="none",
                       edgecolor="#c53030", linewidth=1.2)
            ax.set_title(f"{ARMS[arm]}, сид {seed}: покрыто {stats['modes_covered']}/8, "
                         f"топ {stats['max_mode_share']:.2f}", fontsize=9)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path or (OUT_SEEDS / "toy8_seeds_samples.png"), dpi=150)
    return fig


def plot_seed_curves(art: dict, path: Path | None = None):
    """Mean over seeds with the min-max band and the individual runs behind it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = art["protocol"]["seeds"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    for arm in ("bce", "wgan_gp"):
        histories = [art["arms"][arm]["seeds"][str(seed)]["history"] for seed in seeds]
        steps = histories[0]["step"]
        for key, ax in (("modes_covered", axes[0]), ("max_mode_share", axes[1])):
            values = np.array([h[key] for h in histories], dtype=float)
            ax.plot(steps, values.mean(0), color=COLORS[arm], linewidth=1.8,
                    label=f"{ARMS[arm]}: среднее по {len(seeds)} сидам")
            if len(seeds) > 1:
                ax.fill_between(steps, values.min(0), values.max(0),
                                color=COLORS[arm], alpha=0.15, linewidth=0)
            for history in histories:
                ax.plot(history["step"], history[key], color=COLORS[arm],
                        linewidth=0.7, alpha=0.45)
    axes[0].set_ylabel("покрыто мод из 8")
    axes[1].set_ylabel("доля крупнейшей моды")
    axes[1].axhline(BIG_MODE_SHARE, color="#c53030", linestyle="--", linewidth=1,
                    label="порог 5%")
    for ax in axes:
        ax.set_xlabel("шаг генератора")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path or (OUT_SEEDS / "toy8_seeds_curves.png"), dpi=150)
    return fig


# ---------------------------------------------------------------- reading back
def load() -> dict:
    art = json.loads((OUT / "toy8.json").read_text())
    blob = np.load(OUT / "toy8_samples.npz")
    art["samples"] = {key: blob[key] for key in blob.files}
    return art


def load_or_train(device: str | None = None, verbose: bool = True) -> dict:
    """The notebook path: reuse the artefacts when they are there, train only if they are not."""
    if (OUT / "toy8.json").exists():
        return load()
    return train_all(device=device, verbose=verbose)


# ---------------------------------------------------------------- figures
def plot_samples(art: dict, path: Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limit = RADIUS + 3 * STD + 0.9
    panels = [("real", "реальные 8 мод", art["samples"]["real"]),
              ("bce", "BCE", art["samples"]["bce"]),
              ("wgan_gp", "WGAN-GP", art["samples"]["wgan_gp"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, sharey=True)
    for ax, (key, title, points) in zip(axes, panels):
        stats = art["real"]["stats"] if key == "real" else art["models"][key]["final"]
        ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.5, color="#2b6cb0")
        ax.scatter(centers()[:, 0], centers()[:, 1], s=70, facecolor="none",
                   edgecolor="#c53030", linewidth=1.4)
        ax.set_title(f"{title}: покрыто {stats['modes_covered']}/8, "
                     f"топ-мода {stats['max_mode_share']:.2f}", fontsize=10)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    out = path or (OUT / "toy8_samples.png")
    fig.savefig(out, dpi=150)
    return fig


def plot_curves(art: dict, path: Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for arm, run in art["models"].items():
        axes[0].plot(run["history"]["step"], run["history"]["modes_covered"],
                     label=run["display"], linewidth=1.6)
        axes[1].plot(run["history"]["step"], run["history"]["max_mode_share"],
                     label=run["display"], linewidth=1.6)
    axes[0].set_ylabel("покрыто мод из 8")
    axes[1].set_ylabel("доля крупнейшей моды")
    axes[1].axhline(BIG_MODE_SHARE, color="#c53030", linestyle="--", linewidth=1,
                    label="порог 5%")
    for ax in axes:
        ax.set_xlabel("шаг генератора")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = path or (OUT / "toy8_curves.png")
    fig.savefig(out, dpi=150)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda", "rocm"])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", default="", help="comma-separated seeds, e.g. 42,43,44")
    args = ap.parse_args()
    if args.seeds:
        art = train_seeds([int(s) for s in args.seeds.split(",") if s.strip()],
                          device=args.device, steps=args.steps)
        for arm in ("bce", "wgan_gp"):
            agg = art["arms"][arm]["agg"]
            print(f"{ARMS[arm]:8s} modes {agg['modes_covered']['mean']:.1f}/8  "
                  f"top {agg['max_mode_share']['mean']:.3f} ± {agg['max_mode_share']['std']:.3f}  "
                  f"outside {agg['outside_share']['mean']:.3f}")
        print("written " + str(OUT_SEEDS))
        return
    art = train_all(device=args.device, steps=args.steps)
    print("")
    for arm in ("bce", "wgan_gp"):
        run = art["models"][arm]
        print(f"{run['display']:8s} modes {run['final']['modes_covered']}/8  "
              f"big {run['final']['modes_big']}/8  top {run['final']['max_mode_share']:.3f}  "
              f"outside {run['final']['outside_share']:.3f}  {run['seconds']:.0f}s")
    print("written " + str(OUT))


if __name__ == "__main__":
    main()
