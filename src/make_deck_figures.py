"""Figures for the Part 2 deck (Fashion-MNIST): reads the finished three-seed run only.

Writes PNGs to figures/. Nothing is retrained here; every number comes from
artifacts/notebook_seeds_run/run.json and the final generator samples.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = json.loads((ROOT / "artifacts" / "notebook_seeds_run" / "run.json").read_text())
FID_JSON = json.loads((ROOT / "artifacts" / "metrics" / "fid_fashion_factorial.json").read_text())
RAW = ROOT / "artifacts" / "data" / "FashionMNIST" / "raw"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

SEEDS = ["42", "43", "44"]
CLASSES = ["футболка", "брюки", "пуловер", "платье", "пальто",
           "сандалии", "рубашка", "кроссовки", "сумка", "ботинок"]
BCE_C, WGAN_C = "#B03A2E", "#1F4E79"
GRAY, LIGHT = "#8C8C8C", "#D9D9D9"
NOISE_FLOOR = FID_JSON["noise_floor"]["fid_real_vs_real"]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": "#7A7A7A",
    "axes.linewidth": 0.7,
    "xtick.color": "#4A4A4A",
    "ytick.color": "#4A4A4A",
    "text.color": "#1A1A1A",
    "grid.color": "#E4E4E4",
    "grid.linewidth": 0.6,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def history(arm: str, seed: str) -> dict:
    return RUN["arms"][arm]["seeds"][seed]["history"]


def final(arm: str, seed: str, key: str) -> float:
    return float(history(arm, seed)[key][-1])


def final_shares(arm: str, seed: str) -> np.ndarray:
    return np.asarray(history(arm, seed)["class_share"][-1], dtype=float)


def read_idx_images(path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        magic = int.from_bytes(handle.read(4), "big")
        count = int.from_bytes(handle.read(4), "big")
        rows = int.from_bytes(handle.read(4), "big")
        cols = int.from_bytes(handle.read(4), "big")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    assert magic == 2051
    return data.reshape(count, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        magic = int.from_bytes(handle.read(4), "big")
        count = int.from_bytes(handle.read(4), "big")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    assert magic == 2049
    return data.reshape(count)


def montage(images: np.ndarray, cols: int = 8, gap: int = 2) -> np.ndarray:
    rows = len(images) // cols
    height, width = images.shape[1:]
    canvas = np.full((rows * height + (rows - 1) * gap,
                      cols * width + (cols - 1) * gap), 255, dtype=np.uint8)
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        top, left = row * (height + gap), col * (width + gap)
        canvas[top:top + height, left:left + width] = image
    return canvas


def generator_samples(arm: str, seed: str) -> np.ndarray:
    import torch
    path = ROOT / "artifacts" / "notebook_seeds_run" / f"{arm}_seed{seed}" / "final.pt"
    return torch.load(path, map_location="cpu", weights_only=False)["final64"]


def real_samples() -> np.ndarray:
    images = read_idx_images(RAW / "t10k-images-idx3-ubyte")
    labels = read_idx_labels(RAW / "t10k-labels-idx1-ubyte")
    rng = np.random.default_rng(0)
    picks: list[int] = []
    for cls in range(10):
        pool = np.flatnonzero(labels == cls)
        picks.extend(int(value) for value in rng.choice(pool, 6, replace=False))
    picks.extend(int(value) for value in rng.choice(len(images), 4, replace=False))
    rng.shuffle(picks)
    return images[picks[:64]]


def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)
    print("written", OUT / name)


def strip(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_samples() -> None:
    panels = [
        ("реальные данные", montage(real_samples())),
        ("BCE, сид 42", montage(generator_samples("bce", "42"))),
        ("WGAN-GP, сид 42", montage(generator_samples("wgan_gp", "42"))),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5))
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.01, wspace=0.06)
    save(fig, "samples_compare.png")


def fig_real() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.imshow(montage(real_samples()), cmap="gray", vmin=0, vmax=255,
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save(fig, "samples_real.png")


def fig_class_share() -> None:
    bce = np.array([final_shares("bce", seed) for seed in SEEDS])
    wgan = np.array([final_shares("wgan_gp", seed) for seed in SEEDS])
    positions = np.arange(len(CLASSES))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.45), sharex=True)
    for ax, values, color, title in ((axes[0], bce, BCE_C, "BCE: доля платья 0.39"),
                                     (axes[1], wgan, WGAN_C, "WGAN-GP: доли 0.08–0.13")):
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
        ax.barh(positions, mean, height=0.62, color=color, alpha=0.55,
                xerr=std, error_kw={"ecolor": "#5F5F5F", "lw": 0.7, "capsize": 2})
        top = int(mean.argmax())
        ax.barh(positions[top], mean[top], height=0.62, color=color)
        ax.axvline(0.10, color=GRAY, linestyle=(0, (4, 3)), linewidth=1)
        for y, value, spread in zip(positions, mean, std):
            ax.text(value + spread + 0.012, y,
                    f"{value:.2f}" if value >= 0.01 else "0",
                    va="center", fontsize=7.5, color="#3A3A3A")
        ax.set_xlim(0, 0.50)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
        ax.set_yticks(positions)
        ax.set_title(title, loc="left", fontsize=10, pad=6)
        ax.grid(axis="x", color="#ECECEC", linewidth=0.6)
        ax.set_axisbelow(True)
        strip(ax)
    axes[0].set_yticklabels(CLASSES, fontsize=8.5)
    axes[1].set_yticklabels([])
    axes[0].text(0.105, positions[-1] - 0.55, "реальные данные — 0.10",
                 fontsize=8, color=GRAY)
    fig.supxlabel("доля сэмплов класса (среднее ± std по трём сидам)", fontsize=8.5,
                  color="#4A4A4A", y=0.01)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.90, bottom=0.16, wspace=0.08)
    save(fig, "class_share.png")


def fig_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.25))
    for ax, key, ylabel, threshold in (
            (axes[0], "max_class_share", "доля топ-класса · порог 0,35", 0.35),
            (axes[1], "mode_coverage", "покрытие мод · порог 0,75", 0.75)):
        for arm, color, label in (("bce", BCE_C, "BCE"), ("wgan_gp", WGAN_C, "WGAN-GP")):
            curves = np.array([history(arm, seed)[key] for seed in SEEDS], dtype=float)
            epochs = np.asarray(history(arm, SEEDS[0])["epoch"])
            ax.plot(epochs, curves.mean(axis=0), color=color, linewidth=1.9, label=label)
            ax.fill_between(epochs, curves.min(axis=0), curves.max(axis=0),
                            color=color, alpha=0.14, linewidth=0)
            for curve in curves:
                ax.plot(epochs, curve, color=color, linewidth=0.6, alpha=0.3)
        ax.axhline(threshold, color=GRAY, linestyle=(0, (4, 3)), linewidth=1)
        ax.set_xlim(1, 80)
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("эпоха")
        ax.set_title(ylabel, loc="left", fontsize=10, pad=6)
        ax.grid(axis="y", color="#ECECEC", linewidth=0.6)
        ax.set_axisbelow(True)
        strip(ax)
    axes[0].legend(loc="upper right", fontsize=8.5, frameon=False)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.15, wspace=0.14)
    save(fig, "curves.png")


def fig_seeds() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 2.95))
    for ax, key, ylabel, fmt, offset in (
            (axes[0], "max_class_share", "доля топ-класса", "{:.3f}", 0.025),
            (axes[1], "mode_coverage", "покрытие мод", "{:.2f}", 0.035)):
        x = np.arange(len(SEEDS))
        for shift, arm, color, label in ((-0.07, "bce", BCE_C, "BCE"),
                                          (0.07, "wgan_gp", WGAN_C, "WGAN-GP")):
            values = [final(arm, seed, key) for seed in SEEDS]
            ax.plot(x + shift, values, "o", color=color, markersize=7, label=label,
                    zorder=3)
            ax.hlines(np.mean(values), -0.32, 2.32, color=color, linewidth=1,
                      linestyle=":", alpha=0.75)
            for xi, value in zip(x + shift, values):
                ax.text(xi, value + offset, fmt.format(value), ha="center", va="bottom",
                        fontsize=7.5, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([f"сид {seed}" for seed in SEEDS])
        ax.set_xlim(-0.45, 2.45)
        ax.set_ylim(0, 1.08 if key == "mode_coverage" else 0.52)
        ax.set_title(ylabel, loc="left", fontsize=10, pad=6)
        ax.grid(axis="y", color="#ECECEC", linewidth=0.6)
        ax.set_axisbelow(True)
        strip(ax)
    axes[0].text(2.42, final("bce", "44", "max_class_share") + 0.02, "BCE",
                 color=BCE_C, fontsize=8.5, va="bottom")
    axes[0].text(2.42, final("wgan_gp", "44", "max_class_share") + 0.02, "WGAN-GP",
                 color=WGAN_C, fontsize=8.5, va="bottom")
    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.16, wspace=0.14)
    save(fig, "seeds.png")


def fig_fid() -> None:
    values = [np.mean([RUN["arms"][arm]["seeds"][seed]["fid"] for seed in SEEDS])
              for arm in ("bce", "wgan_gp")]
    spreads = [np.std([RUN["arms"][arm]["seeds"][seed]["fid"] for seed in SEEDS], ddof=1)
               for arm in ("bce", "wgan_gp")]
    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    bars = ax.bar([0, 1], values, width=0.52, color=[BCE_C, WGAN_C], alpha=0.85,
                  yerr=spreads, capsize=3,
                  error_kw={"ecolor": "#5F5F5F", "lw": 0.8})
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(NOISE_FLOOR, color=GRAY, linestyle=(0, (4, 3)), linewidth=1,
               label=f"порог шума {NOISE_FLOOR:.1f}".replace(".", ","))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BCE", "WGAN-GP"])
    ax.set_ylabel("FID")
    ax.set_ylim(0, 95)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", color="#ECECEC", linewidth=0.6)
    ax.set_axisbelow(True)
    strip(ax)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.16)
    save(fig, "fid.png")


if __name__ == "__main__":
    fig_samples()
    fig_real()
    fig_class_share()
    fig_curves()
    fig_seeds()
    fig_fid()
