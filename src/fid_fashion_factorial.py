"""FID for the three capacity-matched Fashion-MNIST arms, written to its own file.

This is src/fid.py's protocol applied to the three arms of the capacity factorial (BCE 40,
BCE 80, WGAN-GP 40), which the earlier FID pass predates. Nothing is retrained and nothing
under artifacts/fashion/ or artifacts/metrics/fid_stats/ is written: the cached real-slice
statistics are read, the generated statistics are computed in memory, and the only output is
artifacts/metrics/fid_fashion_factorial.json.

Usage:
  TORCH_HOME=<workspace>/data/torch_home python src/fid_fashion_factorial.py [--device mps]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fid
import ganlib as G

ROOT = Path(__file__).resolve().parent.parent
FASHION = ROOT / "artifacts" / "fashion"
OUT = ROOT / "artifacts" / "metrics" / "fid_fashion_factorial.json"

DATASET = "fashion"
ARM_ORDER = ["fashion_strongD", "fashion_strongD_80ep", "fashion_strongD_wgan"]
N = fid.N_SAMPLES


def layer(name: str) -> dict:
    return {"checkpoint": str(FASHION / name / "models" / (name + "_checkpoint.pt")),
            "history": str(FASHION / name / "metrics" / (name + "_history.json"))}


def real_stats() -> dict:
    """The two 5000-image test slices fid.py already cached — the same reference it used."""
    out = {}
    for tag in ("a", "b"):
        path = fid.STATS_DIR / ("real_" + DATASET + "_" + tag + "_n" + str(N) + "_"
                                + str(fid.FEATURE_DIM) + "d.npz")
        if not path.exists():
            raise SystemExit("missing cached real statistics: " + str(path))
        blob = np.load(path)
        out[tag] = (blob["mu"], blob["sigma"].astype(np.float64))
    return out


def history_summary(path: Path) -> dict:
    h = json.loads(path.read_text())
    return {
        "epochs": len(h["epoch"]),
        "params_g": h["params_g"],
        "params_d": h["params_d"],
        "d_over_g": h["params_d"] / h["params_g"],
        "max_class_share_last": h["max_class_share"][-1],
        "mode_coverage_last": h["mode_coverage"][-1],
        "class_entropy_last": h["class_entropy"][-1],
        "intra_class_spread_last": h["intra_class_spread"][-1],
        "d_margin_last": h["d_margin"][-1],
        "zero_classes_last": [i for i, share in enumerate(h["class_share"][-1]) if share < 0.01],
        "config": h["config"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["cpu", "mps"])
    ap.add_argument("--n", type=int, default=N)
    args = ap.parse_args()
    device = torch.device(args.device)

    started = time.time()
    model = fid.make_inception(device)
    print("[setup] inception ready on " + args.device, flush=True)
    real = real_stats()
    floor = fid.frechet_distance(real["a"][0], real["a"][1], real["b"][0], real["b"][1])
    print(f"[noise floor] real-vs-real FID = {floor:.2f}", flush=True)

    result = {
        "protocol": dict(fid_protocol := {
            "n_samples": args.n,
            "batch_size": fid.BATCH,
            "feature_dim": fid.FEATURE_DIM,
            "inception": "pytorch_fid.inception.InceptionV3, block_idx 3, resize+normalize",
            "distance": "Fréchet distance via eigendecomposition of the symmetric form",
            "preprocess": "image in [-1,1] -> (x+1)/2 -> repeat to 3 channels -> 299x299",
            "real_reference": "cached statistics of test split indices 0.."
                              + str(args.n - 1) + " (same slices as fid.json)",
            "latent_seed": fid.LATENT_SEED,
            "device": args.device,
        }),
        "noise_floor": {
            "dataset": DATASET,
            "fid_real_vs_real": floor,
            "n_each": args.n,
            "note": "два непересекающихся среза по " + str(args.n)
                    + " изображений тестовой части; разница меньше этого порога не значит ничего",
        },
        "arms": {},
        "notes": [
            "Абсолютные значения FID здесь несравнимы с публикуемыми: 28x28 в оттенках серого "
            "растянуто до 299x299 RGB, а Inception обучен на ImageNet.",
            "Сравнивать можно только эти три плеча между собой и с порогом шума "
            "FD(real↔real) на Fashion-MNIST.",
            "Новый файл, чтобы не перезаписывать artifacts/metrics/fid.json прошлого прогона.",
        ],
    }

    for name in ARM_ORDER:
        paths = layer(name)
        gen, cfg, err = fid.load_generator(paths["checkpoint"])
        if err:
            raise SystemExit(f"[{name}] {err}")
        z = fid.latent(args.n, int(cfg.get("z_dim", 100)))
        with torch.no_grad():
            batches = [gen(z[start:start + fid.BATCH]) for start in range(0, args.n, fid.BATCH)]
        mu_g, sigma_g = fid.stats_of(fid.features_of_batches(batches, model, device))
        value = fid.frechet_distance(mu_g, sigma_g, real["a"][0], real["a"][1])
        row = {"fid": value, "dataset": DATASET, **history_summary(Path(paths["history"]))}
        result["arms"][name] = row
        print(f"[{name}] FID = {value:8.2f}  top={row['max_class_share_last']:.3f}  "
              f"cov={row['mode_coverage_last']:.2f}", flush=True)
        OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))

    collapsed = [n for n in ARM_ORDER if result["arms"][n]["max_class_share_last"] >= 0.35]
    best = min(ARM_ORDER, key=lambda n: result["arms"][n]["fid"])
    result["absurdity_check"] = {
        "collapsed_arms": collapsed,
        "lowest_fid_arm": best,
        "verdict": ("коллапснувшее плечо получило лучший FID — так и записано, без подгонки"
                    if best in collapsed else
                    "FID у коллапснувших плеч не лучший: совпадает с ожиданием"),
        "note": "FID награждает гладкие 'типичные' сэмплы и не различает коллапс: одно "
                "доминирующее платье фотографично и близко к среднему, но разнообразия нет — "
                "поэтому рядом стоят покрытие и энтропия классов.",
    }
    result["seconds_total"] = round(time.time() - started, 1)
    result["generated_utc"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print("")
    print(f"{'arm':<24s}{'FID':>8s}{'top':>7s}{'cov':>6s}{'H':>7s}{'intra':>8s}{'margin':>8s}")
    for name in ARM_ORDER:
        row = result["arms"][name]
        print(f"{name:<24s}{row['fid']:>8.2f}{row['max_class_share_last']:>7.3f}"
              f"{row['mode_coverage_last']:>6.2f}{row['class_entropy_last']:>7.3f}"
              f"{row['intra_class_spread_last']:>8.2f}{row['d_margin_last']:>8.2f}")
    print(f"noise floor = {floor:.2f}   written {OUT} in {result['seconds_total']:.0f}s")


if __name__ == "__main__":
    main()
