"""Assemble notebooks/mode_collapse_gan_seeds.ipynb: the same experiment, three seeds, GPU.

The notebook trains both halves itself, as the single-seed notebook does: the synthetic
8-Gaussian pair and the two Fashion-MNIST arms, each on SEEDS = [42, 43, 44]. Per-epoch
history and checkpoints are written to artifacts/notebook_seeds_<tag>/, so a long run can be
resumed; the prose numbers are formatted from artifacts/notebook_seeds_run/run.json at
``--finalize`` time, without re-running the training.

Usage:
  python src/build_notebook_seeds.py                  # build the notebook (80 epochs)
  python src/build_notebook_seeds.py --epochs 2 --tag test --out notebooks/_test.ipynb
  python src/build_notebook_seeds.py --finalize        # bake numbers into markdown
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "mode_collapse_gan_seeds.ipynb"
RUN_JSON = ROOT / "artifacts" / "notebook_seeds_run" / "run.json"
TOY_JSON = ROOT / "artifacts" / "toy8_seeds" / "toy8_seeds.json"
REF_BCE80 = ROOT / "artifacts" / "fashion" / "fashion_strongD_80ep" / "metrics" / \
    "fashion_strongD_80ep_history.json"
REF_WGAN = ROOT / "artifacts" / "fashion" / "fashion_strongD_wgan" / "metrics" / \
    "fashion_strongD_wgan_history.json"

SEEDS = [42, 43, 44]
RU_CLASS = {"T-shirt/top": "футболка", "Trouser": "брюки", "Pullover": "пуловер",
            "Dress": "платье", "Coat": "пальто", "Sandal": "сандалии", "Shirt": "рубашка",
            "Sneaker": "кроссовки", "Bag": "сумка", "Ankle boot": "ботинок"}


# ------------------------------------------------------------------ numbers
def _fmt(stats: dict | None, fmt: str = "{:.3f}") -> str:
    if stats is None:
        return "—"
    return f"{fmt.format(stats['mean'])} ± {fmt.format(stats['std'])}"


def _final(history: dict, key: str) -> float:
    return float(history[key][-1])


def _agg(values) -> dict:
    values = list(values)
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 \
        if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "min": min(values), "max": max(values),
            "per_seed": values}


def load_stats() -> dict | None:
    """Everything the prose needs; None until the notebook has actually run."""
    if not RUN_JSON.exists():
        return None
    run = json.loads(RUN_JSON.read_text())
    toy = json.loads(TOY_JSON.read_text()) if TOY_JSON.exists() else None
    stats = {"config": run["config"], "device": run.get("device", ""), "arms": {}, "toy": toy}
    for arm in ("bce", "wgan_gp"):
        seeds = run["arms"][arm]["seeds"]
        seed_ids = sorted(seeds, key=int)
        finals = [seeds[s] for s in seed_ids]
        stats["arms"][arm] = {
            "seeds": seed_ids,
            "top": _agg(_final(f["history"], "max_class_share") for f in finals),
            "coverage": _agg(_final(f["history"], "mode_coverage") for f in finals),
            "entropy": _agg(_final(f["history"], "class_entropy") for f in finals),
            "intra": _agg(_final(f["history"], "intra_class_spread") for f in finals),
            "margin": _agg(_final(f["history"], "d_margin") for f in finals),
            "fid": _agg(f["fid"] for f in finals),
            "seconds": _agg(f["seconds"] for f in finals),
            "final_fired": [bool(h["max_class_share"][-1] >= 0.35
                                 or h["mode_coverage"][-1] <= 0.75)
                            for h in (f["history"] for f in finals)],
            "fired_share": _agg(
                sum(1 for top, cov in zip(h["max_class_share"], h["mode_coverage"])
                    if top >= 0.35 or cov <= 0.75) / len(h["epoch"])
                for h in (f["history"] for f in finals)),
        }
        # collapse criterion: how many seeds collapse at the final epoch
        stats["arms"][arm]["collapsed_seeds"] = sum(stats["arms"][arm]["final_fired"])
    return stats


def load_toy_stats(stats: dict | None) -> dict | None:
    if stats is None:
        return None
    toy = stats["toy"]
    if not toy:
        return None
    return {
        "seeds": toy["protocol"]["seeds"],
        "bce": toy["arms"]["bce"]["agg"],
        "wgan_gp": toy["arms"]["wgan_gp"]["agg"],
        "real": toy["real"]["stats"],
    }


# ------------------------------------------------------------------ markdown
def md_intro(stats: dict | None) -> str:
    config = (stats or {}).get("config", {})
    device = (stats or {}).get("device", "GPU (ROCm/CUDA)")
    epochs = config.get("EPOCHS", 80)
    seeds = config.get("SEEDS", SEEDS)
    return f"""# Коллапс мод: тот же эксперимент на трёх сидах (ROCm/CUDA)

Это повтор эксперимента из `mode_collapse_gan.ipynb`: те же данные, тот же критик и те же
две функции потерь (обычный BCE против WGAN-GP), но каждое плечо обучается заново на
{len(seeds)} сидах — {', '.join(map(str, seeds))} — и все метрики приводятся как среднее ±
стандартное отклонение по сидам. Один сид не отличает «BCE коллапсирует» от «так выпало»:
на синтетике это видно по тому, какие моды теряются, на Fashion-MNIST — по разбросу доли
топ-класса и покрытия между прогонами.

Обучение идёт на {device}. Оба плеча получают одинаковый бюджет в {epochs} эпох; сид 42
совпадает с исходным одиночным прогоном, сиды 43 и 44 проверяют, не было ли это везением
одного запуска. Ноутбук сам обучает обе половины (синтетику и Fashion-MNIST) и сам пишет
историю в `artifacts/notebook_seeds_run/`, поэтому длинный прогон можно продолжить с чекпоинта.
"""


def md_toy(stats: dict | None) -> str:
    toy = load_toy_stats(stats)
    head = """## 1. Синтетика: 8 гауссиан, три сида

Та же постановка, что в разделе 1 основного ноутбука: 8 центров радиуса 2.0, σ = 0.1,
одинаковые MLP у генератора и критика, Adam с lr 0.0001 и betas (0.5, 0.9), batch 256,
5000 шагов генератора, сид меняется только у инициализации и потока реальных батчей.
Оценка идёт на общем для всех сидов латентном банке (4000 точек), поэтому разброс между
сидами — это разброс выученного распределения, а не разных точек замера."""
    if toy is None:
        return head + "\n\nЧисла появятся в таблице ниже после исполнения ячейки обучения."
    bce, wgp, real = toy["bce"], toy["wgan_gp"], toy["real"]
    bce_cover = ", ".join(f"{v:.0f}" for v in bce['modes_covered']['per_seed'])
    wgp_cover = ", ".join(f"{v:.0f}" for v in wgp['modes_covered']['per_seed'])
    worst = min(wgp['modes_covered']['per_seed'])
    note = ""
    if worst < 8:
        note = (f"\nНа синтетике разброс между сидами заметнее, чем на Fashion-MNIST: WGAN-GP "
                f"покрывает по сидам {wgp_cover} из 8, BCE — {bce_cover}; худший прогон WGAN-GP "
                f"оставляет {wgp['outside_share']['max']:.3f} точек вне мод. Трёх сидов здесь "
                f"мало, чтобы говорить об устойчивости WGAN-GP на кольце.")
    return head + f"""

**Результат.** BCE покрывает {bce['modes_covered']['mean']:.2f} из 8 мод в среднем
(по сидам {bce_cover}), крупнейшая мода забирает {_fmt(bce['max_mode_share'])} сэмплов,
а вне всех мод остаётся {_fmt(bce['outside_share'])}. WGAN-GP покрывает
{wgp['modes_covered']['mean']:.2f} из 8 (по сидам {wgp_cover}), крупнейшая мода —
{_fmt(wgp['max_mode_share'])}, вне мод — {_fmt(wgp['outside_share'])}.
У реальных данных доля крупнейшей моды {max(real['share']):.3f} при равномерных 1/8 = 0.125.
{note}"""


def md_fashion(stats: dict | None) -> str:
    config = (stats or {}).get("config", {})
    epochs = config.get("EPOCHS", 80)
    seeds = config.get("SEEDS", SEEDS)
    return f"""## 2. Fashion-MNIST: те же два плеча, три сида

Критик зафиксирован и одинаков у обоих плеч ({config.get('D_WIDTHS', (128, 256, 256))},
без BatchNorm, два stride-2 блока и stride-1 блок на 7×7), генератор — DCGAN
(128 → 64, z(100) → 7×7 → 14×14 → 28×28, Tanh). Различается ровно одно: BCE
(non-saturating `-log D(G(z))`) против WGAN-GP с λ = 10 и пятью шагами критика на шаг
генератора в обоих плечах. Бюджет — {epochs} эпох на плечо, сиды
{', '.join(map(str, seeds))}; сид 42 — исходный.

На ROCm свёртки критика считаются через `unfold + linear` (`ganlib.FastConv2d`): это та же
самая функция с теми же параметрами, но двойной backward, который нужен штрафу за градиент,
идёт через обычные GEMM вместо медленного пути MIOpen (замер: 12 мс против 80 мс на
двойной backward в штрафе). На CUDA/MPS/CPU используется штатный `nn.Conv2d`, поэтому ноутбук переносим.

Метрики мод считает тот же замороженный разметчик из `artifacts/models/mode_classifier.pt`:
класс сгенерированной картинки — `argmax` по 10 логитам, доля класса — гистограмма по
2048 сэмплам фиксированного латентного банка (общего для всех сидов). Критерий коллапса
зафиксирован заранее: доля топ-класса ≥ 0.35 **или** покрытие мод ≤ 0.75."""


def md_results(stats: dict | None) -> str:
    if stats is None:
        return """## Что показал повтор

Числа появятся здесь после того, как ноутбук исполнится: таблицы и графики ниже считаются
из `artifacts/notebook_seeds_run/run.json` и `artifacts/toy8_seeds/toy8_seeds.json`."""
    bce, wgp = stats["arms"]["bce"], stats["arms"]["wgan_gp"]
    epochs = stats["config"].get("EPOCHS", 80)
    seeds = stats["config"].get("SEEDS", SEEDS)
    bce_fired = bce["collapsed_seeds"]
    wgp_fired = wgp["collapsed_seeds"]
    toy = load_toy_stats(stats) or {}
    toy_cover = ", ".join(f"{v:.0f}" for v in (toy.get("wgan_gp", {}).get("modes_covered", {})
                                               .get("per_seed", []))) or "—"
    toy_bce_cover = ", ".join(f"{v:.0f}" for v in (toy.get("bce", {}).get("modes_covered", {})
                                                   .get("per_seed", []))) or "—"
    return f"""## Что показал повтор

Всё ниже — среднее ± std по сидам {', '.join(map(str, seeds))}, {epochs} эпох на плечо.

* **BCE, {epochs} эпох.** Доля топ-класса {_fmt(bce['top'])}, покрытие
  {_fmt(bce['coverage'], '{:.2f}')}, H = {_fmt(bce['entropy'])}, FID = {_fmt(bce['fid'], '{:.1f}')}.
  Критерий коллапса на финальной эпохе выполнен у {bce_fired} из {len(seeds)} сидов.
* **WGAN-GP, тот же критик и бюджет.** Доля топ-класса {_fmt(wgp['top'])}, покрытие
  {_fmt(wgp['coverage'], '{:.2f}')}, H = {_fmt(wgp['entropy'])}, FID = {_fmt(wgp['fid'], '{:.1f}')}.
  Критерий на финальной эпохе — у {wgp_fired} из {len(seeds)} сидов.
* **Синтетика.** Там сидов не хватает для устойчивого вывода: у WGAN-GP покрытие по сидам
  {toy_cover}, у BCE — {toy_bce_cover}; на Fashion-MNIST разрыв между плечами
  несравнимо стабильнее. Это ограничение трёх сидов, а не отмена результата.
* **Разброс между сидами.** У BCE стандартное отклонение доли топ-класса
  {bce['top']['std']:.3f} при среднем {bce['top']['mean']:.3f}; у WGAN-GP
  {wgp['top']['std']:.3f} при {wgp['top']['mean']:.3f}. Покрытие BCE
  {_fmt(bce['coverage'], '{:.2f}')} против {_fmt(wgp['coverage'], '{:.2f}')} у WGAN-GP:
  разрыв между функциями потерь больше, чем разброс между сидами внутри плеча, — это и есть
  ответ на вопрос, следствие коллапса функция потерь или конкретного запуска.
* **Внутриклассовый разброс** (насколько разнообразны картинки внутри доминирующего класса):
  BCE {_fmt(bce['intra'], '{:.1f}')}, WGAN-GP {_fmt(wgp['intra'], '{:.1f}')}. Margin D:
  {_fmt(bce['margin'], '{:.1f}')} против {_fmt(wgp['margin'], '{:.1f}')} — у BCE критик
  уверенно разделяет классы, у WGAN-GP почти не разделяет, и генератору невыгодно
  сбегаться в одну моду.
* **Время.** Один прогон BCE — {_fmt(bce['seconds'], '{:.0f}')} с, один прогон WGAN-GP —
  {_fmt(wgp['seconds'], '{:.0f}')} с; всё обучение шло на GPU через ROCm.
"""


def md_limits(stats: dict | None) -> str:
    return """## Ограничения

* **Три сида — это три, а не тридцать.** Среднее и std по трём точкам сами по себе шумные:
  std здесь показывает масштаб разброса, а не доверительный интервал. Утверждение «BCE
  коллапсирует» подкреплено тремя независимыми прогонами, но не оценкой вероятности.
* **Сиды 42/43/44, а не случайная выборка.** 42 — исходный сид основного ноутбука, 43 и 44
  добавлены, чтобы проверить устойчивость; это фиксированный список, а не случайные сиды из
  распределения.
* **ROCm-сборка и MIOpen.** PyTorch собран под ROCm 7.2, системный ROCm — 10.1; на длинных
  сериях мелких ядер включается `HSA_TOOLS_DISABLE_REGISTER=1` (иначе rocprofiler-sdk падает
  с segfault, ROCm issue #5071), а поиск алгоритмов MIOpen переведён в `FAST`. Числа не
  бит-в-бит совпадают с эталоном на MPS: другая карта, другие ядра, другая арифметика.
* **Свёртки через `unfold + linear`.** Математически это тот же `Conv2d` (проверено: максимум
  расхождения 9e-7 на выходе и 0 на градиенте), но порядок суммирования другой, поэтому
  траектории обучения отличаются от штатного `nn.Conv2d` в последних знаках.
* **FID** считается на 5000 сэмплах против кэша тестовой части, 28×28 растянуто до 299×299
  RGB; абсолютные значения несравнимы с публикуемыми, сравнимы только плечи между собой.
* **Сравнение с эталоном** `artifacts/fashion/` — это comparison с прогоном на MPS, а не
  независимая репликация: другой сид (там только 42), другое железо. Поэтому эталонные
  кривые на графиках — пунктир, а не истина.
"""


def md_toy_section(stats: dict | None) -> str:
    return md_toy(stats)


# ------------------------------------------------------------------ code cells
CODE_HEAD = '''import os
import sys
import time
import json
from io import BytesIO
from pathlib import Path

ROOT = next(p for p in Path.cwd().resolve().parents if (p / "src" / "toy8.py").exists())

# ROCm: keep MIOpen's find cache and the torch hub inside the repo (home may be read-only),
# prefer the fast solver search, and disable the HSA tool registration that segfaults on
# long runs of small kernels (ROCm issue #5071). All of this must happen before torch loads.
os.environ.setdefault("HSA_TOOLS_DISABLE_REGISTER", "1")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".jupyter"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch_home"))
os.environ.setdefault("MIOPEN_USER_DB_PATH", str(ROOT / ".miopen-db"))
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(ROOT / ".miopen-cache"))
for _path in (os.environ["MPLCONFIGDIR"], os.environ["MIOPEN_USER_DB_PATH"],
              os.environ["MIOPEN_CUSTOM_CACHE_DIR"]):
    Path(_path).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import Image, display

import ganlib as G
import toy8
import fid as FIDLIB


def show(fig):
    """Figure -> inline PNG; the notebook leaves nothing on disk from this."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buffer.getvalue()))


DEVICE = G.resolve_device()          # cuda on ROCm/CUDA, else mps, else cpu
print(G.device_report(DEVICE))
print(f"torch {torch.__version__}, HIP {torch.version.hip}, CUDA {torch.version.cuda}")
assert DEVICE.type == "cuda", "этот ноутбук рассчитан на GPU (ROCm или CUDA)"

SEEDS = [42, 43, 44]
TOY_STEPS = 5000
'''

CODE_TOY_TRAIN = '''started = time.time()
toy = toy8.train_seeds(SEEDS, device=DEVICE, steps=TOY_STEPS, verbose=False)
toy_seconds = time.time() - started
print(f"синтетика: оба плеча на {len(SEEDS)} сидах за {toy_seconds:.0f} с")
'''

CODE_TOY_TABLE = '''def cell(value, fmt="{:.3f}"):
    return fmt.format(value["mean"]) + " ± " + fmt.format(value["std"])

print(f"{'плечо':<9}{'покрыто мод':>16}{'топ-мода':>18}{'вне мод':>16}")
for arm in ("bce", "wgan_gp"):
    a = toy["arms"][arm]["agg"]
    print(f"{toy['arms'][arm]['display']:<9}{a['modes_covered']['mean']:>10.2f}/8   "
          f"{cell(a['max_mode_share']):>16}   {cell(a['outside_share']):>14}")
print("")
print("доля точек по модам, среднее ± std по сидам (порядок центров кольца):")
for arm in ("bce", "wgan_gp"):
    print(f"  {toy['arms'][arm]['display']:<9}" + " ".join(
        f"{value['mean']:.3f}" for value in toy["arms"][arm]["agg"]["share"]))
print(f"  {'реальные':<9}" + " ".join(f"{value:.3f}" for value in toy["real"]["stats"]["share"]))
print("")
for arm in ("bce", "wgan_gp"):
    print(f"  {toy['arms'][arm]['display']:<9} покрыто по сидам: "
          f"{', '.join(str(int(v)) for v in toy['arms'][arm]['agg']['modes_covered']['per_seed'])}")
'''

CODE_TOY_FIGURE = '''show(toy8.plot_seed_samples(toy))
show(toy8.plot_seed_curves(toy))
'''

CODE_CONFIG = '''# --------------------------------------------------- конфиг повтора: меняйте здесь
EPOCHS = 80        # эпох на плечо: одинаковый бюджет у BCE и у WGAN-GP
BATCH = 128
LR = 2e-4          # Adam, betas (0.5, 0.999), одинаковый шаг у D и у G
N_CRITIC = 5       # шагов критика на шаг генератора — как в основном ноутбуке
LAMBDA_GP = 10.0   # вес штрафа за градиент (только WGAN-GP)
D_WIDTHS = (128, 256, 256)   # каналы критика: два stride-2 (28→14→7) + stride-1 блок 7×7
G_WIDTHS = (128, 64)         # ширины генератора DCGAN
Z_DIM = 100
N_EVAL = 2048      # латентный банк, общий для всех сидов
EVAL_SEED = 2024   # сид этого банка (не сид обучения!)
COMPUTE_FID = True
RESUME = True      # продолжать прерванный прогон из artifacts/notebook_seeds_<tag>
RUN_TAG = "run"    # каталог с чекпоинтами: artifacts/notebook_seeds_<tag>
N_CLUSTERS = 20    # кластеров в готовом k-means-референсе, переобучать не нужно

print(f"плечи: BCE и WGAN-GP, сиды {SEEDS}, по {EPOCHS} эпох каждое")
print(f"устройство: {G.device_report(DEVICE)}")
'''

CODE_NETS = '''import torch.nn as nn

CONV = G.conv2d_cls()          # FastConv2d на ROCm, nn.Conv2d на остальных устройствах


class Generator(nn.Module):
    """DCGAN-генератор из основного ноутбука: z(100) → 7×7 → 14×14 → 28×28, Tanh."""

    def __init__(self):
        super().__init__()
        w0, w1 = G_WIDTHS
        self.net = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, w0, 7, 1, 0), nn.BatchNorm2d(w0), nn.ReLU(True),
            nn.ConvTranspose2d(w0, w1, 4, 2, 1), nn.BatchNorm2d(w1), nn.ReLU(True),
            nn.ConvTranspose2d(w1, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    """Критик из основного ноутбука: два stride-2 блока (28→14→7) и stride-1 блок на 7×7."""

    def __init__(self):
        super().__init__()
        layers, width = [], 1
        for w in D_WIDTHS[:2]:
            layers += [CONV(width, w, 4, 2, 1), nn.LeakyReLU(0.2, True)]
            width = w
        for w in D_WIDTHS[2:]:
            layers += [CONV(width, w, 3, 1, 1), nn.LeakyReLU(0.2, True)]
            width = w
        layers += [nn.Flatten(), nn.Linear(width * 7 * 7, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def sample_z(n):
    return torch.randn(n, Z_DIM, 1, 1, device=DEVICE)


def gradient_penalty(critic, real, fake):
    """WGAN-GP: штраф на норму градиента критика вдоль отрезка real–fake."""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(interp)
    grads = torch.autograd.grad(score, interp, torch.ones_like(score), create_graph=True)[0]
    return LAMBDA_GP * ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


n_d, n_g = G.count_parameters(Critic()), G.count_parameters(Generator())
print(f"критик {n_d:,} параметров, генератор {n_g:,}, D/G = {n_d / n_g:.3f}".replace(",", " "))
print(f"свёртки критика: {CONV.__name__}")
'''

CODE_TRAIN = '''import json
import time

ART = ROOT / "artifacts"
RUN_DIR = ART / f"notebook_seeds_{RUN_TAG}"
REF_BCE80 = ART / "fashion" / "fashion_strongD_80ep" / "metrics" / "fashion_strongD_80ep_history.json"
REF_WGAN = ART / "fashion" / "fashion_strongD_wgan" / "metrics" / "fashion_strongD_wgan_history.json"
RUN_DIR.mkdir(parents=True, exist_ok=True)

CLF = G.ModeClassifier()
CLF.load_state_dict(torch.load(ART / "models" / "mode_classifier.pt", map_location="cpu",
                               weights_only=True)["state_dict"])
CLF.eval()
_blob = np.load(ART / "metrics" / "real_reference.npz")
REF = {"centers": _blob["centers"], "feature_mean": _blob["feature_mean"],
       "feature_std": _blob["feature_std"],
       "dist_threshold": float(_blob["dist_threshold"])}

X = G.load_train_tensor(G.Config(dataset="fashion")).to(DEVICE)
ITERS = X.size(0) // BATCH
# общий латентный банк: сиды отличаются моделью, а не точками, на которых их меряют
Z_EVAL = torch.randn(N_EVAL, Z_DIM, 1, 1, device=DEVICE,
                     generator=torch.Generator(device=DEVICE).manual_seed(EVAL_SEED))
print(f"данные {tuple(X.shape)} на {DEVICE}, {ITERS} итераций в эпохе")

if COMPUTE_FID and RUN_TAG == "run":
    INCEPTION = FIDLIB.make_inception(DEVICE)
    _real = np.load(FIDLIB.STATS_DIR / f"real_fashion_a_n{FIDLIB.N_SAMPLES}_2048d.npz")
    REAL_MU, REAL_SIGMA = _real["mu"], _real["sigma"].astype(np.float64)
    print("FID: InceptionV3 загружен, эталон тестовой части прочитан")
else:
    INCEPTION = None
    REAL_MU = REAL_SIGMA = None

HISTORY_KEYS = ("epoch", "max_class_share", "mode_coverage", "class_entropy",
                "intra_class_spread", "d_margin", "seconds", "loss_g", "loss_d",
                "class_share")


def run_path(arm, seed):
    return RUN_DIR / f"{arm}_seed{seed}"


def arm_done(arm, seed, epochs):
    """Совсем готовый прогон: история нужной длины и финальные сэмплы с FID."""
    metrics = run_path(arm, seed) / "metrics.json"
    final = run_path(arm, seed) / "final.pt"
    if not (metrics.exists() and final.exists()):
        return None
    history = json.loads(metrics.read_text())
    if len(history["epoch"]) < epochs or "fid" not in history:
        return None
    blob = torch.load(final, map_location="cpu", weights_only=False)
    total = float(history.get("total_seconds", history["seconds"][-1]))
    return {"history": history, "final64": blob["final64"], "fid": float(history["fid"]),
            "seconds": total, "gen": None}


def train_arm(arm, epochs, seed):
    """Одно плечо с нуля на заданном сиде. Каждую эпоху пишет метрики и чекпоинт."""
    done = arm_done(arm, seed, epochs) if RESUME else None
    if done is not None:
        print(f"[{arm} сид {seed}] уже обучено на {epochs} эпох — читаю с диска")
        return done
    with open(RUN_DIR / "progress.log", "a") as handle:
        handle.write(f"[{arm} сид {seed}] старт\\n")

    path = run_path(arm, seed)
    path.mkdir(parents=True, exist_ok=True)
    metrics_path = path / "metrics.json"
    state_path = path / "state.pt"

    G.set_seed(seed)
    gen, disc = Generator().to(DEVICE), Critic().to(DEVICE)
    opt_g = torch.optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    history = {key: [] for key in HISTORY_KEYS}
    start, started = 1, time.time()
    if RESUME and state_path.exists() and metrics_path.exists():
        blob = torch.load(state_path, map_location=DEVICE, weights_only=False)
        gen.load_state_dict(blob["gen"])
        disc.load_state_dict(blob["disc"])
        opt_g.load_state_dict(blob["opt_g"])
        opt_d.load_state_dict(blob["opt_d"])
        torch.set_rng_state(blob["rng_cpu"])
        torch.cuda.set_rng_state(blob["rng_cuda"].cpu())
        history = json.loads(metrics_path.read_text())
        start = int(blob["epoch"]) + 1
        print(f"[{arm} сид {seed}] продолжаю с эпохи {start}")

    for epoch in range(start, epochs + 1):
        t0 = time.time()
        gen.train()
        disc.train()
        log = {"loss_g": [], "loss_d": [], "d_margin": []}
        perm = torch.randperm(X.size(0), device=DEVICE)
        for step in range(ITERS):
            real = X[perm[step * BATCH:(step + 1) * BATCH]]
            z = sample_z(BATCH)
            if arm == "bce":                              # обычный GAN, non-saturating
                fake = gen(z).detach()
                out_real, out_fake = disc(real), disc(fake)
                loss_d = (bce(out_real, torch.ones_like(out_real))
                          + bce(out_fake, torch.zeros_like(out_fake)))
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                log["loss_d"].append(loss_d.item())
                log["d_margin"].append(out_real.mean().item() - out_fake.mean().item())
                if (step + 1) % N_CRITIC == 0:
                    out_g = disc(gen(sample_z(BATCH)))
                    loss_g = bce(out_g, torch.ones_like(out_g))
                    opt_g.zero_grad()
                    loss_g.backward()
                    opt_g.step()
                    log["loss_g"].append(loss_g.item())
            else:                                         # WGAN-GP: критик и штраф
                real = real.requires_grad_(False)
                with torch.no_grad():
                    fake = gen(z)
                d_real, d_fake = disc(real), disc(fake)
                gp = gradient_penalty(disc, real, fake)
                loss_d = d_fake.mean() - d_real.mean() + gp
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                log["loss_d"].append((d_fake.mean() - d_real.mean()).item())
                log["d_margin"].append(d_real.mean().item() - d_fake.mean().item())
                if (step + 1) % N_CRITIC == 0:
                    loss_g = -disc(gen(sample_z(BATCH))).mean()
                    opt_g.zero_grad()
                    loss_g.backward()
                    opt_g.step()
                    log["loss_g"].append(loss_g.item())

        modes = G.measure_modes(gen, Z_EVAL, CLF, REF, N_CLUSTERS)   # 2048 сэмплов
        seconds = time.time() - t0
        history["epoch"].append(epoch)
        history["max_class_share"].append(modes["max_class_share"])
        history["mode_coverage"].append(modes["mode_coverage"])
        history["class_entropy"].append(modes["class_entropy"])
        history["intra_class_spread"].append(modes["intra_class_spread"])
        history["d_margin"].append(float(np.mean(log["d_margin"])))
        history["seconds"].append(seconds)
        history["class_share"].append([float(share) for share in modes["class_share"]])
        history["loss_d"].append(float(np.mean(log["loss_d"])))
        history["loss_g"].append(float(np.mean(log["loss_g"])) if log["loss_g"]
                                 else float("nan"))
        history["total_seconds"] = time.time() - started
        metrics_path.write_text(json.dumps(history))
        if epoch % 5 == 0 or epoch == epochs:
            torch.save({"gen": gen.state_dict(), "disc": disc.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                        "epoch": epoch, "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state()}, state_path)
        line = (f"[{arm} сид {seed}] эпоха {epoch:02d}/{epochs}: доля топ "
                f"{modes['max_class_share']:.3f}, H {modes['class_entropy']:.3f}, покрытие "
                f"{modes['mode_coverage']:.2f}, {seconds:.0f} с")
        print(line, flush=True)
        # nbconvert отдаёт stdout только по завершении ячейки, поэтому прогресс дублируется
        # в файл: по нему видно, что четырёхчасовой прогон жив.
        with open(RUN_DIR / "progress.log", "a") as handle:
            handle.write(line + "\\n")

    gen.eval()
    with torch.no_grad():
        snap = G.generate_samples(gen, Z_EVAL[:64])
    final64 = ((snap.cpu() * 0.5 + 0.5).clamp(0, 1).mul(255).byte().squeeze(1).numpy())
    torch.save({"final64": final64}, path / "final.pt")

    if INCEPTION is not None:
        z = FIDLIB.latent(FIDLIB.N_SAMPLES, Z_DIM).to(DEVICE)
        with torch.no_grad():
            batches = [gen(z[i:i + FIDLIB.BATCH])
                       for i in range(0, len(z), FIDLIB.BATCH)]
            mu, sigma = FIDLIB.stats_of(
                FIDLIB.features_of_batches(batches, INCEPTION, DEVICE))
        history["fid"] = float(FIDLIB.frechet_distance(mu, sigma, REAL_MU, REAL_SIGMA))
    else:
        history["fid"] = float("nan")
    metrics_path.write_text(json.dumps(history))
    print(f"[{arm} сид {seed}] готово: {history['total_seconds']:.0f} с, "
          f"FID {history['fid']:.1f}", flush=True)
    with open(RUN_DIR / "progress.log", "a") as handle:
        handle.write(f"[{arm} сид {seed}] готово: {history['total_seconds']:.0f} с, "
                     f"FID {history['fid']:.1f}\\n")
    return {"history": history, "final64": final64, "fid": history["fid"],
            "seconds": history["total_seconds"], "gen": gen}


RUNS = {}
for _seed in SEEDS:
    for _arm in ("wgan_gp", "bce"):     # тяжёлое плечо первым
        RUNS[(_arm, _seed)] = train_arm(_arm, EPOCHS, _seed)

print("")
print(f"обучение завершено: {sum(run['seconds'] for run in RUNS.values()) / 60:.0f} мин "
      f"на {len(SEEDS)} сидов")
'''

CODE_METRICS = '''def final(arm, seed, key):
    return float(RUNS[(arm, seed)]["history"][key][-1])


def spread(values, fmt="{:.3f}"):
    values = list(values)
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 \\
        if len(values) > 1 else 0.0
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def fired(history):
    return [epoch for epoch, top, cov in zip(
        history["epoch"], history["max_class_share"], history["mode_coverage"])
        if top >= 0.35 or cov <= 0.75]


print("по каждому сиду отдельно (финальная эпоха):")
print(f"{'плечо':<8}{'сид':>5}{'доля топ':>10}{'покрытие':>10}{'H':>7}{'intra':>7}"
      f"{'margin':>8}{'FID':>7}{'эпох с коллапсом':>19}")
for arm in ("bce", "wgan_gp"):
    for seed in SEEDS:
        h = RUNS[(arm, seed)]["history"]
        print(f"{arm:<8}{seed:>5}{h['max_class_share'][-1]:>10.3f}"
              f"{h['mode_coverage'][-1]:>10.2f}{h['class_entropy'][-1]:>7.3f}"
              f"{h['intra_class_spread'][-1]:>7.2f}{h['d_margin'][-1]:>8.2f}"
              f"{h['fid']:>7.1f}{len(fired(h)):>9}/{len(h['epoch']):<5}")

print("")
print("среднее ± std по сидам (финальная эпоха):")
print(f"{'плечо':<8}{'доля топ':>18}{'покрытие':>16}{'H':>16}{'FID':>16}")
for arm in ("bce", "wgan_gp"):
    top = spread(final(arm, s, "max_class_share") for s in SEEDS)
    cov = spread((final(arm, s, "mode_coverage") for s in SEEDS), "{:.2f}")
    ent = spread((final(arm, s, "class_entropy") for s in SEEDS))
    fid = spread((RUNS[(arm, s)]["fid"] for s in SEEDS), "{:.1f}")
    print(f"{arm:<8}{top:>18}{cov:>16}{ent:>16}{fid:>16}")

print("")
print("критерий коллапса (доля топ ≥ 0.35 или покрытие ≤ 0.75):")
for arm in ("bce", "wgan_gp"):
    shares = [len(fired(RUNS[(arm, s)]["history"])) / len(RUNS[(arm, s)]["history"]["epoch"])
              for s in SEEDS]
    finals = [bool(fired(RUNS[(arm, s)]["history"])) and
              fired(RUNS[(arm, s)]["history"])[-1] == EPOCHS for s in SEEDS]
    print(f"  {arm:<8} на финальной эпохе — {sum(finals)} из {len(SEEDS)} сидов; "
          f"доля эпох с критерием {spread(shares, '{:.2f}')}")

# эталон основного ноутбука для контекста
ref_bce = json.loads(REF_BCE80.read_text()) if REF_BCE80.exists() else None
ref_wgan = json.loads(REF_WGAN.read_text()) if REF_WGAN.exists() else None
if ref_bce or ref_wgan:
    print("")
    print("для контекста — эталон основного ноутбука (MPS, один сид 42):")
    fid_ref = json.loads((ART / "metrics" / "fid_fashion_factorial.json").read_text())["arms"]
    if ref_bce:
        print(f"  BCE, 80 эпох: топ {ref_bce['max_class_share'][-1]:.3f}, "
              f"покрытие {ref_bce['mode_coverage'][-1]:.2f}, "
              f"FID {fid_ref['fashion_strongD_80ep']['fid']:.1f}")
    if ref_wgan:
        print(f"  WGAN-GP, 40 эпох: топ {ref_wgan['max_class_share'][-1]:.3f}, "
              f"покрытие {ref_wgan['mode_coverage'][-1]:.2f}, "
              f"FID {fid_ref['fashion_strongD_wgan']['fid']:.1f}")

# сводка для прозы и для artifacts/notebook_seeds_<tag>/run.json
summary = {
    "config": {"EPOCHS": EPOCHS, "SEEDS": list(SEEDS), "BATCH": BATCH, "LR": LR,
               "N_CRITIC": N_CRITIC, "LAMBDA_GP": LAMBDA_GP,
               "D_WIDTHS": list(D_WIDTHS), "G_WIDTHS": list(G_WIDTHS), "Z_DIM": Z_DIM,
               "N_EVAL": N_EVAL, "EVAL_SEED": EVAL_SEED, "device": DEVICE.type},
    "device": G.device_report(DEVICE),
    "torch": torch.__version__, "hip": torch.version.hip,
    "arms": {arm: {"seeds": {str(seed): {
        "history": RUNS[(arm, seed)]["history"],
        "fid": RUNS[(arm, seed)]["fid"],
        "seconds": RUNS[(arm, seed)]["seconds"]} for seed in SEEDS}}
        for arm in ("bce", "wgan_gp")},
}
(RUN_DIR / "run.json").write_text(json.dumps(summary, indent=1))
print("")
print(f"записано {(RUN_DIR / 'run.json').relative_to(ROOT)}")
'''

CODE_CURVES = '''COLORS = {"bce": "#c53030", "wgan_gp": "#2b6cb0"}

refs = {"bce": json.loads(REF_BCE80.read_text()) if REF_BCE80.exists() else None,
        "wgan_gp": json.loads(REF_WGAN.read_text()) if REF_WGAN.exists() else None}

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for arm in ("bce", "wgan_gp"):
    histories = [RUNS[(arm, seed)]["history"] for seed in SEEDS]
    epochs = histories[0]["epoch"]
    for key, ax, reference in (("max_class_share", axes[0], refs[arm]),
                               ("mode_coverage", axes[1], refs[arm])):
        values = np.array([h[key] for h in histories], dtype=float)
        ax.plot(epochs, values.mean(0), color=COLORS[arm], linewidth=1.8,
                label=f"{arm}: среднее по {len(SEEDS)} сидам")
        ax.fill_between(epochs, values.min(0), values.max(0), color=COLORS[arm],
                        alpha=0.18, linewidth=0)
        for history in histories:
            ax.plot(history["epoch"], history[key], color=COLORS[arm],
                    linewidth=0.7, alpha=0.45)
        if reference is not None:
            ax.plot(reference["epoch"], reference[key], color=COLORS[arm],
                    linestyle=":", linewidth=1.0, alpha=0.9,
                    label=f"{arm}: эталон MPS, сид 42")
axes[0].set_ylabel("доля топ-класса")
axes[1].set_ylabel("покрытие мод")
axes[0].axhline(0.35, color="#333333", linestyle="--", linewidth=1, label="порог 0.35")
axes[1].axhline(0.75, color="#333333", linestyle="--", linewidth=1, label="порог 0.75")
for ax in axes:
    ax.set_xlabel("эпоха")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
fig.suptitle("Fashion-MNIST: три сида, среднее и min–max", fontsize=10)
show(fig)
'''

CODE_SAMPLES = '''def tile(images):
    """64 сэмпла (8x8) -> одна картинка 224x224."""
    return images.reshape(8, 8, 28, 28).transpose(0, 2, 1, 3).reshape(8 * 28, 8 * 28)


fig, axes = plt.subplots(2, len(SEEDS), figsize=(3.2 * len(SEEDS), 6.6))
for row, arm in enumerate(("bce", "wgan_gp")):
    for col, seed in enumerate(SEEDS):
        ax = axes[row][col]
        h = RUNS[(arm, seed)]["history"]
        ax.imshow(tile(RUNS[(arm, seed)]["final64"]), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{arm}, сид {seed}: топ {h['max_class_share'][-1]:.3f}, "
                     f"покрытие {h['mode_coverage'][-1]:.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
fig.tight_layout()
show(fig)
'''

CODE_ARTIFACTS = '''for path in sorted(RUN_DIR.iterdir()):
    print(path.relative_to(ROOT))
print("")
print("каждый прогон лежит в своей папке: metrics.json (история по эпохам),")
print("state.pt (последний чекпоинт), final.pt (сэмплы финальной эпохи).")
'''


# ------------------------------------------------------------------ assembly
def markdown_cells(stats: dict | None) -> dict[str, str]:
    return {
        "md-intro": md_intro(stats),
        "md-toy": md_toy_section(stats),
        "md-fashion": md_fashion(stats),
        "md-results": md_results(stats),
        "md-limits": md_limits(stats),
    }


def build(epochs: int = 80, tag: str = "run", out: Path = OUT,
          toy_steps: int = 5000) -> Path:
    stats = load_stats()
    md = markdown_cells(stats)
    config = CODE_CONFIG.replace("EPOCHS = 80", f"EPOCHS = {epochs}") \
                        .replace('RUN_TAG = "run"', f'RUN_TAG = "{tag}"')
    head = CODE_HEAD.replace("TOY_STEPS = 5000", f"TOY_STEPS = {toy_steps}")
    cells = [
        nbf.v4.new_markdown_cell(md["md-intro"], id="md-intro"),
        nbf.v4.new_code_cell(head, id="code-head"),
        nbf.v4.new_markdown_cell(md["md-toy"], id="md-toy"),
        nbf.v4.new_code_cell(CODE_TOY_TRAIN, id="code-toy-train"),
        nbf.v4.new_code_cell(CODE_TOY_TABLE, id="code-toy-table"),
        nbf.v4.new_code_cell(CODE_TOY_FIGURE, id="code-toy-figure"),
        nbf.v4.new_markdown_cell(md["md-fashion"], id="md-fashion"),
        nbf.v4.new_code_cell(config, id="code-config"),
        nbf.v4.new_code_cell(CODE_NETS, id="code-nets"),
        nbf.v4.new_code_cell(CODE_TRAIN, id="code-train"),
        nbf.v4.new_code_cell(CODE_METRICS, id="code-metrics"),
        nbf.v4.new_code_cell(CODE_CURVES, id="code-curves"),
        nbf.v4.new_code_cell(CODE_SAMPLES, id="code-samples"),
        nbf.v4.new_markdown_cell(md["md-results"], id="md-results"),
        nbf.v4.new_markdown_cell(md["md-limits"], id="md-limits"),
        nbf.v4.new_code_cell(CODE_ARTIFACTS, id="code-artifacts"),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                 "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.13"}
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    print(f"written {out} (epochs={epochs}, tag={tag}, stats={'yes' if stats else 'no'})")
    return out


def finalize(path: Path = OUT) -> None:
    """Bake prose numbers into an already executed notebook; code outputs stay as they are."""
    stats = load_stats()
    if stats is None:
        raise SystemExit("run.json is missing: execute the notebook first")
    nb = nbf.read(str(path), as_version=4)
    md = markdown_cells(stats)
    changed = 0
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.get("id") in md:
            cell.source = md[cell.get("id")]
            changed += 1
    nbf.write(nb, str(path))
    print(f"finalized {changed} markdown cells in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--toy-steps", type=int, default=5000)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.finalize:
        finalize(args.out)
    else:
        build(args.epochs, args.tag, args.out, args.toy_steps)


if __name__ == "__main__":
    main()
