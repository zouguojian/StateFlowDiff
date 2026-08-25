from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


COLORS = {
    "history": "#CDBCAC",
    "truth": "#A7806B",
    "itr": "#506F83",
    "sim": "#AE804A",
    "ours": "#B25647",
    "future": "#F4F1EE",
    "grid": "#E5E7EB",
}

TITLES = [
    "(a) Post-peak level transition",
    "(b) Recovery after transient spike",
    "(c) Rising-trend restoration after local dip",
    "(d) Rebound after an early-horizon trough",
]


def register_times_new_roman() -> None:
    font_dir = Path(__file__).resolve().parents[1] / "assets" / "fonts"
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        path = font_dir / name
        if path.exists():
            font_manager.fontManager.addfont(path)


def cache_case(cache_dir: Path, sample_index: int, sensor_index: int) -> dict:
    ours = np.load(cache_dir / "ours.npz", allow_pickle=True)
    itr = np.load(cache_dir / "itr.npz", allow_pickle=True)
    sim = np.load(cache_dir / "sim.npz", allow_pickle=True)
    return {
        "history": ours["hist"][sample_index, :, sensor_index].tolist(),
        "ground_truth": ours["trues"][sample_index, :, sensor_index].tolist(),
        "predictions": {
            "iTransformer": itr["preds"][sample_index, :, sensor_index].tolist(),
            "SimDiff": sim["preds"][sample_index, :, sensor_index].tolist(),
            "ResDiff": ours["preds"][sample_index, :, sensor_index].tolist(),
        },
    }


def draw_case(ax, case: dict, title: str) -> list:
    history = np.asarray(case["history"], dtype=float)
    truth = np.asarray(case["ground_truth"], dtype=float)
    itr = np.asarray(case["predictions"]["iTransformer"], dtype=float)
    sim = np.asarray(case["predictions"]["SimDiff"], dtype=float)
    ours = np.asarray(case["predictions"]["ResDiff"], dtype=float)

    # Compress the long history window so the forecast region remains wide
    # enough to compare trajectory shapes clearly.
    history_span = 38.0
    forecast_scale = 2.0 / 3.0
    xh = np.linspace(-history_span, 0.0, len(history))
    xf = np.arange(0, len(truth) + 1, dtype=float) * forecast_scale
    anchor = history[-1]

    ax.axvspan(0, len(truth) * forecast_scale, color=COLORS["future"], zorder=0)
    h0, = ax.plot(xh, history, color=COLORS["history"], lw=0.8, alpha=0.9, label="History", zorder=1)
    ax.axvline(0, color="#777777", ls=(0, (3, 2)), lw=0.75, zorder=2)
    h1, = ax.plot(xf, np.r_[anchor, truth], color=COLORS["truth"], lw=1.25, label="Ground truth", zorder=4)
    h2, = ax.plot(xf, np.r_[anchor, itr], color=COLORS["itr"], ls=(0, (4, 2)), lw=1.15, alpha=0.95, label="iTransformer", zorder=3)
    h3, = ax.plot(xf, np.r_[anchor, sim], color=COLORS["sim"], ls=(0, (2.5, 1.1)), lw=1.15, alpha=0.95, label="SimDiff", zorder=3)
    h4, = ax.plot(xf, np.r_[anchor, ours], color=COLORS["ours"], lw=1.25, label="StateFlowDiff", zorder=5)

    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="center", pad=4)
    ax.set_ylabel("Traffic flow", fontsize=12.0)
    history_tick_labels = [-96, -72, -48, -24, 0]
    history_tick_positions = np.linspace(-history_span, 0.0, len(history_tick_labels))
    future_ticks = [6, 12, 18, 24, 30, 36]
    future_tick_positions = [t * forecast_scale for t in future_ticks]
    ax.set_xlim(-history_span - 1.0, len(truth) * forecast_scale + 0.5)
    ax.set_xticks(list(history_tick_positions) + future_tick_positions)
    ax.set_xticklabels([str(v) for v in history_tick_labels] + [str(v) for v in future_ticks])
    ax.grid(True, axis="y", color=COLORS["grid"], lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#55595D")
    ax.spines["bottom"].set_color("#55595D")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(labelsize=10.5, length=3.2, width=0.8, colors="#333333")
    return [h0, h1, h2, h3, h4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_json", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidates = json.loads(Path(args.case_json).read_text(encoding="utf-8"))
    cases = [candidates[i - 1] for i in (1, 6, 15, 18)]

    register_times_new_roman()
    plt.rcParams.update({"font.family": "Times New Roman", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(4, 1, figsize=(7.8, 8.5), facecolor="white")
    handles = None
    for idx, (ax, case, title) in enumerate(zip(axes, cases, TITLES)):
        panel_handles = draw_case(ax, case, title)
        handles = handles or panel_handles
        if idx < len(axes) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Time step (forecast horizon)", fontsize=12.0, labelpad=4)

    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.032),
        ncol=5,
        frameon=False,
        fontsize=11.7,
        handlelength=2.4,
        columnspacing=1.35,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.985, bottom=0.145, hspace=0.25)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
