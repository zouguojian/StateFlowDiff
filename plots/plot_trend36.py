from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "trend_fidelity_h36.csv"
OUTPUT_DIR = ROOT / "figures"

MODEL_ORDER = [
    "StateFlowDiff",
    "iTransformer",
    "SimDiff",
    "Diffusion-TS",
    "TSDiff",
    "DLinear",
    "PatchTST",
]
COLORS = {
    "StateFlowDiff": "#C83E4D",
    "iTransformer": "#4C78A8",
    "SimDiff": "#5B9A6F",
    "Diffusion-TS": "#6D6D6D",
    "TSDiff": "#2A9D8F",
    "DLinear": "#E29A45",
    "PatchTST": "#8A6FA8",
}
MARKERS = {
    "StateFlowDiff": "X",
    "iTransformer": "o",
    "SimDiff": "D",
    "Diffusion-TS": "P",
    "TSDiff": "v",
    "DLinear": "^",
    "PatchTST": "s",
}
LINESTYLES = {
    "StateFlowDiff": "-",
    "iTransformer": "--",
    "SimDiff": "-.",
    "Diffusion-TS": (0, (3, 1.4)),
    "TSDiff": (0, (5, 1.5, 1.2, 1.5)),
    "DLinear": ":",
    "PatchTST": (0, (4, 1.5)),
}


def style_axis(ax, ylabel, title, horizons):
    ax.set_facecolor("white")
    ax.grid(True, color="#D9DEE5", linewidth=0.8, linestyle=":", alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=14.5, width=1.15, length=4.6)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.set_xticks(horizons)
    ax.set_xlabel(r"Trend horizon $m$", fontsize=15.5, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=15.5, fontweight="bold")
    ax.set_title(title, fontsize=16.5, fontweight="bold", pad=6)


def main():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    df = pd.read_csv(CSV_PATH)
    horizons = sorted(df["m"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.35), dpi=300)
    for ax, metric, ylabel, title in (
        (axes[0], "TA", "Trend Alignment", "(a) Trend alignment"),
        (axes[1], "SSC", "Step-Sign Consistency", "(b) Step-sign consistency"),
    ):
        for model in MODEL_ORDER:
            values = df[df["model"] == model].sort_values("m")
            ax.plot(
                values["m"],
                values[metric],
                label=model,
                color=COLORS[model],
                marker=MARKERS[model],
                linestyle=LINESTYLES[model],
                linewidth=2.5 if model == "StateFlowDiff" else 2.0,
                markersize=7.6 if model == "StateFlowDiff" else 6.6,
                markerfacecolor=COLORS[model],
                markeredgecolor="white",
                markeredgewidth=0.75,
            )
        style_axis(ax, ylabel, title, horizons)
        ax.margins(x=0.015, y=0.08)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.012),
        ncol=7,
        frameon=False,
        fontsize=16.0,
        handlelength=1.75,
        columnspacing=0.32,
        handletextpad=0.24,
    )
    fig.subplots_adjust(left=0.072, right=0.993, bottom=0.225, top=0.93, wspace=0.155)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "trend_h36_preview.png", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT_DIR / "trend_h36_preview.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
