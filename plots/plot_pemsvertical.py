#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_model_name(name: str) -> str:
    name = str(name)
    lower = name.lower()
    if lower in {"itransformer", "itrans", "i_transformer"}:
        return "iTransformer"
    if lower in {"stateflowdiff", "regdiff", "reg_diff", "resdiff", "method"}:
        return "StateFlowDiff"
    return name


def parse_source(dataset_name: str) -> str:
    name = str(dataset_name)
    if "PEMS03" in name:
        return "PEMS03"
    if "PEMS04" in name:
        return "PEMS04"
    if "PEMS08" in name:
        return "PEMS08"
    raise ValueError(f"Cannot parse source from dataset name: {dataset_name}")


def prepare_boundary_stats(csv_path: str, metric: str = "mae") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ratio" not in df.columns and "history_ratio" in df.columns:
        df = df.rename(columns={"history_ratio": "ratio"})
    required = {"dataset", "model", "ratio", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    df["source"] = df["dataset"].apply(parse_source)
    df["model_plot"] = df["model"].apply(normalize_model_name)
    df["ratio"] = df["ratio"].astype(float)
    df = df[df["model_plot"].isin(["iTransformer", "StateFlowDiff"])].copy()

    stats = (
        df.groupby(["source", "model_plot", "ratio"], as_index=False)[metric]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    return stats


def plot_pems_boundary_vertical(
    csv_path: str,
    output_path: str,
    metric: str = "mae",
    title: str = "Applicability Boundary under Different Continuous-History Ratios",
) -> None:
    stats = prepare_boundary_stats(csv_path, metric=metric)

    sources = ["PEMS03", "PEMS04", "PEMS08"]
    panel_titles = {
        "PEMS03": "PeMS03",
        "PEMS04": "PeMS04",
        "PEMS08": "PeMS08",
    }
    model_order = ["iTransformer", "StateFlowDiff"]
    colors = {
        "iTransformer": "#506F83",
        "StateFlowDiff": "#B25647",
    }
    fill_colors = {
        "iTransformer": "#C7D0D6",
        "StateFlowDiff": "#D9B8B1",
    }

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 19.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 14.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(11.2, 12.0), sharex=False)

    for idx, (ax, source) in enumerate(zip(axes, sources)):
        source_stats = stats[stats["source"] == source].copy()

        for model in model_order:
            s = source_stats[source_stats["model_plot"] == model].sort_values("ratio")
            if s.empty:
                continue

            x = s["ratio"].to_numpy(dtype=float)
            y = s["mean"].to_numpy(dtype=float)
            ymin = s["min"].to_numpy(dtype=float)
            ymax = s["max"].to_numpy(dtype=float)

            ax.fill_between(
                x,
                ymin,
                ymax,
                color=fill_colors[model],
                alpha=0.22 if model == "StateFlowDiff" else 0.20,
                linewidth=0,
                zorder=1,
            )

            ax.plot(
                x,
                y,
                color=colors[model],
                marker="s" if model == "StateFlowDiff" else "o",
                markersize=6.0,
                markeredgewidth=0.45,
                markeredgecolor="white",
                linewidth=1.45,
                label=model,
                zorder=3,
            )

        ax.set_title(
            f"({chr(ord('a') + idx)}) {panel_titles[source]}",
            fontsize=17,
            fontweight="bold",
            pad=7,
            loc="center",
        )
        if idx == len(axes) - 1:
            ax.set_xlabel(r"Continuous-history ratio $\xi$", labelpad=5)
        else:
            ax.set_xlabel("")
        ax.set_ylabel(metric.upper() if metric.lower() != "mae" else "MAE", labelpad=7)

        ratios = sorted(source_stats["ratio"].unique())
        ax.set_xticks(ratios)
        ax.set_xticklabels([f"{r:.2f}" for r in ratios])
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(17)
            tick.set_fontweight("bold")
        ax.tick_params(axis="both", width=1.15, length=4.0)

        ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.75, color="#BFC5CC")
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#55595D")

        leg = ax.legend(
            loc="upper right",
            frameon=False,
            fancybox=False,
            framealpha=0.92,
            borderpad=0.38,
            handlelength=1.8,
        )
        for text in leg.get_texts():
            text.set_fontweight("normal")

        y_all = []
        for model in model_order:
            s = source_stats[source_stats["model_plot"] == model].sort_values("ratio")
            if not s.empty:
                y_all.extend(s["min"].tolist())
                y_all.extend(s["max"].tolist())
        if y_all:
            y_min = float(np.nanmin(y_all))
            y_max = float(np.nanmax(y_all))
            pad = max((y_max - y_min) * 0.14, 1e-4)
            ax.set_ylim(y_min - pad, y_max + pad)

    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.965)
    fig.tight_layout(rect=[0.045, 0.03, 0.99, 0.99], h_pad=2.0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    print(f"Saved figure to: {output}")

    if output.suffix.lower() != ".pdf":
        pdf_path = output.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved vector figure to: {pdf_path}")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument(
        "--out",
        type=str,
        default="figures/pems_vertical.png",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mae",
        choices=["mae", "mse", "rmse", "mape"],
    )
    args = parser.parse_args()

    plot_pems_boundary_vertical(
        csv_path=args.csv,
        output_path=args.out,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
