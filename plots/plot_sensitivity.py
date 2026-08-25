#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_k_sensitivity(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)

    if "MAE_mean" in df.columns:
        x = df["K"]
        mae = df["MAE_mean"]
        mae_err = df["MAE_std"] if "MAE_std" in df.columns else None
        if "time_mean" in df.columns:
            time_val = df["time_mean"]
        else:
            time_val = df["inference_time_seconds_mean"]
        time_err = df["time_std"] if "time_std" in df.columns else None
    else:
        x = df["K"]
        mae = df["MAE"]
        mae_err = None
        time_val = df["inference_time_seconds"]
        time_err = None

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12.0,
            "axes.labelsize": 13.7,
            "axes.labelweight": "normal",
            "xtick.labelsize": 11.7,
            "ytick.labelsize": 11.7,
            "legend.fontsize": 11.5,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax1 = plt.subplots(figsize=(8.2, 3.55))

    color_mae = "#506F83"
    color_time = "#B25647"

    if mae_err is not None:
        ax1.errorbar(
            x,
            mae,
            yerr=mae_err,
            color=color_mae,
            marker="o",
            linewidth=1.35,
            markersize=4.8,
            capsize=2.2,
            capthick=0.8,
            label="MAE",
        )
    else:
        ax1.plot(
            x,
            mae,
            color=color_mae,
            marker="o",
            linewidth=1.35,
            markersize=4.8,
            label="MAE",
        )

    ax1.set_xlabel("Number of sampled realizations $K$")
    ax1.set_ylabel("MAE", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, axis="y", linestyle=":", linewidth=0.55, color="#BFC5CC", alpha=0.75)

    ax2 = ax1.twinx()

    if time_err is not None:
        ax2.errorbar(
            x,
            time_val,
            yerr=time_err,
            color=color_time,
            marker="s",
            linestyle="-",
            linewidth=1.25,
            markersize=4.8,
            capsize=2.2,
            capthick=0.8,
            label="Inference time",
        )
    else:
        ax2.plot(
            x,
            time_val,
            color=color_time,
            marker="s",
            linestyle="-",
            linewidth=1.25,
            markersize=4.8,
            label="Inference time",
        )

    ax2.set_ylabel("Inference time (s)", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax1.set_xticks(list(x))
    for tick in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
        tick.set_fontweight("normal")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.8,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.28, top=0.97)

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, dpi=400, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved figure to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument(
        "--out",
        type=str,
        default="figures/k_sensitivity.png",
    )
    args = parser.parse_args()

    plot_k_sensitivity(args.csv, args.out)


if __name__ == "__main__":
    main()
