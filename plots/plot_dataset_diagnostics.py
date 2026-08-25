#!/usr/bin/env python3
"""
Compute dataset-level traffic-state diagnostics and draw a 2x2 paper figure.

Updated figure design:
  (A) Daily high-state ratio distribution q_d
  (B) Average daily flow trend (time-of-day aggregated profile)
  (C) Conditional future ambiguity CFA
  (D) Node-level heterogeneity NH

Expected default data layout under --data-root:
  fujian-30/fujian30_clean.csv
  pems03/clean.csv
  pems04/clean.csv
  pems08/clean.csv

Each CSV should be in long format with columns:
  time_slot, station_index, traffic_flow
Optional columns such as is_holiday are ignored for these diagnostics.

Outputs:
  dataset_diagnostics.png
  dataset_diagnostics.pdf
  dataset_diagnostics_table.csv
  dataset_diagnostics_table.tex
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class DiagnosticResult:
    name: str
    label: str
    T: int
    N: int
    days_used: int
    slots_per_day: int
    DSD: float
    CFA: float
    NH: float
    q_d: np.ndarray
    node_het_daily: np.ndarray
    avg_profile_hours: np.ndarray
    avg_profile: np.ndarray
    n_cfa_groups: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and plot traffic dataset diagnostics.")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Root directory containing dataset folders.")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"), help="Output directory.")
    parser.add_argument("--gamma", type=float, default=0.85, help="High-state quantile threshold computed on the training segment.")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Prefix ratio used to estimate high-state thresholds.")
    parser.add_argument("--L", type=int, default=96, help="History window length used for CFA contexts.")
    parser.add_argument("--H", type=int, default=12, help="Prediction horizon used for CFA future high-state ratio.")
    parser.add_argument("--context-bins", type=int, default=4, help="Quantile bins per numerical context feature for CFA grouping.")
    parser.add_argument("--tod-bins", type=int, default=6, help="Number of time-of-day bins for CFA grouping.")
    parser.add_argument("--min-group-size", type=int, default=8, help="Minimum samples required for a CFA condition group.")
    parser.add_argument("--sample-stride", type=int, default=10, help="Stride for CFA samples; use >1 for faster exploratory runs.")
    parser.add_argument("--min-day-coverage", type=float, default=0.8, help="Minimum fraction of modal daily slots required to keep a day.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=("Optional dataset override in the form Name=/path/to/file.csv. "
              "Can be supplied multiple times. If omitted, default paths are scanned."),
    )
    parser.add_argument("--fig-width", type=float, default=3.5, help="Figure width in inches for single-column layout.")
    parser.add_argument("--fig-height", type=float, default=3.25, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG export DPI.")
    return parser.parse_args()


def default_dataset_paths(root: Path) -> Dict[str, Path]:
    return {
        "Fujian-30": root / "fujian-30" / "fujian30_clean.csv",
        "PeMS03": root / "pems03" / "clean.csv",
        "PeMS04": root / "pems04" / "clean.csv",
        "PeMS08": root / "pems08" / "clean.csv",
    }


def parse_dataset_overrides(items: List[str] | None) -> Dict[str, Path] | None:
    if not items:
        return None
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset value: {item}. Use Name=/path/to/file.csv")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path).expanduser().resolve()
    return out


def short_label(name: str) -> str:
    return name


def load_long_csv(path: Path) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    print(f"       Loading with chunked optimized reader...")
    
    chunks = []
    chunksize = 2000000
    chunk_num = 0
    
    for chunk in pd.read_csv(path, 
                             usecols=["time_slot", "station_index", "traffic_flow"],
                             dtype={"station_index": "int32", "traffic_flow": "float32"},
                             chunksize=chunksize,
                             engine='c'):
        chunk = chunk.drop_duplicates(subset=["time_slot", "station_index"], keep="first")
        chunks.append(chunk)
        chunk_num += 1
        if chunk_num % 3 == 0:
            print(f"       ... processing chunk {chunk_num} (~{chunk_num * chunksize // 1000000}M rows)")
    
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"       ... loaded {len(df)} unique records")
    
    df["time_slot"] = pd.to_datetime(df["time_slot"], cache=True)
    
    unique_stations = sorted(df["station_index"].unique())
    n_stations = len(unique_stations)
    station_map = {old_id: new_id for new_id, old_id in enumerate(unique_stations)}
    df["station_index"] = df["station_index"].map(station_map).astype("int16")
    
    print(f"       ... pivoting to matrix ({df['time_slot'].nunique()} x {n_stations})...")
    mat = df.pivot(index="time_slot", columns="station_index", values="traffic_flow")
    del df
    
    mat = mat.sort_index().sort_index(axis=1)
    print(f"       Final shape: {mat.shape[0]} x {mat.shape[1]}")
    
    mat = mat.interpolate(axis=0, limit_direction="both", limit=5)
    
    return pd.DatetimeIndex(mat.index), mat.to_numpy(dtype=np.float32)


def safe_quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.array([], dtype=int)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    bins = np.floor(ranks / max(len(values), 1) * n_bins).astype(int)
    return np.clip(bins, 0, n_bins - 1)


def rolling_mean_1d(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    cumsum = np.concatenate([[0.0], np.cumsum(values)])
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def tod_to_hour(tod: str) -> float:
    parts = tod.split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    s = int(parts[2]) if len(parts) > 2 else 0
    return h + m / 60.0 + s / 3600.0


def compute_average_daily_profile(
    times: pd.DatetimeIndex,
    X: np.ndarray,
    valid_dates: set,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean-normalized average daily profile and its time-of-day grid in hours."""
    time_df = pd.DataFrame({"date": times.date, "tod": times.strftime("%H:%M:%S")})
    node_mean = np.nanmean(X, axis=1)
    prof_df = pd.DataFrame({"date": time_df["date"].values, "tod": time_df["tod"].values, "mean_flow": node_mean})
    prof = prof_df.pivot_table(index="date", columns="tod", values="mean_flow", aggfunc="mean")
    prof = prof.reindex(sorted(prof.columns), axis=1)
    prof = prof.loc[[d for d in prof.index if d in valid_dates]]
    complete = prof.dropna(axis=0)
    if len(complete) >= max(3, len(prof) // 3):
        prof_use = complete
    else:
        prof_use = prof.interpolate(axis=1, limit_direction="both").dropna(axis=0)
    if len(prof_use) == 0:
        return np.array([]), np.array([])
    daily_mean = prof_use.mean(axis=1).replace(0, np.nan)
    normalized = prof_use.div(daily_mean, axis=0).replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    if len(normalized) == 0:
        return np.array([]), np.array([])
    avg_profile = normalized.mean(axis=0).to_numpy(dtype=float)
    hours = np.asarray([tod_to_hour(c) for c in normalized.columns], dtype=float)
    return hours, avg_profile


def compute_diagnostics(
    name: str,
    times: pd.DatetimeIndex,
    X: np.ndarray,
    gamma: float = 0.85,
    train_ratio: float = 0.6,
    L: int = 96,
    H: int = 12,
    context_bins: int = 4,
    tod_bins: int = 6,
    min_group_size: int = 8,
    sample_stride: int = 1,
    min_day_coverage: float = 0.8,
) -> DiagnosticResult:
    T, N = X.shape
    train_end = max(int(T * train_ratio), 1)
    theta = np.nanquantile(X[:train_end], gamma, axis=0)
    high = X > theta[None, :]

    time_df = pd.DataFrame({"date": times.date, "tod": times.strftime("%H:%M:%S")})
    slots_per_day = time_df.groupby("date")["tod"].nunique()
    modal_slots = int(slots_per_day.mode().iloc[0])
    min_slots = int(np.ceil(modal_slots * min_day_coverage))
    valid_dates = set(slots_per_day[slots_per_day >= min_slots].index)

    # Daily high-state ratio q_d and daily node heterogeneity h_d = std_n(q_{d,n}).
    q_d: List[float] = []
    node_het_daily: List[float] = []
    for date, idxs in time_df.groupby("date").indices.items():
        if date not in valid_dates:
            continue
        idx = np.asarray(idxs, dtype=int)
        high_day = high[idx]  # slots x nodes
        q_d.append(float(high_day.mean()))
        q_dn = high_day.mean(axis=0)
        node_het_daily.append(float(np.std(q_dn)))
    q_d_arr = np.asarray(q_d, dtype=float)
    node_het_arr = np.asarray(node_het_daily, dtype=float)
    DSD = float(np.std(q_d_arr)) if len(q_d_arr) else np.nan
    NH = float(np.mean(node_het_arr)) if len(node_het_arr) else np.nan

    # Average daily profile for panel (B); used as visual evidence rather than a scalar metric.
    avg_hours, avg_profile = compute_average_daily_profile(times, X, valid_dates)

    # Conditional future ambiguity.
    node_mean = np.nanmean(X, axis=1)
    high_ratio_time = high.mean(axis=1)
    if T >= L + H + 1:
        hist_mean_all = rolling_mean_1d(node_mean, L)
        hist_high_all = rolling_mean_1d(high_ratio_time, L)
        t_valid = np.arange(L - 1, T - H, sample_stride, dtype=int)
        hist_idx = t_valid - (L - 1)
        hist_mean = hist_mean_all[hist_idx]
        hist_high = hist_high_all[hist_idx]
        hist_trend = (node_mean[t_valid] - node_mean[t_valid - L + 1]) / max(L - 1, 1)
        future_high_all = rolling_mean_1d(high_ratio_time[1:], H)
        future_high = future_high_all[t_valid]

        minute_of_day = times[t_valid].hour * 60 + times[t_valid].minute
        tod_group = np.floor(minute_of_day / 1440.0 * tod_bins).astype(int)
        tod_group = np.clip(tod_group, 0, tod_bins - 1)

        b0 = safe_quantile_bins(hist_mean, context_bins)
        b1 = safe_quantile_bins(hist_trend, context_bins)
        b2 = safe_quantile_bins(hist_high, context_bins)
        group_id = b0.astype(np.int64)
        group_id = group_id * context_bins + b1
        group_id = group_id * context_bins + b2
        group_id = group_id * tod_bins + tod_group

        grouped = pd.DataFrame({"group": group_id, "future_high": future_high}).groupby("group")["future_high"]
        sizes = grouped.size()
        variances = grouped.var(ddof=0)
        keep = sizes >= min_group_size
        if keep.sum() > 0:
            CFA = float((sizes[keep] * variances[keep]).sum() / sizes[keep].sum())
            n_cfa_groups = int(keep.sum())
        else:
            CFA = np.nan
            n_cfa_groups = 0
    else:
        CFA = np.nan
        n_cfa_groups = 0

    return DiagnosticResult(
        name=name,
        label=short_label(name),
        T=T,
        N=N,
        days_used=len(q_d_arr),
        slots_per_day=modal_slots,
        DSD=DSD,
        CFA=CFA,
        NH=NH,
        q_d=q_d_arr,
        node_het_daily=node_het_arr,
        avg_profile_hours=avg_hours,
        avg_profile=avg_profile,
        n_cfa_groups=n_cfa_groups,
    )


def save_tables(results: List[DiagnosticResult], out_dir: Path) -> None:
    rows = []
    for r in results:
        rows.append({
            "Dataset": r.name,
            "Nodes": r.N,
            "Time Steps": r.T,
            "Days Used": r.days_used,
            "Slots/Day": r.slots_per_day,
            "DSD": r.DSD,
            "CFA": r.CFA,
            "NH": r.NH,
            "CFA Groups": r.n_cfa_groups,
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "dataset_diagnostics_table.csv", index=False)

    compact = table[["Dataset", "DSD", "CFA", "NH"]].copy()
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & DSD & CFA & NH \\",
        r"\midrule",
    ]
    for _, row in compact.iterrows():
        vals = [
            row["Dataset"],
            "--" if pd.isna(row["DSD"]) else f"{row['DSD']:.4f}",
            "--" if pd.isna(row["CFA"]) else f"{row['CFA']:.4f}",
            "--" if pd.isna(row["NH"]) else f"{row['NH']:.4f}",
        ]
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "dataset_diagnostics_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_diagnostics(results: List[DiagnosticResult], out_dir: Path, fig_width: float, fig_height: float, dpi: int) -> None:
    PALETTE = {"Fujian-30": "#B25647", "PeMS03": "#506F83", "PeMS04": "#AE804A", "PeMS08": "#FABD29"}
    EDGE = "#2F2F2F"
    GRID = "#D9D9D9"

    labels = [r.label for r in results]
    colors = [PALETTE.get(lbl, "#888888") for lbl in labels]
    x = np.arange(len(results))

    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 11.7,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11.7,
        "xtick.labelsize": 10.4,
        "ytick.labelsize": 10.4,
        "axes.linewidth": 0.9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "grid.color": GRID,
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.15), dpi=dpi)
    plt.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.10, wspace=0.24, hspace=0.34)

    def _style_boxplot(bp: dict, colors: list) -> None:
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.52)
            patch.set_edgecolor(EDGE)
            patch.set_linewidth(1.0)
        for part in ["whiskers", "caps"]:
            for item in bp[part]:
                item.set_color(EDGE)
                item.set_linewidth(0.9)
        for item in bp["medians"]:
            item.set_color(EDGE)
            item.set_linewidth(1.4)

    # (A) Average daily flow trend.
    ax = axes[0, 0]
    for r, color in zip(results, colors):
        if len(r.avg_profile_hours) == 0:
            continue
        zo = 4 if r.label == "Fujian-30" else 2
        ax.plot(
            r.avg_profile_hours,
            r.avg_profile,
            linewidth=1.75 if r.label == "Fujian-30" else 1.05,
            color=color,
            alpha=0.96 if r.label == "Fujian-30" else 0.90,
            marker=None,
            label=r.label,
            zorder=zo,
        )
    ax.set_title("(a) Average daily flow trend", pad=6)
    ax.set_ylabel("Normalized flow")
    ax.set_xlabel("Hour of day")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"])
    ax.legend(ncol=2, handlelength=1.8, columnspacing=0.9, fontsize=9.7, loc="lower right")

    # (B) Daily high-state ratio distribution.
    ax = axes[0, 1]
    q_values = [r.q_d for r in results]
    vp = ax.violinplot(q_values, positions=np.arange(1, len(results) + 1), widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
    bp = ax.boxplot(
        q_values,
        positions=np.arange(1, len(results) + 1),
        showfliers=False,
        patch_artist=True,
        widths=0.28,
        medianprops=dict(color=EDGE, linewidth=1.4),
        boxprops=dict(edgecolor=EDGE, linewidth=1.0),
        whiskerprops=dict(color=EDGE, linewidth=0.9),
        capprops=dict(color=EDGE, linewidth=0.9),
    )
    _style_boxplot(bp, colors)
    for i, vals in enumerate(q_values, start=1):
        if len(vals) == 0:
            continue
        rng = np.random.default_rng(100 + i)
        sample = vals if len(vals) <= 36 else rng.choice(vals, 36, replace=False)
        jitter = rng.normal(i, 0.04, size=len(sample))
        ax.scatter(jitter, sample, s=9, alpha=0.22, linewidths=0, color=colors[i - 1], zorder=3)
        ax.text(
            i + 0.10,
            np.nanpercentile(vals, 87) + 0.015,
            f"{results[i - 1].DSD:.3f}",
            ha="left",
            va="bottom",
            fontsize=9.6,
            fontweight="bold",
            color=EDGE,
        )
    ax.set_title("(b) Daily high-state ratio", pad=6)
    ax.set_ylabel(r"$q_d$  (DSD)")
    ax.set_xticks(np.arange(1, len(results) + 1))
    ax.set_xticklabels(labels, rotation=0)
    for tick in ax.get_xticklabels():
        tick.set_fontstyle("normal")
    ax.set_ylim(bottom=0)

    # (C) Conditional future ambiguity.
    ax = axes[1, 0]
    cfa = np.array([r.CFA for r in results], dtype=float)
    bars = ax.bar(x, cfa, width=0.58, color=colors, edgecolor="white", linewidth=0.9)
    for b in bars:
        b.set_alpha(0.88)
    ax.set_title("(c) Conditional future ambiguity", pad=6)
    ax.set_ylabel("CFA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if np.isfinite(cfa).any():
        ax.set_ylim(0, np.nanmax(cfa) * 1.28)
        for rect, val in zip(bars, cfa):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                val + np.nanmax(cfa) * 0.03,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9.6,
                fontweight="bold",
                color=EDGE,
            )

    # (D) Node-level heterogeneity distribution.
    ax = axes[1, 1]
    nh_values = [r.node_het_daily for r in results]
    vp = ax.violinplot(nh_values, positions=np.arange(1, len(results) + 1), widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.22)
    bp = ax.boxplot(
        nh_values,
        positions=np.arange(1, len(results) + 1),
        showfliers=False,
        patch_artist=True,
        widths=0.28,
        medianprops=dict(color=EDGE, linewidth=1.4),
        boxprops=dict(edgecolor=EDGE, linewidth=1.0),
        whiskerprops=dict(color=EDGE, linewidth=0.9),
        capprops=dict(color=EDGE, linewidth=0.9),
    )
    _style_boxplot(bp, colors)
    for i, vals in enumerate(nh_values, start=1):
        if len(vals) == 0:
            continue
        rng = np.random.default_rng(200 + i)
        sample = vals if len(vals) <= 36 else rng.choice(vals, 36, replace=False)
        jitter = rng.normal(i, 0.04, size=len(sample))
        ax.scatter(jitter, sample, s=9, alpha=0.22, linewidths=0, color=colors[i - 1], zorder=3)
        ax.text(
            i + 0.10,
            np.nanpercentile(vals, 87) + 0.007,
            f"{results[i - 1].NH:.3f}",
            ha="left",
            va="bottom",
            fontsize=9.6,
            fontweight="bold",
            color=EDGE,
        )
    ax.set_title("(d) Node-level heterogeneity", pad=6)
    ax.set_ylabel("NH")
    ax.set_xticks(np.arange(1, len(results) + 1))
    ax.set_xticklabels(labels, rotation=0)
    for tick in ax.get_xticklabels():
        tick.set_fontstyle("normal")
    ax.set_ylim(bottom=0)

    for ax in axes.ravel():
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_edgecolor(EDGE)
        ax.tick_params(axis="both", length=2.5, width=0.7)

    fig.savefig(out_dir / "dataset_diagnostics.png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / "dataset_diagnostics.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = parse_dataset_overrides(args.dataset) or default_dataset_paths(args.data_root)
    results: List[DiagnosticResult] = []
    for name, path in dataset_paths.items():
        if not path.exists():
            print(f"[skip] {name}: {path} not found")
            continue
        print(f"[load] {name}: {path}")
        times, X = load_long_csv(path)
        print(f"       shape={X.shape}, time={times[0]} -> {times[-1]}")
        result = compute_diagnostics(
            name=name,
            times=times,
            X=X,
            gamma=args.gamma,
            train_ratio=args.train_ratio,
            L=args.L,
            H=args.H,
            context_bins=args.context_bins,
            tod_bins=args.tod_bins,
            min_group_size=args.min_group_size,
            sample_stride=args.sample_stride,
            min_day_coverage=args.min_day_coverage,
        )
        print(
            f"       DSD={result.DSD:.4f}, CFA={result.CFA:.4f}, "
            f"NH={result.NH:.4f}, days={result.days_used}"
        )
        results.append(result)

    if not results:
        raise RuntimeError("No datasets were found. Check --data-root or --dataset arguments.")

    save_tables(results, args.out_dir)
    plot_diagnostics(results, args.out_dir, args.fig_width, args.fig_height, args.dpi)
    print(f"[done] outputs saved to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
