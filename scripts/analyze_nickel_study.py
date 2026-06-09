#!/usr/bin/env python3
"""Analyze Week 1 nickel study runs and produce publication-quality plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEC_PER_DAY = 86400.0
L_SUN = 3.998e33

# Match run_nickel_study.py
NI_MASSES = [0.00, 0.01, 0.03, 0.05, 0.10, 0.15]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ni_run_id(ni_mass: float) -> str:
    return f"ni_{int(round(ni_mass * 1000)):03d}"


def load_lightcurve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, 2)
    time_days = data[:, 0] / SEC_PER_DAY
    lum_lsun = data[:, 1] / L_SUN
    return time_days, lum_lsun


def luminosity_at_day(time_days: np.ndarray, lum: np.ndarray, day: float) -> float:
    return float(np.interp(day, time_days, lum))


def plateau_duration(time_days: np.ndarray, lum: np.ndarray) -> float:
    """
    Plateau duration: span where L > 0.85 * L(50 d), between 30 and 150 days.
    """
    mask = (time_days >= 30) & (time_days <= 150)
    if not np.any(mask):
        return 0.0
    t = time_days[mask]
    l = lum[mask]
    ref = luminosity_at_day(time_days, lum, 50.0)
    threshold = 0.85 * ref
    above = l >= threshold
    if not np.any(above):
        return 0.0
    # Longest contiguous segment above threshold
    best = 0.0
    start = None
    for i, ok in enumerate(above):
        if ok and start is None:
            start = t[i]
        elif not ok and start is not None:
            best = max(best, t[i - 1] - start)
            start = None
    if start is not None:
        best = max(best, t[-1] - start)
    return best


def analyze_run(ni_mass: float, lc_path: Path) -> dict:
    time_days, lum = load_lightcurve(lc_path)
    peak_idx = int(np.argmax(lum))
    return {
        "ni_mass": ni_mass,
        "run_id": ni_run_id(ni_mass),
        "peak_luminosity_Lsun": float(lum[peak_idx]),
        "peak_time_days": float(time_days[peak_idx]),
        "plateau_duration_days": plateau_duration(time_days, lum),
        "luminosity_100d_Lsun": luminosity_at_day(time_days, lum, 100.0),
        "luminosity_150d_Lsun": luminosity_at_day(time_days, lum, 150.0),
        "luminosity_200d_Lsun": luminosity_at_day(time_days, lum, 200.0),
    }


def setup_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def plot_overlaid_lightcurves(
    runs: list[tuple[float, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(runs)))
    for (ni_mass, t, l), color in zip(runs, cmap):
        ax.plot(t, l, label=rf"$M_{{\mathrm{{Ni}}}} = {ni_mass:.2f}\,M_\odot$", color=color)
    ax.set_xlabel("Time since explosion (days)")
    ax.set_ylabel(r"Bolometric luminosity ($L_\odot$)")
    ax.set_title("Week 1: Light Curves vs ⁵⁶Ni Mass (15 M☉ RSG)")
    ax.set_xlim(0, 220)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_vs_ni_mass(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(df["ni_mass"], df[column], "o-", color="C0", markersize=8, linewidth=1.5)
    ax.set_xlabel(r"$M_{\mathrm{Ni}}$ ($M_\odot$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    setup_plot_style()
    root = repo_root()
    runs_root = root / "snec" / "runs" / "week1_nickel_study"
    results_dir = root / "results" / "week1_nickel_study"
    results_dir.mkdir(parents=True, exist_ok=True)

    records = []
    curves: list[tuple[float, np.ndarray, np.ndarray]] = []

    for ni_mass in NI_MASSES:
        run_id = ni_run_id(ni_mass)
        lc_path = runs_root / run_id / "Data" / "lum_observed.dat"
        if not lc_path.exists():
            print(f"Skipping {run_id}: {lc_path} not found")
            continue
        t, l = load_lightcurve(lc_path)
        curves.append((ni_mass, t, l))
        records.append(analyze_run(ni_mass, lc_path))

    if not records:
        print("No completed runs found. Execute scripts/run_nickel_study.py first.")
        return

    df = pd.DataFrame(records).sort_values("ni_mass")
    df.to_csv(results_dir / "nickel_study_metrics.csv", index=False)

    with (results_dir / "nickel_study_metrics.json").open("w") as f:
        json.dump(records, f, indent=2)

    plot_overlaid_lightcurves(curves, results_dir / "lightcurves_overlaid.png")
    plot_vs_ni_mass(
        df,
        "peak_luminosity_Lsun",
        r"Peak luminosity ($L_\odot$)",
        "Peak Luminosity vs ⁵⁶Ni Mass",
        results_dir / "peak_luminosity_vs_ni_mass.png",
    )
    plot_vs_ni_mass(
        df,
        "plateau_duration_days",
        "Plateau duration (days)",
        "Plateau Duration vs ⁵⁶Ni Mass",
        results_dir / "plateau_duration_vs_ni_mass.png",
    )
    plot_vs_ni_mass(
        df,
        "luminosity_150d_Lsun",
        r"$L_{\mathrm{bol}}$ at 150 d ($L_\odot$)",
        "Luminosity at 150 Days vs ⁵⁶Ni Mass",
        results_dir / "luminosity_150d_vs_ni_mass.png",
    )

    print(f"Analysis complete. Results in {results_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
