#!/usr/bin/env python3
"""Plot temperature and photospheric-radius diagnostics for completed SNEC runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEC_PER_DAY = 86400.0
R_SUN = 6.957e10

# Match the intentionally small low/mid/high nickel study.
NI_MASSES = [0.00, 0.05, 0.15]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ni_run_id(ni_mass: float) -> str:
    return f"ni_{int(round(ni_mass * 1000)):03d}"


def load_two_column(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a SNEC scalar output file and convert the first column to days."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, 2)

    time_days = data[:, 0] / SEC_PER_DAY
    values = data[:, 1]
    keep = np.isfinite(time_days) & np.isfinite(values)
    return time_days[keep], values[keep]


def positive_values(
    time_days: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop zeros before log plotting."""
    keep = values > 0
    return time_days[keep], values[keep]


def setup_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def plot_diagnostics(
    runs: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8.5),
        constrained_layout=True,
        sharex="col",
    )
    ax_teff_full, ax_teff_early = axes[0]
    ax_rad_full, ax_rad_early = axes[1]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(runs)))

    for (ni_mass, t_temp, temp, t_rad, radius_cm), color in zip(runs, colors):
        label = rf"$M_{{\mathrm{{Ni}}}} = {ni_mass:.2f}\,M_\odot$"

        t_temp_pos, temp_pos = positive_values(t_temp, temp)
        radius_rsun = radius_cm / R_SUN
        t_rad_pos, radius_pos = positive_values(t_rad, radius_rsun)

        ax_teff_full.plot(t_temp_pos, temp_pos, color=color, linewidth=1.5, label=label)
        ax_teff_early.plot(t_temp_pos, temp_pos, color=color, linewidth=1.5)
        ax_rad_full.plot(t_rad_pos, radius_pos, color=color, linewidth=1.5, label=label)
        ax_rad_early.plot(t_rad_pos, radius_pos, color=color, linewidth=1.5)

    ax_teff_full.set_title("Effective Temperature Evolution")
    ax_teff_full.set_ylabel(r"$T_{\mathrm{eff}}$ (K)")
    ax_teff_full.set_xlim(0, 220)
    ax_teff_full.grid(True, alpha=0.3)
    ax_teff_full.legend(loc="upper right", framealpha=0.9)

    ax_teff_early.set_title("Early Effective Temperature (0-20 d)")
    ax_teff_early.set_xlim(0, 20)
    ax_teff_early.grid(True, alpha=0.3)

    ax_rad_full.set_title("Photospheric Radius Evolution")
    ax_rad_full.set_xlabel("Time since explosion (days)")
    ax_rad_full.set_ylabel(r"$R_{\mathrm{photo}}$ ($R_\odot$)")
    ax_rad_full.set_xlim(0, 220)
    ax_rad_full.grid(True, alpha=0.3)

    ax_rad_early.set_title("Early Photospheric Radius (0-20 d)")
    ax_rad_early.set_xlabel("Time since explosion (days)")
    ax_rad_early.set_xlim(0, 20)
    ax_rad_early.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=repo_root() / "snec" / "runs" / "week1_nickel_study",
        help="Directory containing ni_*/Data outputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "results" / "week1_nickel_study" / "supernova_diagnostics.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--masses",
        nargs="+",
        type=float,
        default=NI_MASSES,
        help="Ni_mass values to plot",
    )
    args = parser.parse_args()

    setup_plot_style()
    runs = []

    for ni_mass in args.masses:
        run_id = ni_run_id(ni_mass)
        data_dir = args.runs_root / run_id / "Data"
        temp_path = data_dir / "T_eff.dat"
        radius_path = data_dir / "rad_photo.dat"

        if not temp_path.exists() or not radius_path.exists():
            print(f"Skipping {run_id}: missing T_eff.dat or rad_photo.dat")
            continue

        t_temp, temp = load_two_column(temp_path)
        t_rad, radius = load_two_column(radius_path)
        runs.append((ni_mass, t_temp, temp, t_rad, radius))

    if not runs:
        raise FileNotFoundError(
            f"No completed diagnostics found under {args.runs_root}. "
            "Run scripts/run_nickel_study.py first."
        )

    plot_diagnostics(runs, args.output)
    print(f"Saved diagnostics plot to {args.output}")


if __name__ == "__main__":
    main()
