#!/usr/bin/env python3
"""Plot bolometric light curve from SNEC lum_observed.dat output."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Seconds per day
SEC_PER_DAY = 86400.0
# Solar luminosity (erg/s)
L_SUN = 3.998e33


def load_lightcurve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load time (days) and luminosity (L_sun) from SNEC scalar output in c.g.s units."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, 2)
    time_s = data[:, 0]
    lum_erg_s = data[:, 1]
    time_days = time_s / SEC_PER_DAY
    lum_lsun = lum_erg_s / L_SUN
    return time_days, lum_lsun


def plot_lightcurve(
    time_days: np.ndarray,
    lum_lsun: np.ndarray,
    output_path: Path,
    title: str = "SNEC Bolometric Light Curve",
) -> None:

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(9, 8),
        constrained_layout=True
    )

    # ----------------------------
    # Full light curve (log scale)
    # ----------------------------
    ax1.plot(time_days, lum_lsun, linewidth=1.5)

    ax1.set_yscale("log")
    ax1.set_xlabel("Time since explosion (days)")
    ax1.set_ylabel(r"Luminosity ($L_\odot$)")
    ax1.set_title(title)
    ax1.grid(True, which="both", alpha=0.3)

    # ----------------------------
    # Early-time zoom
    # ----------------------------
    mask = time_days <= 10

    ax2.plot(
        time_days[mask],
        lum_lsun[mask],
        linewidth=1.5
    )

    ax2.set_xlabel("Time since explosion (days)")
    ax2.set_ylabel(r"Luminosity ($L_\odot$)")
    ax2.set_title("Early-Time Evolution (0–10 days)")
    ax2.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=250)
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "snec" / "Data" / "lum_observed.dat",
        help="Path to lum_observed.dat",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "results" / "baseline" / "lightcurve.png",
        help="Output PNG path",
    )
    parser.add_argument("--title", default="15 M☉ RSG Baseline — Bolometric Light Curve")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Light curve file not found: {args.input}\n"
            "Run the SNEC baseline simulation first."
        )

    time_days, lum_lsun = load_lightcurve(args.input)
    plot_lightcurve(time_days, lum_lsun, args.output, title=args.title)


if __name__ == "__main__":
    main()
