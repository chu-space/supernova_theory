#!/usr/bin/env python3
"""Compare the first 0-48 hours of the WN3 shock-cooling grid."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEC_PER_HOUR = 3600.0
L_SUN = 3.998e33
R_SUN = 6.957e10

SAMPLE_HOURS = [0.5, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def setup_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_two_column(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, 2)
    keep = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    return data[keep, 0] / SEC_PER_HOUR, data[keep, 1]


def positive(hours: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(values) & (values > 0.0)
    return hours[keep], values[keep]


def load_lightcurve(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    hours, luminosity = load_two_column(data_dir / "lum_observed.dat")
    return positive(hours, luminosity / L_SUN)


def load_teff(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    hours, temperature = load_two_column(data_dir / "T_eff.dat")
    return positive(hours, temperature)


def load_radius(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    hours, radius_cm = load_two_column(data_dir / "rad_photo.dat")
    return positive(hours, radius_cm / R_SUN)


def interp_at_hour(hours: np.ndarray, values: np.ndarray, hour: float) -> float:
    if len(hours) == 0 or hour < hours[0] or hour > hours[-1]:
        return float("nan")
    return float(np.interp(hour, hours, values))


def available_runs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    available = []
    for row in rows:
        data_dir = Path(row["data_dir"])
        required = [
            data_dir / "lum_observed.dat",
            data_dir / "T_eff.dat",
            data_dir / "rad_photo.dat",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            print(f"Skipping {row['run_id']}: missing {missing[0]}")
            continue
        available.append(row)
    return available


def label_for(row: dict[str, str], max_hours: float | None = None) -> str:
    label = f"{row['label']}, {float(row['energy_foe']):g} foe"
    if max_hours is not None and max_hours < 47.9:
        label = f"{label} (partial {max_hours:.1f} h)"
    return label


def plot_grid(rows: list[dict[str, str]], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True, sharex=True)
    ax_lum, ax_teff, ax_radius = axes

    colors = {
        "bare": "C0",
        "mass_m0p01_r5": "C1",
        "radius_m0p01_r50": "C2",
    }
    styles = {0.5: ":", 1.0: "-", 2.0: "--"}

    for row in rows:
        data_dir = Path(row["data_dir"])
        structure_key = row["structure_key"]
        energy = float(row["energy_foe"])
        color = colors.get(structure_key, None)
        style = styles.get(energy, "-")
        h_lum, lum = load_lightcurve(data_dir)
        h_teff, teff = load_teff(data_dir)
        h_radius, radius = load_radius(data_dir)
        max_hours = min(h_lum[-1], h_teff[-1], h_radius[-1])
        label = label_for(row, max_hours=max_hours)

        mask_lum = h_lum <= 48.0
        mask_teff = h_teff <= 48.0
        mask_radius = h_radius <= 48.0

        ax_lum.plot(h_lum[mask_lum], lum[mask_lum], style, color=color, label=label)
        ax_teff.plot(h_teff[mask_teff], teff[mask_teff], style, color=color, label=label)
        ax_radius.plot(h_radius[mask_radius], radius[mask_radius], style, color=color, label=label)

    ax_lum.set_title("WN3 SCE Grid: Luminosity, 0-48 Hours")
    ax_lum.set_ylabel(r"Luminosity ($L_\odot$)")
    ax_lum.set_yscale("log")

    ax_teff.set_title(r"Effective Temperature, 0-48 Hours")
    ax_teff.set_ylabel("Temperature (K)")
    ax_teff.set_yscale("log")

    ax_radius.set_title(r"Photospheric Radius, 0-48 Hours")
    ax_radius.set_xlabel("Time since explosion (hours)")
    ax_radius.set_ylabel(r"Radius ($R_\odot$)")
    ax_radius.set_yscale("log")

    for ax in axes:
        ax.set_xlim(0, 48)
        ax.grid(True, which="both", alpha=0.3)

    handles, labels = ax_lum.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        framealpha=0.9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def metric_record(row: dict[str, str]) -> dict[str, float | str]:
    data_dir = Path(row["data_dir"])
    h_lum, lum = load_lightcurve(data_dir)
    h_teff, teff = load_teff(data_dir)
    h_radius, radius = load_radius(data_dir)

    early = h_lum <= 48.0
    peak_idx = int(np.argmax(lum[early]))
    peak_hours = h_lum[early][peak_idx]
    peak_lum = lum[early][peak_idx]
    max_hours = min(h_lum[-1], h_teff[-1], h_radius[-1])

    record: dict[str, float | str] = {
        "run_id": row["run_id"],
        "label": row["label"],
        "structure_key": row["structure_key"],
        "role": row["role"],
        "profile": row["profile"],
        "energy_foe": float(row["energy_foe"]),
        "ni_mass_msun": float(row["ni_mass_msun"]),
        "max_time_hours": float(max_hours),
        "complete_48h": "yes" if max_hours >= 47.9 else "no",
        "peak_time_hours_0_48h": float(peak_hours),
        "peak_luminosity_Lsun_0_48h": float(peak_lum),
    }

    for hour in SAMPLE_HOURS:
        tag = str(hour).replace(".", "p")
        record[f"luminosity_{tag}h_Lsun"] = interp_at_hour(h_lum, lum, hour)
        record[f"teff_{tag}h_K"] = interp_at_hour(h_teff, teff, hour)
        record[f"rphoto_{tag}h_Rsun"] = interp_at_hour(h_radius, radius, hour)

    return record


def write_metrics(records: list[dict[str, float | str]], output_dir: Path) -> None:
    csv_path = output_dir / "wn3_sce_grid_metrics.csv"
    json_path = output_dir / "wn3_sce_grid_metrics.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with json_path.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved metrics to {csv_path}")
    print(f"Saved metrics to {json_path}")


def main() -> None:
    setup_plot_style()
    root = repo_root()
    output_dir = root / "results" / "wn3_sce_grid"
    manifest_path = output_dir / "wn3_sce_grid_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Grid manifest not found: {manifest_path}\n"
            "Run scripts/run_wn3_sce_grid.py --dry-run first."
        )

    rows = available_runs(load_manifest(manifest_path))
    if not rows:
        print("No completed WN3 SCE grid runs found yet.")
        return

    records = [metric_record(row) for row in rows]
    write_metrics(records, output_dir)
    plot_grid(rows, output_dir / "wn3_sce_grid_0_48h.png")
    print(f"Saved plot to {output_dir / 'wn3_sce_grid_0_48h.png'}")

    for record in records:
        print(
            f"{record['run_id']}: peak={record['peak_luminosity_Lsun_0_48h']:.3e} "
            f"Lsun at {record['peak_time_hours_0_48h']:.2f} h"
        )


if __name__ == "__main__":
    main()
