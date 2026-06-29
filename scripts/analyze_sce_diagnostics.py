#!/usr/bin/env python3
"""Make polished shock-cooling diagnostics for WN3-grid and Avishai models."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

SEC_PER_HOUR = 3600.0
L_SUN = 3.998e33
R_SUN = 6.957e10

SAMPLE_HOURS = [0.25, 0.5, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0]


@dataclass(frozen=True)
class ModelSpec:
    run_id: str
    label: str
    group: str
    profile: str
    energy_foe: float
    ni_mass_msun: float
    data_dir: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def setup_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
    })


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


def interp_at(hours: np.ndarray, values: np.ndarray, sample_hour: float) -> float:
    if len(hours) == 0 or sample_hour < hours[0] or sample_hour > hours[-1]:
        return float("nan")
    return float(np.interp(sample_hour, hours, values))


def required_outputs_exist(spec: ModelSpec) -> bool:
    return all(
        (spec.data_dir / name).exists()
        for name in ("lum_observed.dat", "T_eff.dat", "rad_photo.dat")
    )


def load_wn3_manifest(root: Path) -> list[ModelSpec]:
    manifest_path = root / "results" / "wn3_sce_grid" / "wn3_sce_grid_manifest.csv"
    if not manifest_path.exists():
        return []

    specs: list[ModelSpec] = []
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            specs.append(
                ModelSpec(
                    run_id=row["run_id"],
                    label=f"{row['label']}, {float(row['energy_foe']):g} foe",
                    group=row["structure_key"],
                    profile=row["profile"],
                    energy_foe=float(row["energy_foe"]),
                    ni_mass_msun=float(row["ni_mass_msun"]),
                    data_dir=Path(row["data_dir"]),
                )
            )
    return [spec for spec in specs if required_outputs_exist(spec)]


def avishai_original_specs(root: Path) -> list[ModelSpec]:
    runs_root = root / "snec" / "runs" / "model_variants"
    specs = [
        ModelSpec(
            "wn3_sce_bare_e1_ni050",
            "Bare Avishai WN3",
            "Avishai original",
            "WN3",
            1.0,
            0.05,
            runs_root / "wn3_sce_bare_e1_ni050" / "Data",
        ),
        ModelSpec(
            "avishai_bsg2_sce_e1_ni050",
            "Avishai BSG2",
            "Avishai original",
            "BSG2",
            1.0,
            0.05,
            runs_root / "avishai_bsg2_sce_e1_ni050" / "Data",
        ),
        ModelSpec(
            "avishai_bsg3_sce_e1_ni050",
            "Avishai BSG3",
            "Avishai original",
            "BSG3",
            1.0,
            0.05,
            runs_root / "avishai_bsg3_sce_e1_ni050" / "Data",
        ),
    ]
    return [spec for spec in specs if required_outputs_exist(spec)]


def metric_record(spec: ModelSpec, window_hours: float = 48.0) -> dict[str, float | str]:
    h_lum, lum = load_lightcurve(spec.data_dir)
    h_teff, teff = load_teff(spec.data_dir)
    h_radius, radius = load_radius(spec.data_dir)

    in_window = h_lum <= window_hours
    if not np.any(in_window):
        in_window = np.ones_like(h_lum, dtype=bool)
    peak_idx = int(np.argmax(lum[in_window]))
    peak_hours = h_lum[in_window][peak_idx]
    peak_lum = lum[in_window][peak_idx]
    max_hours = min(h_lum[-1], h_teff[-1], h_radius[-1])

    record: dict[str, float | str] = {
        "run_id": spec.run_id,
        "label": spec.label,
        "group": spec.group,
        "profile": spec.profile,
        "energy_foe": spec.energy_foe,
        "ni_mass_msun": spec.ni_mass_msun,
        "max_time_hours": float(max_hours),
        "peak_time_hours_0_48h": float(peak_hours),
        "peak_luminosity_Lsun_0_48h": float(peak_lum),
    }
    for hour in SAMPLE_HOURS:
        tag = str(hour).replace(".", "p")
        record[f"luminosity_{tag}h_Lsun"] = interp_at(h_lum, lum, hour)
        record[f"teff_{tag}h_K"] = interp_at(h_teff, teff, hour)
        record[f"rphoto_{tag}h_Rsun"] = interp_at(h_radius, radius, hour)
    return record


def write_records(records: list[dict[str, float | str]], csv_path: Path, json_path: Path) -> None:
    if not records:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with json_path.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved table to {csv_path}")
    print(f"Saved table to {json_path}")


def plot_luminosity_windows(
    specs: list[ModelSpec],
    output_path: Path,
    title: str,
    color_by: dict[str, str],
    style_by_energy: bool = True,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    windows = [(1.0, "First Hour"), (6.0, "First 6 Hours"), (48.0, "First 48 Hours")]
    styles = {0.5: ":", 1.0: "-", 2.0: "--"}

    for ax, (limit, panel_title) in zip(axes, windows):
        for spec in specs:
            h_lum, lum = load_lightcurve(spec.data_dir)
            mask = h_lum <= limit
            if not np.any(mask):
                continue
            color = color_by.get(spec.group, color_by.get(spec.profile, None))
            style = styles.get(spec.energy_foe, "-") if style_by_energy else "-"
            ax.plot(
                h_lum[mask],
                lum[mask],
                style,
                color=color,
                linewidth=1.8,
                label=spec.label,
            )
        ax.set_title(panel_title)
        ax.set_xlim(0, limit)
        ax.set_yscale("log")
        ax.set_ylabel(r"Luminosity ($L_\odot$)")
        ax.grid(True, which="both", alpha=0.28)

    axes[-1].set_xlabel("Time since explosion (hours)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, y=1.02, fontsize=15)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_three_panel_diagnostics(
    specs: list[ModelSpec],
    output_path: Path,
    title: str,
    color_by: dict[str, str],
    style_by_energy: bool = True,
    limit_hours: float = 48.0,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True, sharex=True)
    ax_lum, ax_teff, ax_radius = axes
    styles = {0.5: ":", 1.0: "-", 2.0: "--"}

    for spec in specs:
        color = color_by.get(spec.group, color_by.get(spec.profile, None))
        style = styles.get(spec.energy_foe, "-") if style_by_energy else "-"
        label = spec.label

        h_lum, lum = load_lightcurve(spec.data_dir)
        h_teff, teff = load_teff(spec.data_dir)
        h_radius, radius = load_radius(spec.data_dir)

        lum_mask = h_lum <= limit_hours
        teff_mask = h_teff <= limit_hours
        radius_mask = h_radius <= limit_hours

        ax_lum.plot(h_lum[lum_mask], lum[lum_mask], style, color=color, linewidth=1.8, label=label)
        ax_teff.plot(h_teff[teff_mask], teff[teff_mask], style, color=color, linewidth=1.8, label=label)
        ax_radius.plot(
            h_radius[radius_mask],
            radius[radius_mask],
            style,
            color=color,
            linewidth=1.8,
            label=label,
        )

    ax_lum.set_title("Bolometric luminosity")
    ax_lum.set_ylabel(r"Luminosity ($L_\odot$)")
    ax_lum.set_yscale("log")

    ax_teff.set_title(r"Effective temperature")
    ax_teff.set_ylabel("Temperature (K)")
    ax_teff.set_yscale("log")

    ax_radius.set_title(r"Photospheric radius")
    ax_radius.set_xlabel("Time since explosion (hours)")
    ax_radius.set_ylabel(r"Radius ($R_\odot$)")
    ax_radius.set_yscale("log")

    for ax in axes:
        ax.set_xlim(0, limit_hours)
        ax.grid(True, which="both", alpha=0.28)

    handles, labels = ax_lum.get_legend_handles_labels()
    fig.suptitle(title, y=1.02, fontsize=15)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def format_sci(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if not np.isfinite(value):
        return "nan"
    return f"{value:.3e}"


def write_markdown_summary(
    wn3_records: list[dict[str, float | str]],
    avishai_records: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    lines = [
        "# Shock-Cooling Diagnostic Summary",
        "",
        "All peak values are measured over the first 48 hours.",
        "",
        "## WN3 Grid",
        "",
        "| Model | Energy (foe) | Peak time (h) | Peak luminosity (Lsun) | L(1h) (Lsun) | Teff(1h) (K) | Rphoto(1h) (Rsun) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in wn3_records:
        lines.append(
            "| {label} | {energy:.1f} | {tpeak:.3f} | {lpeak} | {l1} | {t1} | {r1} |".format(
                label=record["label"],
                energy=float(record["energy_foe"]),
                tpeak=float(record["peak_time_hours_0_48h"]),
                lpeak=format_sci(record["peak_luminosity_Lsun_0_48h"]),
                l1=format_sci(record["luminosity_1p0h_Lsun"]),
                t1=format_sci(record["teff_1p0h_K"]),
                r1=format_sci(record["rphoto_1p0h_Rsun"]),
            )
        )

    lines.extend([
        "",
        "## Original Avishai Models",
        "",
        "| Model | Peak time (h) | Peak luminosity (Lsun) | L(1h) (Lsun) | L(6h) (Lsun) | Teff(1h) (K) | Rphoto(1h) (Rsun) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for record in avishai_records:
        lines.append(
            "| {label} | {tpeak:.3f} | {lpeak} | {l1} | {l6} | {t1} | {r1} |".format(
                label=record["label"],
                tpeak=float(record["peak_time_hours_0_48h"]),
                lpeak=format_sci(record["peak_luminosity_Lsun_0_48h"]),
                l1=format_sci(record["luminosity_1p0h_Lsun"]),
                l6=format_sci(record["luminosity_6p0h_Lsun"]),
                t1=format_sci(record["teff_1p0h_K"]),
                r1=format_sci(record["rphoto_1p0h_Rsun"]),
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved summary to {output_path}")


def main() -> None:
    setup_plot_style()
    root = repo_root()
    output_dir = root / "results" / "sce_diagnostics"

    wn3_specs = load_wn3_manifest(root)
    avishai_specs = avishai_original_specs(root)

    if not wn3_specs:
        print("No completed WN3 grid runs found.")
    if len(avishai_specs) < 3:
        available = ", ".join(spec.run_id for spec in avishai_specs) or "none"
        print(f"Only found Avishai original outputs for: {available}")

    wn3_records = [metric_record(spec) for spec in wn3_specs]
    avishai_records = [metric_record(spec) for spec in avishai_specs]

    write_records(
        wn3_records,
        output_dir / "wn3_grid_summary_metrics.csv",
        output_dir / "wn3_grid_summary_metrics.json",
    )
    write_records(
        avishai_records,
        output_dir / "avishai_original_summary_metrics.csv",
        output_dir / "avishai_original_summary_metrics.json",
    )

    if wn3_specs:
        wn3_colors = {
            "bare": "C0",
            "mass_m0p01_r5": "C1",
            "radius_m0p01_r50": "C2",
        }
        plot_luminosity_windows(
            wn3_specs,
            output_dir / "wn3_grid_luminosity_zoom_windows.png",
            "WN3 Grid: Shock-Cooling Luminosity Windows",
            wn3_colors,
            style_by_energy=True,
        )
        plot_three_panel_diagnostics(
            wn3_specs,
            output_dir / "wn3_grid_diagnostics_0_48h.png",
            "WN3 Grid: Luminosity, Temperature, Radius",
            wn3_colors,
            style_by_energy=True,
            limit_hours=48.0,
        )

    if avishai_specs:
        avishai_colors = {"WN3": "C0", "BSG2": "C3", "BSG3": "C4"}
        plot_luminosity_windows(
            avishai_specs,
            output_dir / "avishai_original_luminosity_zoom_windows.png",
            "Original Avishai Models: Shock-Cooling Luminosity Windows",
            avishai_colors,
            style_by_energy=False,
        )
        plot_three_panel_diagnostics(
            avishai_specs,
            output_dir / "avishai_original_diagnostics_0_48h.png",
            "Original Avishai Models: Luminosity, Temperature, Radius",
            avishai_colors,
            style_by_energy=False,
            limit_hours=48.0,
        )

    write_markdown_summary(
        wn3_records,
        avishai_records,
        output_dir / "sce_diagnostic_summary.md",
    )


if __name__ == "__main__":
    main()
