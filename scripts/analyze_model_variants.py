#!/usr/bin/env python3
"""Analyze progenitor and explosion-energy SNEC model variants."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEC_PER_DAY = 86400.0
L_SUN = 3.998e33
R_SUN = 6.957e10


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    profile: str
    energy_foe: float
    data_dir: Path
    complete: bool = True


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def load_two_column(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, 2)
    keep = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    return data[keep, 0] / SEC_PER_DAY, data[keep, 1]


def load_lightcurve(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    time_days, lum_erg_s = load_two_column(data_dir / "lum_observed.dat")
    return time_days, lum_erg_s / L_SUN


def load_teff(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    time_days, temp = load_two_column(data_dir / "T_eff.dat")
    temp = np.where(temp > 0.0, temp, np.nan)
    return time_days, temp


def load_radius(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    time_days, radius_cm = load_two_column(data_dir / "rad_photo.dat")
    radius_rsun = radius_cm / R_SUN
    # After the photosphere vanishes, SNEC may leave a tiny positive placeholder.
    radius_rsun = np.where(radius_rsun > 100.0, radius_rsun, np.nan)
    valid = np.where(np.isfinite(radius_rsun))[0]
    if len(valid) > 1:
        prev = radius_rsun[valid[:-1]]
        curr = radius_rsun[valid[1:]]
        collapse = np.where((time_days[valid[1:]] > 20.0) & (curr < 0.5 * prev))[0]
        if len(collapse) > 0:
            radius_rsun[valid[collapse[0] + 1]:] = np.nan
    return time_days, radius_rsun


def load_teff_with_photosphere(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    time_days, temp = load_teff(data_dir)
    radius_time, radius = load_radius(data_dir)
    valid_radius = np.isfinite(radius)

    if not np.any(valid_radius):
        return time_days, np.full_like(temp, np.nan)

    first = radius_time[valid_radius][0]
    last = radius_time[valid_radius][-1]
    in_range = (time_days >= first) & (time_days <= last)
    temp = np.where(in_range, temp, np.nan)
    return time_days, temp


def positive(time_days: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(values) & (values > 0)
    return time_days[keep], values[keep]


def interp_at_day(time_days: np.ndarray, values: np.ndarray, day: float) -> float:
    if day < time_days[0] or day > time_days[-1]:
        return float("nan")
    return float(np.interp(day, time_days, values))


def plateau_duration(time_days: np.ndarray, lum_lsun: np.ndarray) -> float:
    mask = (time_days >= 30.0) & (time_days <= 150.0)
    if not np.any(mask):
        return float("nan")

    ref = interp_at_day(time_days, lum_lsun, 50.0)
    if not np.isfinite(ref):
        return float("nan")

    threshold = 0.85 * ref
    t = time_days[mask]
    y = lum_lsun[mask] >= threshold
    if not np.any(y):
        return 0.0

    best = 0.0
    start = None
    for i, ok in enumerate(y):
        if ok and start is None:
            start = t[i]
        elif not ok and start is not None:
            best = max(best, t[i - 1] - start)
            start = None
    if start is not None:
        best = max(best, t[-1] - start)
    return float(best)


def run_specs(root: Path) -> tuple[list[RunSpec], list[RunSpec]]:
    completed = [
        RunSpec(
            "rsg_e1_ni050",
            "RSG 1.0 foe",
            "RSG",
            1.0,
            root / "snec" / "runs" / "week1_nickel_study" / "ni_050" / "Data",
        ),
        RunSpec(
            "rsg_e2_ni050",
            "RSG 2.0 foe",
            "RSG",
            2.0,
            root / "snec" / "runs" / "model_variants" / "rsg_e2_ni050" / "Data",
        ),
        RunSpec(
            "stripped_e1_ni050",
            "Stripped 1.0 foe",
            "Stripped",
            1.0,
            root / "snec" / "runs" / "model_variants" / "stripped_baseline" / "Data",
        ),
        RunSpec(
            "stripped_e2_ni050",
            "Stripped 2.0 foe",
            "Stripped",
            2.0,
            root / "snec" / "runs" / "model_variants" / "stripped_e2_ni050" / "Data",
        ),
    ]
    partial = [
        RunSpec(
            "rsg_e0p5_ni050",
            "RSG 0.5 foe (partial)",
            "RSG",
            0.5,
            root / "snec" / "runs" / "model_variants" / "rsg_e0p5_ni050" / "Data",
            complete=False,
        ),
        RunSpec(
            "stripped_e0p5_ni050",
            "Stripped 0.5 foe (partial)",
            "Stripped",
            0.5,
            root / "snec" / "runs" / "model_variants" / "stripped_e0p5_ni050" / "Data",
            complete=False,
        ),
    ]
    return completed, partial


def validate_specs(specs: list[RunSpec]) -> list[RunSpec]:
    available = []
    for spec in specs:
        path = spec.data_dir / "lum_observed.dat"
        if path.exists():
            available.append(spec)
        else:
            print(f"Skipping {spec.run_id}: missing {path}")
    return available


def plot_lightcurves(specs: list[RunSpec], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), constrained_layout=True)
    ax_full, ax_early, ax_tail = axes
    colors = {
        ("RSG", 1.0): "C0",
        ("RSG", 2.0): "C1",
        ("Stripped", 1.0): "C2",
        ("Stripped", 2.0): "C3",
    }
    styles = {"RSG": "-", "Stripped": "--"}

    for spec in specs:
        time_days, lum_lsun = positive(*load_lightcurve(spec.data_dir))
        color = colors.get((spec.profile, spec.energy_foe), None)
        style = styles.get(spec.profile, "-")

        ax_full.plot(time_days, lum_lsun, style, color=color, linewidth=1.5, label=spec.label)

        early = time_days <= 20.0
        ax_early.plot(
            time_days[early],
            lum_lsun[early],
            style,
            color=color,
            linewidth=1.5,
            label=spec.label,
        )

        tail = time_days >= 20.0
        ax_tail.plot(
            time_days[tail],
            lum_lsun[tail],
            style,
            color=color,
            linewidth=1.5,
            label=spec.label,
        )

    ax_full.set_title("Completed Model Variants: Full Bolometric Light Curves")
    ax_full.set_xlim(0, 200)
    ax_full.set_yscale("log")
    ax_full.set_ylabel(r"Luminosity ($L_\odot$)")
    ax_full.grid(True, which="both", alpha=0.3)
    ax_full.legend(loc="best", framealpha=0.9)

    ax_early.set_title("Early Evolution (0-20 d)")
    ax_early.set_xlim(0, 20)
    ax_early.set_yscale("log")
    ax_early.set_ylabel(r"Luminosity ($L_\odot$)")
    ax_early.grid(True, which="both", alpha=0.3)

    ax_tail.set_title("Plateau and Radioactive Tail (20-200 d)")
    ax_tail.set_xlim(20, 200)
    ax_tail.set_yscale("log")
    ax_tail.set_xlabel("Time since explosion (days)")
    ax_tail.set_ylabel(r"Luminosity ($L_\odot$)")
    ax_tail.grid(True, which="both", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_diagnostics(specs: list[RunSpec], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    ax_t_early, ax_r_early = axes[0]
    ax_t_tail, ax_r_tail = axes[1]
    colors = {
        ("RSG", 1.0): "C0",
        ("RSG", 2.0): "C1",
        ("Stripped", 1.0): "C2",
        ("Stripped", 2.0): "C3",
    }
    styles = {"RSG": "-", "Stripped": "--"}

    for spec in specs:
        color = colors.get((spec.profile, spec.energy_foe), None)
        style = styles.get(spec.profile, "-")
        t_temp, temp = positive(*load_teff_with_photosphere(spec.data_dir))
        t_rad, radius = positive(*load_radius(spec.data_dir))

        early_temp = t_temp <= 20.0
        tail_temp = (t_temp >= 20.0) & (t_temp <= 200.0)
        early_rad = t_rad <= 20.0
        tail_rad = (t_rad >= 20.0) & (t_rad <= 200.0)

        ax_t_early.plot(t_temp[early_temp], temp[early_temp], style, color=color, label=spec.label)
        ax_t_tail.plot(t_temp[tail_temp], temp[tail_temp], style, color=color, label=spec.label)
        ax_r_early.plot(t_rad[early_rad], radius[early_rad], style, color=color, label=spec.label)
        ax_r_tail.plot(t_rad[tail_rad], radius[tail_rad], style, color=color, label=spec.label)

    ax_t_early.set_title(r"Early $T_{\mathrm{eff}}$ (0-20 d)")
    ax_t_early.set_xlim(0, 20)
    ax_t_early.set_ylabel("Temperature (K)")
    ax_t_early.grid(True, alpha=0.3)
    ax_t_early.legend(loc="best", framealpha=0.9)

    ax_r_early.set_title(r"Early $R_{\mathrm{photo}}$ (0-20 d)")
    ax_r_early.set_xlim(0, 20)
    ax_r_early.set_ylabel(r"Radius ($R_\odot$)")
    ax_r_early.grid(True, alpha=0.3)

    ax_t_tail.set_title(r"Plateau/Tail $T_{\mathrm{eff}}$ (20-200 d)")
    ax_t_tail.set_xlim(20, 200)
    ax_t_tail.set_xlabel("Time since explosion (days)")
    ax_t_tail.set_ylabel("Temperature (K)")
    ax_t_tail.grid(True, alpha=0.3)

    ax_r_tail.set_title(r"Plateau/Tail $R_{\mathrm{photo}}$ (20-200 d)")
    ax_r_tail.set_xlim(20, 200)
    ax_r_tail.set_xlabel("Time since explosion (days)")
    ax_r_tail.set_ylabel(r"Radius ($R_\odot$)")
    ax_r_tail.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_partial_low_energy(specs: list[RunSpec], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for spec in specs:
        time_days, lum_lsun = positive(*load_lightcurve(spec.data_dir))
        ax.plot(time_days, lum_lsun, linewidth=1.5, label=f"{spec.label} to {time_days[-1]:.1f} d")

    ax.set_title("Interrupted Low-Energy Runs: Early Progress Only")
    ax.set_xlabel("Time since explosion (days)")
    ax.set_ylabel(r"Luminosity ($L_\odot$)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def metric_record(spec: RunSpec) -> dict[str, float | str | bool]:
    time_days, lum_lsun = positive(*load_lightcurve(spec.data_dir))
    peak_idx = int(np.argmax(lum_lsun))

    t_temp, temp = positive(*load_teff_with_photosphere(spec.data_dir))
    t_rad, radius = positive(*load_radius(spec.data_dir))

    record: dict[str, float | str | bool] = {
        "run_id": spec.run_id,
        "label": spec.label,
        "profile": spec.profile,
        "energy_foe": spec.energy_foe,
        "complete": spec.complete,
        "max_time_days": float(time_days[-1]),
        "peak_time_days": float(time_days[peak_idx]),
        "peak_luminosity_Lsun": float(lum_lsun[peak_idx]),
        "plateau_duration_days": plateau_duration(time_days, lum_lsun),
    }

    for day in (5.0, 20.0, 50.0, 100.0, 150.0):
        record[f"luminosity_{int(day)}d_Lsun"] = interp_at_day(time_days, lum_lsun, day)
        record[f"teff_{int(day)}d_K"] = interp_at_day(t_temp, temp, day)
        record[f"rphoto_{int(day)}d_Rsun"] = interp_at_day(t_rad, radius, day)

    return record


def write_metrics(records: list[dict[str, float | str | bool]], output_dir: Path) -> None:
    csv_path = output_dir / "model_variant_metrics.csv"
    json_path = output_dir / "model_variant_metrics.json"

    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with json_path.open("w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved metrics to {csv_path}")
    print(f"Saved metrics to {json_path}")


def main() -> None:
    setup_plot_style()
    root = repo_root()
    output_dir = root / "results" / "model_variants"
    completed_specs, partial_specs = run_specs(root)
    completed_specs = validate_specs(completed_specs)
    partial_specs = validate_specs(partial_specs)

    if not completed_specs:
        raise FileNotFoundError("No completed model-variant runs found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = [metric_record(spec) for spec in completed_specs + partial_specs]
    write_metrics(records, output_dir)

    plot_lightcurves(completed_specs, output_dir / "lightcurves_energy_progenitor.png")
    plot_diagnostics(completed_specs, output_dir / "diagnostics_energy_progenitor.png")
    if partial_specs:
        plot_partial_low_energy(partial_specs, output_dir / "partial_low_energy_progress.png")

    print(f"Saved plots to {output_dir}")
    for record in records:
        print(
            f"{record['label']}: max_time={record['max_time_days']:.1f} d, "
            f"peak={record['peak_luminosity_Lsun']:.3e} Lsun at "
            f"{record['peak_time_days']:.2f} d"
        )


if __name__ == "__main__":
    main()
