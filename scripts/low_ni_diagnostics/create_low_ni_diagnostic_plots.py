#!/usr/bin/env python3
"""Create low-Ni WN3 diagnostic plots from completed SNEC outputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SNEC_ROOT = Path("/Users/arifchu/Desktop/GitHub/supernova_theory")
OUT_DIR = Path("/Users/arifchu/Documents/Codex/2026-06-09/scientific-purpose-looks-like-you-re/output/low_ni_diagnostics")

SEC_PER_HOUR = 3600.0
L_SUN = 3.998e33
R_SUN = 6.957e10
M_SUN = 1.98847e33
SIGMA_SB = 5.670374419e-5


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    structure: str
    data_dir: Path


RUNS = [
    RunSpec(
        "wn3_sce_bare_e1_ni001",
        "Bare WN3, Ni=0.001",
        "bare",
        SNEC_ROOT / "snec" / "runs" / "model_variants" / "wn3_sce_bare_e1_ni001" / "Data",
    ),
    RunSpec(
        "wn3_sce_mass_m0p01_r5_e1_ni001",
        "WN3 + 0.01 Msol to 5 Rsun, Ni=0.001",
        "m0p01_r5",
        SNEC_ROOT / "snec" / "runs" / "model_variants" / "wn3_sce_mass_m0p01_r5_e1_ni001" / "Data",
    ),
    RunSpec(
        "wn3_sce_radius_m0p01_r50_e1_ni001",
        "WN3 + 0.01 Msol to 50 Rsun, Ni=0.001",
        "m0p01_r50",
        SNEC_ROOT / "snec" / "runs" / "model_variants" / "wn3_sce_radius_m0p01_r50_e1_ni001" / "Data",
    ),
]

PROFILE_SPECS = [
    (
        "WN3 + 0.01 Msol to 50 Rsun",
        SNEC_ROOT / "snec" / "profiles" / "generated" / "avishai_wn3_ext_m0p01_r50.short",
        "C2",
        "-",
        1.5,
        2,
    ),
    (
        "WN3 + 0.01 Msol to 5 Rsun",
        SNEC_ROOT / "snec" / "profiles" / "generated" / "avishai_wn3_ext_m0p01_r5.short",
        "C1",
        "-.",
        1.7,
        3,
    ),
    (
        "Bare Avishai WN3 control",
        SNEC_ROOT / "avishai_models" / "MW-M25M13.75P4-primary-WN3.short",
        "black",
        "--",
        2.4,
        5,
    ),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
        }
    )


def style_paper_axis(ax) -> None:
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=3)


def load_two_column(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, 2)
    keep = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    return data[keep, 0] / SEC_PER_HOUR, data[keep, 1]


def positive(hours: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(values) & (values > 0)
    return hours[keep], values[keep]


def load_luminosity(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    hours, lum = load_two_column(data_dir / "lum_observed.dat")
    return positive(hours, lum / L_SUN)


def load_luminosity_cgs(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    hours, lum = load_two_column(data_dir / "lum_observed.dat")
    return positive(hours, lum)


def load_teff(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    hours, teff = load_two_column(data_dir / "T_eff.dat")
    return positive(hours, teff)


def load_radius(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    hours, radius = load_two_column(data_dir / "rad_photo.dat")
    return positive(hours, radius / R_SUN)


def load_radius_cgs(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    hours, radius = load_two_column(data_dir / "rad_photo.dat")
    return positive(hours, radius)


def interp_at(hours: np.ndarray, values: np.ndarray, sample_hour: float) -> float:
    if len(hours) == 0 or sample_hour < hours[0] or sample_hour > hours[-1]:
        return float("nan")
    return float(np.interp(sample_hour, hours, values))


def blackbody_luminosity_from_teff_radius(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    h_teff, teff = load_teff(data_dir)
    h_radius, radius_cm = load_radius_cgs(data_dir)
    keep = (h_teff >= h_radius[0]) & (h_teff <= h_radius[-1])
    hours = h_teff[keep]
    radius_interp = np.interp(hours, h_radius, radius_cm)
    lum_lsun = 4.0 * np.pi * radius_interp**2 * SIGMA_SB * teff[keep] ** 4 / L_SUN
    return positive(hours, lum_lsun)


def plot_luminosity_zoom_windows() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / "low_ni_luminosity_zoom_windows.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    windows = [(1.0, "First hour"), (6.0, "First 6 hours"), (48.0, "First 48 hours")]
    colors = {"bare": "C0", "m0p01_r5": "C1", "m0p01_r50": "C2"}

    for ax, (limit, title) in zip(axes, windows):
        for spec in RUNS:
            h, lum = load_luminosity(spec.data_dir)
            mask = h <= limit
            ax.plot(h[mask], lum[mask], color=colors[spec.structure], linewidth=1.8, label=spec.label)
        ax.set_title(title)
        ax.set_xlim(0, limit)
        ax.set_yscale("log")
        ax.set_ylabel(r"Luminosity ($L_\odot$)")
        style_paper_axis(ax)

    axes[-1].set_xlabel("Time since explosion (hours)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Low-Ni WN3 Shock-Cooling Luminosity Windows", y=1.02)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=1)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_three_panel_diagnostics() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / "low_ni_diagnostics_0_48h.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True, sharex=True)
    colors = {"bare": "C0", "m0p01_r5": "C1", "m0p01_r50": "C2"}

    for spec in RUNS:
        h_lum, lum = load_luminosity(spec.data_dir)
        h_teff, teff = load_teff(spec.data_dir)
        h_radius, radius = load_radius(spec.data_dir)
        color = colors[spec.structure]

        axes[0].plot(h_lum[h_lum <= 48], lum[h_lum <= 48], color=color, linewidth=1.8, label=spec.label)
        axes[1].plot(h_teff[h_teff <= 48], teff[h_teff <= 48], color=color, linewidth=1.8, label=spec.label)
        axes[2].plot(h_radius[h_radius <= 48], radius[h_radius <= 48], color=color, linewidth=1.8, label=spec.label)

    axes[0].set_title("Bolometric luminosity")
    axes[0].set_ylabel(r"Luminosity ($L_\odot$)")
    axes[0].set_yscale("log")
    axes[1].set_title(r"Effective temperature")
    axes[1].set_ylabel("Temperature (K)")
    axes[1].set_yscale("log")
    axes[2].set_title(r"Photospheric radius")
    axes[2].set_xlabel("Time since explosion (hours)")
    axes[2].set_ylabel(r"Radius ($R_\odot$)")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.set_xlim(0, 48)
        style_paper_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Low-Ni WN3 Diagnostics: Luminosity, Temperature, Radius", y=1.02)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=1)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_luminosity_cgs_comparison() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / "low_ni_bolometric_luminosity_cgs_0_48h.png"
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = {"bare": "C0", "m0p01_r5": "C1", "m0p01_r50": "C2"}
    for spec in RUNS:
        h, lum = load_luminosity_cgs(spec.data_dir)
        mask = h <= 48
        ax.plot(h[mask], lum[mask], color=colors[spec.structure], linewidth=1.8, label=spec.label)
    ax.set_title("Low-Ni WN3 Bolometric Luminosity, 0-48 h")
    ax.set_xlabel("Time since explosion (hours)")
    ax.set_ylabel("Luminosity (erg/s)")
    ax.set_yscale("log")
    style_paper_axis(ax)
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_presentation_theory_overlay() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / "low_ni_presentation_theory_overlay.png"
    colors = {"bare": "C0", "m0p01_r5": "C1", "m0p01_r50": "C2"}
    zoom_limit_h = 6.0

    fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[2.35, 1.0, 1.0],
        height_ratios=[1.45, 0.55, 1.0],
    )
    ax = fig.add_subplot(gs[:2, 0])
    ax_ratio = fig.add_subplot(gs[2, 0], sharex=ax)
    ax_models = fig.add_subplot(gs[0, 1:])
    ax_theory = fig.add_subplot(gs[1:, 1:])

    for spec in RUNS:
        h_lum, lum = load_luminosity(spec.data_dir)
        h_bb, lum_bb = blackbody_luminosity_from_teff_radius(spec.data_dir)
        color = colors[spec.structure]
        mask = (h_lum <= zoom_limit_h) & (h_lum >= 0.01)
        bb_mask = (h_bb <= zoom_limit_h) & (h_bb >= 0.01)
        ax.plot(h_lum[mask], lum[mask], color=color, linewidth=2.4, label=spec.label)
        ax.plot(
            h_bb[bb_mask],
            lum_bb[bb_mask],
            color="black",
            linestyle=":",
            linewidth=2.2,
            alpha=0.88,
            zorder=6,
        )

        ratio_mask = (
            (h_lum <= zoom_limit_h)
            & (h_lum >= max(0.01, h_bb[0]))
            & (h_lum <= h_bb[-1])
            & (lum > 1.0e6)
        )
        bb_interp = np.interp(h_lum[ratio_mask], h_bb, lum_bb)
        ax_ratio.plot(
            h_lum[ratio_mask],
            bb_interp / lum[ratio_mask],
            color=color,
            linewidth=1.8,
        )

    ax.set_title("Early-Time Low-Ni WN3 Shock-Cooling: SNEC vs Thermal Theory")
    ax.set_ylabel(r"Luminosity ($L_\odot$)")
    ax.set_xlim(0, zoom_limit_h)
    ax.set_ylim(1.0e6, 3.0e11)
    ax.set_yscale("log")
    style_paper_axis(ax)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.text(
        0.03,
        0.95,
        "solid: SNEC lum_observed\n"
        r"black dotted: $4\pi R_{\rm ph}^{2}\sigma_{\rm SB}T_{\rm eff}^{4}$",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 5},
    )

    ax_ratio.axhline(1.0, color="0.25", linewidth=1.0)
    ax_ratio.set_title("Theory / SNEC luminosity ratio", fontsize=10)
    ax_ratio.set_xlabel("Time since explosion (hours)")
    ax_ratio.set_ylabel(r"$L_{\rm BB}/L_{\rm SNEC}$")
    ax_ratio.set_xlim(0, zoom_limit_h)
    ax_ratio.set_ylim(0.985, 1.015)
    style_paper_axis(ax_ratio)

    for text_ax in (ax_models, ax_theory):
        text_ax.axis("off")

    ax_models.set_title("Model setup", loc="left", fontsize=13, pad=8)
    model_text = (
        "Fixed across all runs:\n"
        "  E = 1 foe, Ni = 0.001 M_sun\n"
        "  Same Avishai WN3 core structure\n"
        "  Added material is WN-surface, He-rich gas\n\n"
        "Bare WN3 control:\n"
        "  no added material; R_* = 0.55 R_sun\n\n"
        "Compact extension:\n"
        "  +0.01 M_sun to R_out = 5 R_sun\n\n"
        "Extended shell/wind proxy:\n"
        "  +0.01 M_sun to R_out = 50 R_sun"
    )
    ax_models.text(
        0.0,
        1.0,
        model_text,
        va="top",
        family="monospace",
        fontsize=10.5,
        linespacing=1.25,
    )

    ax_theory.set_title("Theory being tested", loc="left", fontsize=13, pad=8)
    theory_text = (
        "Shock-cooling emission is thermal radiation from\n"
        "shock-heated outer layers, not nickel power.\n\n"
        r"$L_{\rm BB}=4\pi R_{\rm ph}^{2}\sigma_{\rm SB}T_{\rm eff}^{4}$" "\n\n"
        "The dotted curves use SNEC's R_ph and T_eff.\n"
        "The lower panel shows this matches lum_observed.\n\n"
        r"$v\sim(2E/M)^{1/2}$,  $r(t)\sim R_0+vt$" "\n\n"
        r"$t_{\rm diff}\sim\kappa M_{\rm ext}/(cR_{\rm ext})$" "\n\n"
        "Analytic shock-cooling scalings predict luminosity\n"
        "grows with emitting radius, so the 50 R_sun model\n"
        "should be brighter and broader than the 5 R_sun model."
    )
    ax_theory.text(
        0.0,
        1.0,
        theory_text,
        va="top",
        fontsize=11,
        linespacing=1.35,
    )

    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def load_short_profile(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    mass = data[:, 1] / M_SUN
    radius = data[:, 2] / R_SUN
    temperature = data[:, 3]
    density = data[:, 4]
    return mass, radius, temperature, density


def plot_profile_structure() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / "low_ni_profile_structure_context.png"
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True, sharex=True)

    for label, path, color, linestyle, linewidth, zorder in PROFILE_SPECS:
        mass, radius, temperature, density = load_short_profile(path)
        axes[0].plot(
            mass,
            density,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
            label=label,
        )
        axes[1].plot(
            mass,
            temperature,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
            label=label,
        )
        axes[2].plot(
            mass,
            radius,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
            label=label,
        )
        axes[2].scatter(
            mass[-1],
            radius[-1],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=36,
            zorder=zorder + 1,
        )

    axes[0].set_title("Pre-SN density profile")
    axes[0].set_ylabel(r"Density (g cm$^{-3}$)")
    axes[0].set_yscale("log")
    axes[1].set_title("Pre-SN temperature profile")
    axes[1].set_ylabel("Temperature (K)")
    axes[1].set_yscale("log")
    axes[2].set_title("Pre-SN radius profile")
    axes[2].set_xlabel(r"Enclosed mass ($M_\odot$)")
    axes[2].set_ylabel(r"Radius ($R_\odot$)")
    axes[2].set_yscale("log")

    for ax in axes:
        style_paper_axis(ax)
        ax.legend(loc="best", framealpha=0.9)

    axes[0].text(
        0.02,
        0.04,
        "Black dashed curve is the bare WN3 core; added-envelope profiles share this inner structure.",
        transform=axes[0].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "pad": 4},
    )

    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def write_metrics() -> Path:
    output_path = OUT_DIR / "low_ni_diagnostic_metrics.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_hours = [0.25, 0.5, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0]
    records: list[dict[str, float | str]] = []

    for spec in RUNS:
        h_lum, lum_lsun = load_luminosity(spec.data_dir)
        h_teff, teff = load_teff(spec.data_dir)
        h_radius, radius = load_radius(spec.data_dir)
        mask = h_lum <= 48
        peak_idx = int(np.argmax(lum_lsun[mask]))
        record: dict[str, float | str] = {
            "run_id": spec.run_id,
            "label": spec.label,
            "ni_mass_msun": 0.001,
            "energy_foe": 1.0,
            "max_time_hours": float(min(h_lum[-1], h_teff[-1], h_radius[-1])),
            "peak_time_hours_0_48h": float(h_lum[mask][peak_idx]),
            "peak_luminosity_Lsun_0_48h": float(lum_lsun[mask][peak_idx]),
        }
        for hour in sample_hours:
            tag = str(hour).replace(".", "p")
            record[f"luminosity_{tag}h_Lsun"] = interp_at(h_lum, lum_lsun, hour)
            record[f"teff_{tag}h_K"] = interp_at(h_teff, teff, hour)
            record[f"rphoto_{tag}h_Rsun"] = interp_at(h_radius, radius, hour)
        records.append(record)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return output_path


def main() -> None:
    setup_style()
    outputs = [
        plot_luminosity_zoom_windows(),
        plot_three_panel_diagnostics(),
        plot_luminosity_cgs_comparison(),
        plot_presentation_theory_overlay(),
        plot_profile_structure(),
        write_metrics(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
