#!/usr/bin/env python3
"""Plot low-Ni SNEC g-band diagnostics against temperature and radius."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SNEC_ROOT = Path("/Users/arifchu/Desktop/GitHub/supernova_theory")
RUN_ROOT = SNEC_ROOT / "snec" / "runs" / "model_variants"
OUT_DIR = Path("output/low_ni_diagnostics")

SEC_PER_HOUR = 3600.0
R_SUN = 6.957e10
PC_CM = 3.0856775814913673e18
TEN_PC_CM = 10.0 * PC_CM
C_LIGHT = 2.99792458e10
LAMBDA_G_CM = 4770.0e-8
NU_G_HZ = C_LIGHT / LAMBDA_G_CM


@dataclass(frozen=True)
class Model:
    run_id: str
    label: str
    short_label: str
    color: str


MODELS = [
    Model(
        "wn3_sce_bare_e1_ni001",
        "Bare WN3",
        "Bare WN3",
        "#2b6cb0",
    ),
    Model(
        "wn3_sce_mass_m0p01_r5_e1_ni001",
        r"WN3 + 0.01 $M_\odot$ to 5 $R_\odot$",
        "0.01 Msun to 5 Rsun",
        "#d97706",
    ),
    Model(
        "wn3_sce_radius_m0p01_r50_e1_ni001",
        r"WN3 + 0.01 $M_\odot$ to 50 $R_\odot$",
        "0.01 Msun to 50 Rsun",
        "#15803d",
    ),
]


def style_paper_axis(ax) -> None:
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=3)


def load_table(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, data.shape[0])
    return data


def load_two_column(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = load_table(path)
    keep = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    return data[keep, 0], data[keep, 1]


def interp_like(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    return np.interp(x_new, x[order], y[order])


def ab_mag_to_nu_lnu(m_ab: np.ndarray) -> np.ndarray:
    """Convert absolute AB magnitude to nu L_nu in erg/s."""
    f_nu_10pc = 10.0 ** (-0.4 * (m_ab + 48.60))
    l_nu = 4.0 * np.pi * TEN_PC_CM**2 * f_nu_10pc
    return NU_G_HZ * l_nu


def load_model(model: Model) -> dict[str, np.ndarray | float | str]:
    data_dir = RUN_ROOT / model.run_id / "Data"
    mag = load_table(data_dir / "magnitudes.dat")
    t_s = mag[:, 0]
    time_h = t_s / SEC_PER_HOUR
    teff_for_bc = mag[:, 1]
    # magnitudes.dat columns:
    # time, T_eff_for_BC, PTF_R_AB, u, g, r, i, z, U, B, V, R, I
    m_g = mag[:, 4]
    nu_lnu_g = ab_mag_to_nu_lnu(m_g)

    t_r, r_photo_cm = load_two_column(data_dir / "rad_photo.dat")
    r_photo_at_mag_rsun = interp_like(t_s, t_r, r_photo_cm) / R_SUN

    rad_initial = load_table(data_dir / "rad_initial.dat")
    if rad_initial.shape[1] > 1:
        r_star_rsun = float(np.nanmax(rad_initial[:, 1]) / R_SUN)
    else:
        r_star_rsun = float(np.nanmax(rad_initial[:, 0]) / R_SUN)

    finite = (
        np.isfinite(time_h)
        & np.isfinite(teff_for_bc)
        & np.isfinite(m_g)
        & np.isfinite(nu_lnu_g)
        & np.isfinite(r_photo_at_mag_rsun)
        & (time_h > 0.0)
        & (time_h <= 48.0)
        & (teff_for_bc > 0.0)
        & (r_photo_at_mag_rsun > 0.0)
    )

    time_h = time_h[finite]
    teff_for_bc = teff_for_bc[finite]
    m_g = m_g[finite]
    nu_lnu_g = nu_lnu_g[finite]
    r_photo_at_mag_rsun = r_photo_at_mag_rsun[finite]

    # The extended-envelope runs contain pre-breakout cool placeholder points.
    # Use the cooling branch after the first hot maximum for model comparison.
    peak_temp_index = int(np.nanargmax(teff_for_bc))
    cooling = np.arange(len(time_h)) >= peak_temp_index

    peak_g_index = int(np.nanargmin(m_g[cooling]))
    cooling_indices = np.where(cooling)[0]
    peak_g_index = int(cooling_indices[peak_g_index])

    return {
        "run_id": model.run_id,
        "label": model.label,
        "short_label": model.short_label,
        "color": model.color,
        "time_h": time_h,
        "teff_K": teff_for_bc,
        "M_g": m_g,
        "nu_Lnu_g": nu_lnu_g,
        "R_ph_Rsun": r_photo_at_mag_rsun,
        "cooling_mask": cooling,
        "R_star_Rsun": r_star_rsun,
        "peak_M_g": float(m_g[peak_g_index]),
        "peak_nu_Lnu_g": float(nu_lnu_g[peak_g_index]),
        "peak_time_h": float(time_h[peak_g_index]),
        "peak_Teff_K": float(teff_for_bc[peak_g_index]),
        "peak_R_ph_Rsun": float(r_photo_at_mag_rsun[peak_g_index]),
    }


def plot_gband_vs_temperature(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), constrained_layout=True)
    ax_mag, ax_lum = axes

    for record in records:
        mask = record["cooling_mask"]
        assert isinstance(mask, np.ndarray)
        teff = record["teff_K"][mask]
        m_g = record["M_g"][mask]
        nu_lnu = record["nu_Lnu_g"][mask]
        time_h = record["time_h"][mask]
        color = str(record["color"])
        label = str(record["label"])

        order = np.argsort(teff)
        ax_mag.plot(teff[order], m_g[order], color=color, lw=2.0, label=label)
        ax_lum.plot(teff[order], nu_lnu[order], color=color, lw=2.0, label=label)

        # Mark the first point on the cooling branch and the 6-hour point when present.
        ax_mag.scatter(teff[0], m_g[0], color=color, edgecolor="black", s=36, zorder=3)
        ax_lum.scatter(teff[0], nu_lnu[0], color=color, edgecolor="black", s=36, zorder=3)
        if np.nanmin(time_h) <= 6.0 <= np.nanmax(time_h):
            teff_6 = np.interp(6.0, time_h, teff)
            m_g_6 = np.interp(6.0, time_h, m_g)
            lum_6 = np.interp(6.0, time_h, nu_lnu)
            ax_mag.scatter(teff_6, m_g_6, marker="s", color=color, edgecolor="black", s=32, zorder=3)
            ax_lum.scatter(teff_6, lum_6, marker="s", color=color, edgecolor="black", s=32, zorder=3)

    ax_mag.set_xscale("log")
    ax_mag.invert_yaxis()
    ax_mag.set_xlabel(r"$T_{\rm eff}$ used for SNEC bolometric correction (K)")
    ax_mag.set_ylabel(r"Absolute $g$ magnitude, $M_g$")
    ax_mag.set_title(r"Low-Ni $g$ Band vs Temperature")
    style_paper_axis(ax_mag)
    ax_mag.legend(fontsize=8)

    ax_lum.set_xscale("log")
    ax_lum.set_yscale("log")
    ax_lum.set_xlabel(r"$T_{\rm eff}$ used for SNEC bolometric correction (K)")
    ax_lum.set_ylabel(r"$\nu_g L_{\nu,g}$ from $M_g$ (erg s$^{-1}$)")
    ax_lum.set_title(r"AB $g$ Magnitude Converted to Band Luminosity")
    style_paper_axis(ax_lum)

    fig.suptitle(
        r"Low-Ni WN3 Models: SNEC $g$ Band Along the Post-Breakout Cooling Branch",
        fontsize=14,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_gband_vs_temperature.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def plot_radius_vs_gband(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), constrained_layout=True)
    ax_star, ax_photo = axes

    r_star = np.array([float(record["R_star_Rsun"]) for record in records])
    peak_mg = np.array([float(record["peak_M_g"]) for record in records])
    peak_lum = np.array([float(record["peak_nu_Lnu_g"]) for record in records])
    colors = [str(record["color"]) for record in records]
    labels = [str(record["short_label"]) for record in records]

    order = np.argsort(r_star)
    ax_star.plot(r_star[order], peak_mg[order], color="#111827", lw=1.3, alpha=0.65)
    for idx in order:
        ax_star.scatter(r_star[idx], peak_mg[idx], s=90, color=colors[idx], edgecolor="black", zorder=3)
        if r_star[idx] == np.nanmax(r_star):
            offset = (-120, 8)
        else:
            offset = (7, 7)
        ax_star.annotate(
            labels[idx],
            (r_star[idx], peak_mg[idx]),
            textcoords="offset points",
            xytext=offset,
            fontsize=8.5,
        )

    ax_star.set_xscale("log")
    ax_star.set_xlim(np.nanmin(r_star) * 0.65, np.nanmax(r_star) * 1.6)
    ax_star.invert_yaxis()
    ax_star.set_xlabel(r"Model outer radius, $R_\star$ ($R_\odot$)")
    ax_star.set_ylabel(r"Peak absolute $g$ magnitude over 0-48 h")
    ax_star.set_title(r"Fixed Progenitor/Envelope Radius vs Peak $g$")
    style_paper_axis(ax_star)

    twin = ax_star.twinx()
    twin.plot(r_star[order], peak_lum[order], ":", color="#4b5563", lw=1.6)
    twin.scatter(r_star, peak_lum, s=26, color=colors, edgecolor="black", zorder=4)
    twin.set_yscale("log")
    twin.set_ylabel(r"Peak $\nu_g L_{\nu,g}$ (erg s$^{-1}$)")
    style_paper_axis(twin)

    for record in records:
        mask = record["cooling_mask"]
        assert isinstance(mask, np.ndarray)
        r_photo = record["R_ph_Rsun"][mask]
        m_g = record["M_g"][mask]
        color = str(record["color"])
        label = str(record["label"])
        ax_photo.plot(r_photo, m_g, color=color, lw=1.9, label=label)
        ax_photo.scatter(r_photo[0], m_g[0], color=color, edgecolor="black", s=34, zorder=3)

    ax_photo.set_xscale("log")
    ax_photo.invert_yaxis()
    ax_photo.set_xlabel(r"SNEC photospheric radius, $R_{\rm ph}(t)$ ($R_\odot$)")
    ax_photo.set_ylabel(r"Absolute $g$ magnitude, $M_g(t)$")
    ax_photo.set_title(r"Time-Evolving Radiating Surface vs $g$")
    style_paper_axis(ax_photo)
    ax_photo.legend(fontsize=8)

    fig.suptitle(
        r"Low-Ni WN3 Models: Radius Controls Early Optical Brightness",
        fontsize=14,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_radius_vs_gband.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def write_metrics(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    path = OUT_DIR / "low_ni_gband_temperature_radius_metrics.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "label",
        "R_star_Rsun",
        "peak_time_h",
        "peak_Teff_K",
        "peak_R_ph_Rsun",
        "peak_M_g",
        "peak_nu_Lnu_g_erg_s",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "run_id": record["run_id"],
                    "label": record["short_label"],
                    "R_star_Rsun": f"{float(record['R_star_Rsun']):.6g}",
                    "peak_time_h": f"{float(record['peak_time_h']):.6g}",
                    "peak_Teff_K": f"{float(record['peak_Teff_K']):.6g}",
                    "peak_R_ph_Rsun": f"{float(record['peak_R_ph_Rsun']):.6g}",
                    "peak_M_g": f"{float(record['peak_M_g']):.6g}",
                    "peak_nu_Lnu_g_erg_s": f"{float(record['peak_nu_Lnu_g']):.6e}",
                }
            )
    return path


def main() -> None:
    records = [load_model(model) for model in MODELS]
    temp_plot = plot_gband_vs_temperature(records)
    radius_plot = plot_radius_vs_gband(records)
    metrics = write_metrics(records)
    print(f"Saved plot to {temp_plot}")
    print(f"Saved plot to {radius_plot}")
    print(f"Saved metrics to {metrics}")


if __name__ == "__main__":
    main()
