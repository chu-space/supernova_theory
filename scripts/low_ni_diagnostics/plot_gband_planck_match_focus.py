#!/usr/bin/env python3
"""Focused comparison of SNEC g-band luminosity and time-evolving Planck recovery."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


SNEC_ROOT = Path("/Users/arifchu/Desktop/GitHub/supernova_theory")
RUN_ROOT = SNEC_ROOT / "snec" / "runs" / "model_variants"
OUT_DIR = Path("output/low_ni_diagnostics")

SEC_PER_HOUR = 3600.0
R_SUN = 6.957e10
PC_CM = 3.0856775814913673e18
TEN_PC_CM = 10.0 * PC_CM
C_LIGHT = 2.99792458e10
H_PLANCK = 6.62607015e-27
K_BOLTZ = 1.380649e-16
LAMBDA_G_CM = 4770.0e-8
NU_G_HZ = C_LIGHT / LAMBDA_G_CM


@dataclass(frozen=True)
class Model:
    run_id: str
    label: str
    short_label: str
    color: str


MODELS = [
    Model("wn3_sce_bare_e1_ni001", "Bare WN3", "bare_wn3", "#2b6cb0"),
    Model(
        "wn3_sce_mass_m0p01_r5_e1_ni001",
        r"WN3 + 0.01 $M_\odot$ to 5 $R_\odot$",
        "m0p01_r5",
        "#d97706",
    ),
    Model(
        "wn3_sce_radius_m0p01_r50_e1_ni001",
        r"WN3 + 0.01 $M_\odot$ to 50 $R_\odot$",
        "m0p01_r50",
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


def interp_like(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    return np.interp(x_new, x[order], y[order])


def ab_mag_to_lnu(m_ab: np.ndarray) -> np.ndarray:
    f_nu_10pc = 10.0 ** (-0.4 * (m_ab + 48.60))
    return 4.0 * np.pi * TEN_PC_CM**2 * f_nu_10pc


def lnu_to_ab_mag(l_nu: np.ndarray) -> np.ndarray:
    f_nu_10pc = l_nu / (4.0 * np.pi * TEN_PC_CM**2)
    result = np.full_like(f_nu_10pc, np.nan, dtype=float)
    valid = np.isfinite(f_nu_10pc) & (f_nu_10pc > 0.0)
    result[valid] = -2.5 * np.log10(f_nu_10pc[valid]) - 48.60
    return result


def planck_bnu(nu_hz: float, temperature_k: np.ndarray) -> np.ndarray:
    x = H_PLANCK * nu_hz / (K_BOLTZ * temperature_k)
    return (2.0 * H_PLANCK * nu_hz**3 / C_LIGHT**2) / np.expm1(x)


def load_model(model: Model) -> dict[str, np.ndarray | float | str]:
    data_dir = RUN_ROOT / model.run_id / "Data"
    mag = load_table(data_dir / "magnitudes.dat")
    rad_photo = load_table(data_dir / "rad_photo.dat")

    time_s = mag[:, 0]
    time_h = time_s / SEC_PER_HOUR
    teff = mag[:, 1]
    m_g = mag[:, 4]
    lnu_snec = ab_mag_to_lnu(m_g)
    nu_lnu_snec = NU_G_HZ * lnu_snec

    r_photo_cm = interp_like(time_s, rad_photo[:, 0], rad_photo[:, 1])
    lnu_planck = 4.0 * np.pi**2 * r_photo_cm**2 * planck_bnu(NU_G_HZ, teff)
    nu_lnu_planck = NU_G_HZ * lnu_planck
    m_g_planck = lnu_to_ab_mag(lnu_planck)

    finite = (
        np.isfinite(time_h)
        & np.isfinite(teff)
        & np.isfinite(m_g)
        & np.isfinite(m_g_planck)
        & np.isfinite(nu_lnu_snec)
        & np.isfinite(nu_lnu_planck)
        & (time_h > 0.0)
        & (time_h <= 48.0)
        & (teff > 0.0)
        & (r_photo_cm > 0.0)
        & (nu_lnu_snec > 0.0)
        & (nu_lnu_planck > 0.0)
    )

    time_h = time_h[finite]
    teff = teff[finite]
    m_g = m_g[finite]
    m_g_planck = m_g_planck[finite]
    nu_lnu_snec = nu_lnu_snec[finite]
    nu_lnu_planck = nu_lnu_planck[finite]

    peak_temp_index = int(np.nanargmax(teff))
    cooling_mask = np.arange(len(time_h)) >= peak_temp_index
    ratio = nu_lnu_planck / nu_lnu_snec
    delta_mag = -2.5 * np.log10(ratio)

    return {
        "run_id": model.run_id,
        "label": model.label,
        "short_label": model.short_label,
        "color": model.color,
        "time_h": time_h,
        "teff_K": teff,
        "M_g_snec": m_g,
        "M_g_planck": m_g_planck,
        "nu_Lnu_snec": nu_lnu_snec,
        "nu_Lnu_planck": nu_lnu_planck,
        "ratio": ratio,
        "delta_mag": delta_mag,
        "cooling_mask": cooling_mask,
    }


def model_handles(records: list[dict[str, np.ndarray | float | str]]) -> list[Line2D]:
    return [
        Line2D([0], [0], color=str(record["color"]), lw=2.4, label=str(record["label"]))
        for record in records
    ]


def style_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#111827", lw=2.2, label="SNEC g-band output"),
        Line2D([0], [0], color="#111827", lw=2.2, ls=":", label="Planck recovery from R_ph(t), T_eff(t)"),
    ]


def make_plot(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)
    ax_mag, ax_ratio = axes

    for record in records:
        mask = record["cooling_mask"]
        assert isinstance(mask, np.ndarray)
        time_h = record["time_h"][mask]
        color = str(record["color"])

        ax_mag.plot(time_h, record["M_g_snec"][mask], color=color, lw=2.2)
        ax_mag.plot(time_h, record["M_g_planck"][mask], color=color, lw=2.2, ls=":")
        ax_ratio.plot(time_h, record["ratio"][mask], color=color, lw=2.0)

    ax_mag.set_xscale("log")
    ax_mag.set_xlim(0.01, 48.0)
    ax_mag.invert_yaxis()
    ax_mag.set_xlabel("Time since explosion (hours)")
    ax_mag.set_ylabel(r"Absolute $g$ magnitude, $M_g$")
    ax_mag.set_title(r"Time-evolving Planck recovery in absolute magnitude")
    style_paper_axis(ax_mag)
    leg1 = ax_mag.legend(handles=model_handles(records), fontsize=8, loc="upper right")
    ax_mag.add_artist(leg1)
    ax_mag.legend(handles=style_handles(), fontsize=8, loc="lower left")

    ax_ratio.axhline(1.0, color="#111827", lw=1.0, alpha=0.75)
    ax_ratio.axhspan(0.8, 1.2, color="#9ca3af", alpha=0.15, lw=0)
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlim(0.01, 48.0)
    ax_ratio.set_ylim(0.45, 1.75)
    ax_ratio.set_xlabel("Time since explosion (hours)")
    ax_ratio.set_ylabel(r"Planck recovery / SNEC $g$ luminosity")
    ax_ratio.set_title("Agreement band")
    style_paper_axis(ax_ratio)

    fig.suptitle(
        r"SNEC $g$ magnitude vs time-dependent Planck recovery",
        fontsize=13,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_gband_magnitude_match_ratio.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def make_bright_phase_plot(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), constrained_layout=True)
    ax_ratio, ax_delta = axes

    for record in records:
        mask = record["cooling_mask"]
        assert isinstance(mask, np.ndarray)
        snec = record["nu_Lnu_snec"]
        bright = mask & (snec >= 0.10 * np.nanmax(snec[mask]))
        time_h = record["time_h"][bright]
        color = str(record["color"])
        ax_ratio.plot(time_h, record["ratio"][bright], color=color, lw=2.0, label=str(record["label"]))
        ax_delta.plot(time_h, record["delta_mag"][bright], color=color, lw=2.0, label=str(record["label"]))

    ax_ratio.axhline(1.0, color="#111827", lw=1.0, alpha=0.75)
    ax_ratio.axhspan(0.8, 1.2, color="#9ca3af", alpha=0.15, lw=0)
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlim(0.03, 48.0)
    ax_ratio.set_ylim(0.75, 1.25)
    ax_ratio.set_xlabel("Time since explosion (hours)")
    ax_ratio.set_ylabel(r"Planck recovery / SNEC $g$ luminosity")
    ax_ratio.set_title(r"Bright SCE phase: luminosity ratio")
    style_paper_axis(ax_ratio)
    ax_ratio.legend(fontsize=8)

    ax_delta.axhline(0.0, color="#111827", lw=1.0, alpha=0.75)
    ax_delta.axhspan(-0.2, 0.2, color="#9ca3af", alpha=0.15, lw=0)
    ax_delta.set_xscale("log")
    ax_delta.set_xlim(0.03, 48.0)
    ax_delta.set_ylim(-0.25, 0.25)
    ax_delta.set_xlabel("Time since explosion (hours)")
    ax_delta.set_ylabel(r"$\Delta M_g = M_{g,\rm Planck} - M_{g,\rm SNEC}$")
    ax_delta.set_title(r"Bright SCE phase: magnitude residual")
    style_paper_axis(ax_delta)

    fig.suptitle(
        r"Time-dependent Planck recovery during the bright shock-cooling phase",
        fontsize=13,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_gband_planck_match_bright_phase.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def write_metrics(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    path = OUT_DIR / "low_ni_gband_planck_match_metrics.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "label",
        "median_ratio",
        "p16_ratio",
        "p84_ratio",
        "median_delta_mag",
        "fraction_within_10_percent",
        "fraction_within_20_percent",
        "fraction_within_0p1_mag",
        "fraction_within_0p2_mag",
        "bright_phase_start_h",
        "bright_phase_end_h",
        "bright_median_ratio",
        "bright_p16_ratio",
        "bright_p84_ratio",
        "bright_fraction_within_20_percent",
        "bright_median_delta_mag",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            mask = record["cooling_mask"]
            assert isinstance(mask, np.ndarray)
            ratio = record["ratio"][mask]
            delta_mag = record["delta_mag"][mask]
            snec = record["nu_Lnu_snec"]
            bright = mask & (snec >= 0.10 * np.nanmax(snec[mask]))
            bright_ratio = record["ratio"][bright]
            bright_delta_mag = record["delta_mag"][bright]
            bright_time_h = record["time_h"][bright]
            writer.writerow(
                {
                    "run_id": record["run_id"],
                    "label": record["short_label"],
                    "median_ratio": f"{float(np.nanmedian(ratio)):.6g}",
                    "p16_ratio": f"{float(np.nanpercentile(ratio, 16)):.6g}",
                    "p84_ratio": f"{float(np.nanpercentile(ratio, 84)):.6g}",
                    "median_delta_mag": f"{float(np.nanmedian(delta_mag)):.6g}",
                    "fraction_within_10_percent": f"{float(np.nanmean(np.abs(ratio - 1.0) <= 0.10)):.6g}",
                    "fraction_within_20_percent": f"{float(np.nanmean(np.abs(ratio - 1.0) <= 0.20)):.6g}",
                    "fraction_within_0p1_mag": f"{float(np.nanmean(np.abs(delta_mag) <= 0.1)):.6g}",
                    "fraction_within_0p2_mag": f"{float(np.nanmean(np.abs(delta_mag) <= 0.2)):.6g}",
                    "bright_phase_start_h": f"{float(bright_time_h[0]):.6g}",
                    "bright_phase_end_h": f"{float(bright_time_h[-1]):.6g}",
                    "bright_median_ratio": f"{float(np.nanmedian(bright_ratio)):.6g}",
                    "bright_p16_ratio": f"{float(np.nanpercentile(bright_ratio, 16)):.6g}",
                    "bright_p84_ratio": f"{float(np.nanpercentile(bright_ratio, 84)):.6g}",
                    "bright_fraction_within_20_percent": f"{float(np.nanmean(np.abs(bright_ratio - 1.0) <= 0.20)):.6g}",
                    "bright_median_delta_mag": f"{float(np.nanmedian(bright_delta_mag)):.6g}",
                }
            )
    return path


def main() -> None:
    records = [load_model(model) for model in MODELS]
    for output in [make_plot(records), make_bright_phase_plot(records), write_metrics(records)]:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
