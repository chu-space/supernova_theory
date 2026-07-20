#!/usr/bin/env python3
"""Compare SNEC g-band light curves with time-dependent blackbody estimates.

The earlier board scaling is useful for intuition, but it becomes quantitative
only after the emitting radius and temperature are allowed to evolve. This
script uses SNEC's photospheric radius and the effective temperature used for
bolometric corrections at every output time.
"""

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
SIGMA_SB = 5.670374419e-5
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
        "bare_wn3",
        "#2b6cb0",
    ),
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
    """Convert absolute AB magnitude to L_nu in erg/s/Hz."""
    f_nu_10pc = 10.0 ** (-0.4 * (m_ab + 48.60))
    return 4.0 * np.pi * TEN_PC_CM**2 * f_nu_10pc


def lnu_to_ab_mag(l_nu: np.ndarray) -> np.ndarray:
    """Convert L_nu in erg/s/Hz to absolute AB magnitude."""
    f_nu_10pc = l_nu / (4.0 * np.pi * TEN_PC_CM**2)
    result = np.full_like(f_nu_10pc, np.nan, dtype=float)
    valid = np.isfinite(f_nu_10pc) & (f_nu_10pc > 0.0)
    result[valid] = -2.5 * np.log10(f_nu_10pc[valid]) - 48.60
    return result


def planck_bnu(nu_hz: float, temperature_k: np.ndarray) -> np.ndarray:
    x = H_PLANCK * nu_hz / (K_BOLTZ * temperature_k)
    return (2.0 * H_PLANCK * nu_hz**3 / C_LIGHT**2) / np.expm1(x)


def rayleigh_jeans_bnu(nu_hz: float, temperature_k: np.ndarray) -> np.ndarray:
    return 2.0 * nu_hz**2 * K_BOLTZ * temperature_k / C_LIGHT**2


def spectral_luminosity_from_bnu(radius_cm: np.ndarray, bnu: np.ndarray) -> np.ndarray:
    """Return L_nu for a spherical blackbody emitter."""
    return 4.0 * np.pi**2 * radius_cm**2 * bnu


def load_model(model: Model) -> dict[str, np.ndarray | float | str]:
    data_dir = RUN_ROOT / model.run_id / "Data"
    mag = load_table(data_dir / "magnitudes.dat")
    rad_photo = load_table(data_dir / "rad_photo.dat")
    lum_photo = load_table(data_dir / "lum_photo.dat")
    rad_initial = load_table(data_dir / "rad_initial.dat")

    time_s = mag[:, 0]
    time_h = time_s / SEC_PER_HOUR
    teff_for_bc = mag[:, 1]
    # magnitudes.dat columns:
    # time, T_eff_for_BC, PTF_R_AB, u, g, r, i, z, U, B, V, R, I
    m_g_snec = mag[:, 4]
    lnu_g_snec = ab_mag_to_lnu(m_g_snec)
    nu_lnu_g_snec = NU_G_HZ * lnu_g_snec

    r_photo_cm = interp_like(time_s, rad_photo[:, 0], rad_photo[:, 1])
    l_photo = interp_like(time_s, lum_photo[:, 0], lum_photo[:, 1])
    t_sb = (l_photo / (4.0 * np.pi * SIGMA_SB * np.maximum(r_photo_cm, 1.0) ** 2)) ** 0.25

    bnu_planck = planck_bnu(NU_G_HZ, teff_for_bc)
    bnu_rj = rayleigh_jeans_bnu(NU_G_HZ, teff_for_bc)
    lnu_planck = spectral_luminosity_from_bnu(r_photo_cm, bnu_planck)
    lnu_rj = spectral_luminosity_from_bnu(r_photo_cm, bnu_rj)
    nu_lnu_planck = NU_G_HZ * lnu_planck
    nu_lnu_rj = NU_G_HZ * lnu_rj
    m_g_planck = lnu_to_ab_mag(lnu_planck)
    m_g_rj = lnu_to_ab_mag(lnu_rj)

    if rad_initial.shape[1] > 1:
        r_star_rsun = float(np.nanmax(rad_initial[:, 1]) / R_SUN)
    else:
        r_star_rsun = float(np.nanmax(rad_initial[:, 0]) / R_SUN)

    finite = (
        np.isfinite(time_h)
        & np.isfinite(teff_for_bc)
        & np.isfinite(t_sb)
        & np.isfinite(m_g_snec)
        & np.isfinite(m_g_planck)
        & np.isfinite(m_g_rj)
        & np.isfinite(nu_lnu_g_snec)
        & np.isfinite(nu_lnu_planck)
        & np.isfinite(nu_lnu_rj)
        & np.isfinite(r_photo_cm)
        & (time_h > 0.0)
        & (time_h <= 48.0)
        & (teff_for_bc > 0.0)
        & (r_photo_cm > 0.0)
        & (nu_lnu_g_snec > 0.0)
        & (nu_lnu_planck > 0.0)
        & (nu_lnu_rj > 0.0)
    )

    time_h = time_h[finite]
    teff_for_bc = teff_for_bc[finite]
    t_sb = t_sb[finite]
    m_g_snec = m_g_snec[finite]
    m_g_planck = m_g_planck[finite]
    m_g_rj = m_g_rj[finite]
    nu_lnu_g_snec = nu_lnu_g_snec[finite]
    nu_lnu_planck = nu_lnu_planck[finite]
    nu_lnu_rj = nu_lnu_rj[finite]
    r_photo_cm = r_photo_cm[finite]

    # The extended runs include cool placeholder points before the breakout.
    # The cooling branch starts at the first hot temperature maximum.
    peak_temp_index = int(np.nanargmax(teff_for_bc))
    cooling_mask = np.arange(len(time_h)) >= peak_temp_index

    peak_g_local = int(np.nanargmin(m_g_snec[cooling_mask]))
    cooling_indices = np.where(cooling_mask)[0]
    peak_g_index = int(cooling_indices[peak_g_local])

    x_g = H_PLANCK * NU_G_HZ / (K_BOLTZ * teff_for_bc)
    delta_planck = m_g_planck - m_g_snec
    delta_rj = m_g_rj - m_g_snec

    return {
        "run_id": model.run_id,
        "label": model.label,
        "short_label": model.short_label,
        "color": model.color,
        "time_h": time_h,
        "teff_K": teff_for_bc,
        "teff_sb_K": t_sb,
        "x_g": x_g,
        "R_ph_Rsun": r_photo_cm / R_SUN,
        "M_g_snec": m_g_snec,
        "M_g_planck": m_g_planck,
        "M_g_rj": m_g_rj,
        "nu_Lnu_snec": nu_lnu_g_snec,
        "nu_Lnu_planck": nu_lnu_planck,
        "nu_Lnu_rj": nu_lnu_rj,
        "delta_planck_mag": delta_planck,
        "delta_rj_mag": delta_rj,
        "cooling_mask": cooling_mask,
        "R_star_Rsun": r_star_rsun,
        "peak_time_h": float(time_h[peak_g_index]),
        "peak_M_g_snec": float(m_g_snec[peak_g_index]),
        "peak_M_g_planck": float(m_g_planck[peak_g_index]),
        "peak_M_g_rj": float(m_g_rj[peak_g_index]),
    }


def style_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#111827", lw=2.0, label="SNEC BolCorr"),
        Line2D([0], [0], color="#111827", lw=2.0, ls=":", label="Planck BB with R_ph(t), T_eff(t)"),
        Line2D([0], [0], color="#111827", lw=1.6, ls="-.", alpha=0.65, label="Rayleigh-Jeans limit"),
    ]


def model_handles(records: list[dict[str, np.ndarray | float | str]]) -> list[Line2D]:
    return [
        Line2D([0], [0], color=str(record["color"]), lw=2.4, label=str(record["label"]))
        for record in records
    ]


def plot_time_dependent_magnitude(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)
    ax_mag, ax_delta = axes

    for record in records:
        mask = record["cooling_mask"]
        assert isinstance(mask, np.ndarray)
        time_h = record["time_h"][mask]
        color = str(record["color"])

        ax_mag.plot(time_h, record["M_g_snec"][mask], color=color, lw=2.1)
        ax_mag.plot(time_h, record["M_g_planck"][mask], color=color, lw=2.0, ls=":")
        ax_mag.plot(time_h, record["M_g_rj"][mask], color=color, lw=1.6, ls="-.", alpha=0.65)

        ax_delta.plot(time_h, record["delta_planck_mag"][mask], color=color, lw=2.0, ls=":")
        ax_delta.plot(time_h, record["delta_rj_mag"][mask], color=color, lw=1.6, ls="-.", alpha=0.65)

    ax_mag.set_xscale("log")
    ax_mag.set_xlim(0.01, 48.0)
    ax_mag.invert_yaxis()
    ax_mag.set_xlabel("Time since explosion (hours)")
    ax_mag.set_ylabel(r"Absolute $g$ magnitude, $M_g$")
    ax_mag.set_title(r"Time-dependent $g$-band magnitude")
    style_paper_axis(ax_mag)
    leg1 = ax_mag.legend(handles=model_handles(records), fontsize=8, loc="upper right")
    ax_mag.add_artist(leg1)
    ax_mag.legend(handles=style_handles(), fontsize=8, loc="lower left")

    ax_delta.axhline(0.0, color="#111827", lw=1.0, alpha=0.7)
    ax_delta.set_xscale("log")
    ax_delta.set_xlim(0.01, 48.0)
    ax_delta.set_xlabel("Time since explosion (hours)")
    ax_delta.set_ylabel(r"$\Delta M_g = M_{g,\rm model} - M_{g,\rm SNEC}$")
    ax_delta.set_title("Difference from SNEC bolometric correction")
    style_paper_axis(ax_delta)

    fig.suptitle(
        r"Low-Ni WN3 Models: Corrected $g$-band Conversion Uses $R_{\rm ph}(t)$ and $T_{\rm eff}(t)$",
        fontsize=13,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_time_dependent_gband_magnitude.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def plot_time_dependent_luminosity(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)
    ax_lum, ax_ratio = axes

    for record in records:
        mask = record["cooling_mask"]
        assert isinstance(mask, np.ndarray)
        time_h = record["time_h"][mask]
        color = str(record["color"])
        snec = record["nu_Lnu_snec"][mask]
        planck = record["nu_Lnu_planck"][mask]
        rj = record["nu_Lnu_rj"][mask]

        ax_lum.plot(time_h, snec, color=color, lw=2.1)
        ax_lum.plot(time_h, planck, color=color, lw=2.0, ls=":")
        ax_lum.plot(time_h, rj, color=color, lw=1.6, ls="-.", alpha=0.65)

        ax_ratio.plot(time_h, planck / snec, color=color, lw=2.0, ls=":")
        ax_ratio.plot(time_h, rj / snec, color=color, lw=1.6, ls="-.", alpha=0.65)

    ax_lum.set_xscale("log")
    ax_lum.set_yscale("log")
    ax_lum.set_xlim(0.01, 48.0)
    ax_lum.set_xlabel("Time since explosion (hours)")
    ax_lum.set_ylabel(r"$\nu_g L_{\nu,g}$ (erg s$^{-1}$)")
    ax_lum.set_title(r"Band luminosity from $g$ magnitude and blackbody theory")
    style_paper_axis(ax_lum)
    leg1 = ax_lum.legend(handles=model_handles(records), fontsize=8, loc="lower right")
    ax_lum.add_artist(leg1)
    ax_lum.legend(handles=style_handles(), fontsize=8, loc="upper left")

    ax_ratio.axhline(1.0, color="#111827", lw=1.0, alpha=0.7)
    ax_ratio.set_xscale("log")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlim(0.01, 48.0)
    ax_ratio.set_xlabel("Time since explosion (hours)")
    ax_ratio.set_ylabel(r"Model / SNEC $\nu_g L_{\nu,g}$")
    ax_ratio.set_title("Luminosity-space residual")
    style_paper_axis(ax_ratio)

    fig.suptitle(
        r"Time-dependent $g$-band luminosity: full Planck vs Rayleigh-Jeans",
        fontsize=13,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_time_dependent_gband_luminosity.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def plot_early_zoom(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)
    ax_mag, ax_lum = axes

    for record in records:
        mask = record["cooling_mask"] & (record["time_h"] <= 6.0)
        assert isinstance(mask, np.ndarray)
        if not np.any(mask):
            continue
        time_h = record["time_h"][mask]
        color = str(record["color"])

        ax_mag.plot(time_h, record["M_g_snec"][mask], color=color, lw=2.1)
        ax_mag.plot(time_h, record["M_g_planck"][mask], color=color, lw=2.0, ls=":")
        ax_mag.plot(time_h, record["M_g_rj"][mask], color=color, lw=1.6, ls="-.", alpha=0.65)

        ax_lum.plot(time_h, record["nu_Lnu_snec"][mask], color=color, lw=2.1)
        ax_lum.plot(time_h, record["nu_Lnu_planck"][mask], color=color, lw=2.0, ls=":")
        ax_lum.plot(time_h, record["nu_Lnu_rj"][mask], color=color, lw=1.6, ls="-.", alpha=0.65)

    ax_mag.set_xlim(0.0, 6.0)
    ax_mag.invert_yaxis()
    ax_mag.set_xlabel("Time since explosion (hours)")
    ax_mag.set_ylabel(r"Absolute $g$ magnitude, $M_g$")
    ax_mag.set_title("First six hours in magnitude space")
    style_paper_axis(ax_mag)
    leg1 = ax_mag.legend(handles=model_handles(records), fontsize=8, loc="upper right")
    ax_mag.add_artist(leg1)
    ax_mag.legend(handles=style_handles(), fontsize=8, loc="lower left")

    ax_lum.set_xlim(0.0, 6.0)
    ax_lum.set_yscale("log")
    ax_lum.set_xlabel("Time since explosion (hours)")
    ax_lum.set_ylabel(r"$\nu_g L_{\nu,g}$ (erg s$^{-1}$)")
    ax_lum.set_title("First six hours in luminosity space")
    style_paper_axis(ax_lum)

    fig.suptitle(r"Early-time zoom: corrected time-dependent $g$-band model", fontsize=13)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_time_dependent_gband_early_zoom.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def first_time_abs_delta_exceeds(
    time_h: np.ndarray, delta_mag: np.ndarray, threshold_mag: float
) -> float:
    exceeds = np.where(np.abs(delta_mag) > threshold_mag)[0]
    if len(exceeds) == 0:
        return float("nan")
    return float(time_h[exceeds[0]])


def last_time_rj_small_x(time_h: np.ndarray, x_g: np.ndarray, max_x: float = 0.3) -> float:
    good = np.where(x_g < max_x)[0]
    if len(good) == 0:
        return float("nan")
    return float(time_h[good[-1]])


def write_metrics(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    path = OUT_DIR / "low_ni_time_dependent_gband_metrics.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "label",
        "R_star_Rsun",
        "post_breakout_start_h",
        "peak_time_h",
        "peak_M_g_snec",
        "peak_M_g_planck",
        "peak_M_g_rayleigh_jeans",
        "median_delta_planck_mag",
        "median_delta_rayleigh_jeans_mag",
        "median_planck_over_snec_nuLnu",
        "median_rayleigh_jeans_over_snec_nuLnu",
        "last_time_hnu_over_kT_lt_0p3_h",
        "first_time_rayleigh_jeans_delta_gt_0p5mag_h",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            mask = record["cooling_mask"]
            assert isinstance(mask, np.ndarray)
            time_h = record["time_h"][mask]
            snec = record["nu_Lnu_snec"][mask]
            planck = record["nu_Lnu_planck"][mask]
            rj = record["nu_Lnu_rj"][mask]
            delta_planck = record["delta_planck_mag"][mask]
            delta_rj = record["delta_rj_mag"][mask]
            x_g = record["x_g"][mask]
            writer.writerow(
                {
                    "run_id": record["run_id"],
                    "label": record["short_label"],
                    "R_star_Rsun": f"{float(record['R_star_Rsun']):.6g}",
                    "post_breakout_start_h": f"{float(time_h[0]):.6g}",
                    "peak_time_h": f"{float(record['peak_time_h']):.6g}",
                    "peak_M_g_snec": f"{float(record['peak_M_g_snec']):.6g}",
                    "peak_M_g_planck": f"{float(record['peak_M_g_planck']):.6g}",
                    "peak_M_g_rayleigh_jeans": f"{float(record['peak_M_g_rj']):.6g}",
                    "median_delta_planck_mag": f"{float(np.nanmedian(delta_planck)):.6g}",
                    "median_delta_rayleigh_jeans_mag": f"{float(np.nanmedian(delta_rj)):.6g}",
                    "median_planck_over_snec_nuLnu": f"{float(np.nanmedian(planck / snec)):.6g}",
                    "median_rayleigh_jeans_over_snec_nuLnu": f"{float(np.nanmedian(rj / snec)):.6g}",
                    "last_time_hnu_over_kT_lt_0p3_h": f"{last_time_rj_small_x(time_h, x_g):.6g}",
                    "first_time_rayleigh_jeans_delta_gt_0p5mag_h": f"{first_time_abs_delta_exceeds(time_h, delta_rj, 0.5):.6g}",
                }
            )
    return path


def main() -> None:
    records = [load_model(model) for model in MODELS]
    outputs = [
        plot_time_dependent_magnitude(records),
        plot_time_dependent_luminosity(records),
        plot_early_zoom(records),
        write_metrics(records),
    ]
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
