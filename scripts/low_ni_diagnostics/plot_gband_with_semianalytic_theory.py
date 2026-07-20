#!/usr/bin/env python3
"""Add a semi-analytic shock-cooling g-band curve to the low-Ni WN3 plots."""

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
SEC_PER_MIN = 60.0
R_SUN = 6.957e10
M_SUN = 1.98847e33
FOE = 1.0e51
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


def lnu_from_blackbody(radius_cm: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    return 4.0 * np.pi**2 * radius_cm**2 * planck_bnu(NU_G_HZ, temperature_k)


def nakar_wr_bolometric_luminosity(
    time_s: np.ndarray,
    mass_msun: float,
    radius_rsun: float,
    explosion_energy_foe: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Approximate WR shock-cooling luminosity from Nakar & Sari 2010.

    This uses the WR-like compact-progenitor scaling: L ~ t^-4/3 before the
    planar/spherical transition and L ~ t^-0.35 afterward. It is a comparison
    curve, not a substitute for SNEC.
    """
    safe_t = np.maximum(time_s, 1.0)
    m15 = mass_msun / 15.0
    r5 = max(radius_rsun / 5.0, 1.0e-3)
    e51 = explosion_energy_foe
    t_s = 90.0 * m15**0.41 * r5**1.33 * e51**-0.58

    t_min = safe_t / SEC_PER_MIN
    t_hr = safe_t / SEC_PER_HOUR
    l_planar = 2.0e42 * m15**-0.33 * r5**2.3 * e51**0.34 * t_min**(-4.0 / 3.0)
    l_spherical = 3.5e41 * m15**-0.73 * r5 * e51**0.91 * t_hr**-0.35
    return np.where(safe_t < t_s, l_planar, l_spherical), t_s


def load_model(model: Model) -> dict[str, np.ndarray | float | str]:
    data_dir = RUN_ROOT / model.run_id / "Data"
    mag = load_table(data_dir / "magnitudes.dat")
    rad_photo = load_table(data_dir / "rad_photo.dat")
    rad_initial = load_table(data_dir / "rad_initial.dat")
    mass_initial = load_table(data_dir / "mass_initial.dat")

    time_s = mag[:, 0]
    time_h = time_s / SEC_PER_HOUR
    teff = mag[:, 1]
    m_g_snec = mag[:, 4]
    lnu_snec = ab_mag_to_lnu(m_g_snec)
    nu_lnu_snec = NU_G_HZ * lnu_snec

    r_photo_cm = interp_like(time_s, rad_photo[:, 0], rad_photo[:, 1])
    lnu_bb_snec_rt = lnu_from_blackbody(r_photo_cm, teff)
    m_g_bb_snec_rt = lnu_to_ab_mag(lnu_bb_snec_rt)
    nu_lnu_bb_snec_rt = NU_G_HZ * lnu_bb_snec_rt

    r_initial = rad_initial[:, 1] if rad_initial.shape[1] > 1 else rad_initial[:, 0]
    m_initial = mass_initial[:, 1] if mass_initial.shape[1] > 1 else mass_initial[:, 0]
    r_star_cm = float(np.nanmax(r_initial))
    r_star_rsun = r_star_cm / R_SUN
    mass_msun = float(np.nanmax(m_initial) / M_SUN)
    explosion_energy_foe = 1.0
    # This is a bulk ejecta speed scale. The fastest outer layers can be faster.
    v_bulk = float(np.sqrt(2.0 * explosion_energy_foe * FOE / (mass_msun * M_SUN)))

    l_bol_theory, t_s_theory = nakar_wr_bolometric_luminosity(
        time_s=time_s,
        mass_msun=mass_msun,
        radius_rsun=r_star_rsun,
        explosion_energy_foe=explosion_energy_foe,
    )
    r_theory_cm = r_star_cm + v_bulk * time_s
    t_theory = (l_bol_theory / (4.0 * np.pi * SIGMA_SB * r_theory_cm**2)) ** 0.25
    lnu_theory = lnu_from_blackbody(r_theory_cm, t_theory)
    m_g_theory = lnu_to_ab_mag(lnu_theory)
    nu_lnu_theory = NU_G_HZ * lnu_theory

    finite = (
        np.isfinite(time_h)
        & np.isfinite(teff)
        & np.isfinite(m_g_snec)
        & np.isfinite(m_g_bb_snec_rt)
        & np.isfinite(m_g_theory)
        & np.isfinite(nu_lnu_snec)
        & np.isfinite(nu_lnu_bb_snec_rt)
        & np.isfinite(nu_lnu_theory)
        & (time_h > 0.0)
        & (time_h <= 48.0)
        & (teff > 0.0)
        & (r_photo_cm > 0.0)
    )

    time_h = time_h[finite]
    teff = teff[finite]
    m_g_snec = m_g_snec[finite]
    m_g_bb_snec_rt = m_g_bb_snec_rt[finite]
    m_g_theory = m_g_theory[finite]
    nu_lnu_snec = nu_lnu_snec[finite]
    nu_lnu_bb_snec_rt = nu_lnu_bb_snec_rt[finite]
    nu_lnu_theory = nu_lnu_theory[finite]
    r_photo_cm = r_photo_cm[finite]
    t_theory = t_theory[finite]

    peak_temp_index = int(np.nanargmax(teff))
    cooling_mask = np.arange(len(time_h)) >= peak_temp_index

    return {
        "run_id": model.run_id,
        "label": model.label,
        "short_label": model.short_label,
        "color": model.color,
        "time_h": time_h,
        "teff_K": teff,
        "R_ph_Rsun": r_photo_cm / R_SUN,
        "M_g_snec": m_g_snec,
        "M_g_bb_snec_rt": m_g_bb_snec_rt,
        "M_g_theory": m_g_theory,
        "nu_Lnu_snec": nu_lnu_snec,
        "nu_Lnu_bb_snec_rt": nu_lnu_bb_snec_rt,
        "nu_Lnu_theory": nu_lnu_theory,
        "T_theory_K": t_theory,
        "cooling_mask": cooling_mask,
        "R_star_Rsun": r_star_rsun,
        "mass_msun": mass_msun,
        "v_bulk_cm_s": v_bulk,
        "t_s_theory_h": t_s_theory / SEC_PER_HOUR,
    }


def model_handles(records: list[dict[str, np.ndarray | float | str]]) -> list[Line2D]:
    return [
        Line2D([0], [0], color=str(record["color"]), lw=2.4, label=str(record["label"]))
        for record in records
    ]


def style_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#111827", lw=2.0, label="SNEC BolCorr"),
        Line2D([0], [0], color="#111827", lw=2.0, ls=":", label="Planck with SNEC R_ph,T_eff"),
        Line2D([0], [0], color="#111827", lw=1.8, ls="--", label="semi-analytic shock cooling"),
    ]


def make_plot(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.9), constrained_layout=True)
    ax_mag, ax_lum = axes

    for record in records:
        mask = record["cooling_mask"]
        assert isinstance(mask, np.ndarray)
        time_h = record["time_h"][mask]
        color = str(record["color"])

        ax_mag.plot(time_h, record["M_g_snec"][mask], color=color, lw=2.1)
        ax_mag.plot(time_h, record["M_g_bb_snec_rt"][mask], color=color, lw=2.0, ls=":")
        ax_mag.plot(time_h, record["M_g_theory"][mask], color=color, lw=1.8, ls="--")

        ax_lum.plot(time_h, record["nu_Lnu_snec"][mask], color=color, lw=2.1)
        ax_lum.plot(time_h, record["nu_Lnu_bb_snec_rt"][mask], color=color, lw=2.0, ls=":")
        ax_lum.plot(time_h, record["nu_Lnu_theory"][mask], color=color, lw=1.8, ls="--")

    ax_mag.set_xscale("log")
    ax_mag.set_xlim(0.01, 48.0)
    ax_mag.invert_yaxis()
    ax_mag.set_xlabel("Time since explosion (hours)")
    ax_mag.set_ylabel(r"Absolute $g$ magnitude, $M_g$")
    ax_mag.set_title(r"$g$ magnitude with a semi-analytic theory curve")
    style_paper_axis(ax_mag)
    leg1 = ax_mag.legend(handles=model_handles(records), fontsize=8, loc="upper right")
    ax_mag.add_artist(leg1)
    ax_mag.legend(handles=style_handles(), fontsize=8, loc="lower left")

    ax_lum.set_xscale("log")
    ax_lum.set_yscale("log")
    ax_lum.set_xlim(0.01, 48.0)
    ax_lum.set_xlabel("Time since explosion (hours)")
    ax_lum.set_ylabel(r"$\nu_g L_{\nu,g}$ (erg s$^{-1}$)")
    ax_lum.set_title(r"$g$-band luminosity from evolving $L_{\rm bol},R,T$")
    style_paper_axis(ax_lum)

    fig.suptitle(
        r"Low-Ni WN3: SNEC vs blackbody conversion vs semi-analytic shock cooling",
        fontsize=13,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_time_dependent_gband_with_semianalytic_theory.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def write_metrics(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    path = OUT_DIR / "low_ni_time_dependent_gband_theory_metrics.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "label",
        "R_star_Rsun",
        "mass_Msun",
        "v_bulk_cm_s",
        "theory_transition_time_h",
        "median_delta_theory_mag",
        "median_theory_over_snec_nuLnu",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            mask = record["cooling_mask"]
            assert isinstance(mask, np.ndarray)
            snec = record["nu_Lnu_snec"][mask]
            theory = record["nu_Lnu_theory"][mask]
            delta_mag = record["M_g_theory"][mask] - record["M_g_snec"][mask]
            writer.writerow(
                {
                    "run_id": record["run_id"],
                    "label": record["short_label"],
                    "R_star_Rsun": f"{float(record['R_star_Rsun']):.6g}",
                    "mass_Msun": f"{float(record['mass_msun']):.6g}",
                    "v_bulk_cm_s": f"{float(record['v_bulk_cm_s']):.6e}",
                    "theory_transition_time_h": f"{float(record['t_s_theory_h']):.6g}",
                    "median_delta_theory_mag": f"{float(np.nanmedian(delta_mag)):.6g}",
                    "median_theory_over_snec_nuLnu": f"{float(np.nanmedian(theory / snec)):.6g}",
                }
            )
    return path


def main() -> None:
    records = [load_model(model) for model in MODELS]
    for output in [make_plot(records), write_metrics(records)]:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
