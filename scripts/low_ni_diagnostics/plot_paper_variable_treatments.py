#!/usr/bin/env python3
"""Compare the paper-motivated variable treatments against SNEC g-band outputs.

The plot separates the conversion problem into:
  1. the bandpass conversion using SNEC R_ph(t) and T_eff(t),
  2. the Nakar/Sari compact-star bolometric shock-cooling luminosity,
  3. the radius and temperature terms that propagate into the g-band prediction.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path("/Users/arifchu/Documents/Codex/2026-06-09/scientific-purpose-looks-like-you-re")
SNEC_ROOT = Path("/Users/arifchu/Desktop/GitHub/supernova_theory")
RUN_ROOT = SNEC_ROOT / "snec" / "runs" / "model_variants"
OUT_DIR = WORKSPACE / "output" / "low_ni_diagnostics"

M_SUN = 1.989e33
R_SUN = 6.957e10
E_51_SCALE = 1.0e51
SIGMA_SB = 5.67e-5
H_PLANCK = 6.626e-27
K_B = 1.38e-16
C_LIGHT = 3.0e10
PC_CM = 3.086e18
SEC_PER_HOUR = 3600.0

LAMBDA_G_CM = 4770.0e-8
NU_G = C_LIGHT / LAMBDA_G_CM


@dataclass(frozen=True)
class Model:
    run_id: str
    label: str
    color: str


MODELS = [
    Model("wn3_sce_bare_e1_ni001", "Bare WN3", "#2b6cb0"),
    Model("wn3_sce_mass_m0p01_r5_e1_ni001", r"$+0.01\,M_\odot$ to $5\,R_\odot$", "#d97706"),
    Model("wn3_sce_radius_m0p01_r50_e1_ni001", r"$+0.01\,M_\odot$ to $50\,R_\odot$", "#15803d"),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.2,
            "axes.labelsize": 10.8,
            "axes.titlesize": 10.8,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.9,
            "lines.solid_capstyle": "round",
        }
    )


def style_axis(ax) -> None:
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=5, width=0.9)
    ax.tick_params(which="minor", length=2.8, width=0.75)


def panel_label(ax, label: str) -> None:
    ax.text(
        0.035,
        0.92,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )


def load_table(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, data.shape[0])
    return data


def load_value_column(path: Path) -> np.ndarray:
    data = load_table(path)
    return data[:, 1] if data.shape[1] > 1 else data[:, 0]


def interp_positive(source_time_h: np.ndarray, source_y: np.ndarray, target_time_h: np.ndarray) -> np.ndarray:
    keep = np.isfinite(source_time_h) & np.isfinite(source_y) & (source_time_h >= 0.0)
    order = np.argsort(source_time_h[keep])
    return np.interp(target_time_h, source_time_h[keep][order], source_y[keep][order])


def ab_mag_to_lnu(m_ab: np.ndarray) -> np.ndarray:
    f_nu_10pc = 10.0 ** (-0.4 * (m_ab + 48.60))
    return 4.0 * np.pi * (10.0 * PC_CM) ** 2 * f_nu_10pc


def planck_bnu(temperature: np.ndarray) -> np.ndarray:
    exponent = np.clip((H_PLANCK * NU_G) / (K_B * temperature), None, 700.0)
    return (2.0 * H_PLANCK * NU_G**3 / C_LIGHT**2) / np.expm1(exponent)


def planck_nu_lnu(radius_cm: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    l_nu = 4.0 * np.pi**2 * radius_cm**2 * planck_bnu(temperature)
    return NU_G * l_nu


def nakar_sari_lbol(time_s: np.ndarray, m_ej_g: float, r_star_cm: float, e_kin_erg: float) -> tuple[np.ndarray, float]:
    m_15 = m_ej_g / (15.0 * M_SUN)
    r_5 = r_star_cm / (5.0 * R_SUN)
    e_51 = e_kin_erg / E_51_SCALE
    t_s = 90.0 * (m_15**0.41) * (r_5**1.33) * (e_51**-0.58)
    t_min = time_s / 60.0
    t_hr = time_s / SEC_PER_HOUR
    planar = 2.0e42 * (m_15**-0.33) * (r_5**2.3) * (e_51**0.34) * (t_min ** (-4.0 / 3.0))
    spherical = 3.5e41 * (m_15**-0.73) * r_5 * (e_51**0.91) * (t_hr**-0.35)
    return np.where(time_s < t_s, planar, spherical), float(t_s)


def nakar_sari_radius_proxy(time_s: np.ndarray, r_star_cm: float, t_s: float) -> np.ndarray:
    return np.where(time_s < t_s, r_star_cm, r_star_cm * (time_s / t_s) ** 0.725)


def load_model(model: Model) -> dict[str, np.ndarray | float | str]:
    data_dir = RUN_ROOT / model.run_id / "Data"
    magnitudes = load_table(data_dir / "magnitudes.dat")
    mass_initial = load_value_column(data_dir / "mass_initial.dat")
    radius_initial = load_value_column(data_dir / "rad_initial.dat")
    lum_photo = load_table(data_dir / "lum_photo.dat")
    rad_photo = load_table(data_dir / "rad_photo.dat")

    time_h = magnitudes[:, 0] / SEC_PER_HOUR
    t_sec = magnitudes[:, 0]
    teff_for_bc = magnitudes[:, 1]
    m_g = magnitudes[:, 4]
    snec_nu_lnu_g = NU_G * ab_mag_to_lnu(m_g)

    keep = (
        np.isfinite(time_h)
        & np.isfinite(teff_for_bc)
        & np.isfinite(snec_nu_lnu_g)
        & (time_h >= 0.03)
        & (time_h <= 48.0)
        & (m_g <= -5.0)
        & (teff_for_bc > 0.0)
        & (snec_nu_lnu_g > 0.0)
    )

    time_h = time_h[keep]
    t_sec = t_sec[keep]
    teff_for_bc = teff_for_bc[keep]
    m_g = m_g[keep]
    snec_nu_lnu_g = snec_nu_lnu_g[keep]

    l_photo = interp_positive(lum_photo[:, 0] / SEC_PER_HOUR, lum_photo[:, 1], time_h)
    r_photo = interp_positive(rad_photo[:, 0] / SEC_PER_HOUR, rad_photo[:, 1], time_h)

    m_ej_g = float(np.nanmax(mass_initial))
    r_star_cm = float(np.nanmax(radius_initial))
    l_ns, t_s = nakar_sari_lbol(t_sec, m_ej_g, r_star_cm, E_51_SCALE)
    r_ns = nakar_sari_radius_proxy(t_sec, r_star_cm, t_s)
    t_ns_from_snec_radius = (l_ns / (4.0 * np.pi * SIGMA_SB * r_photo**2)) ** 0.25

    snec_planck_nu_lnu = planck_nu_lnu(r_photo, teff_for_bc)
    ns_snec_radius_nu_lnu = planck_nu_lnu(r_photo, t_ns_from_snec_radius)

    valid = (
        np.isfinite(l_photo)
        & np.isfinite(r_photo)
        & np.isfinite(l_ns)
        & np.isfinite(r_ns)
        & np.isfinite(t_ns_from_snec_radius)
        & (l_photo > 0.0)
        & (r_photo > 0.0)
        & (l_ns > 0.0)
        & (r_ns > 0.0)
        & (t_ns_from_snec_radius > 0.0)
    )

    return {
        "run_id": model.run_id,
        "label": model.label,
        "color": model.color,
        "time_h": time_h[valid],
        "m_g": m_g[valid],
        "teff_snec": teff_for_bc[valid],
        "r_photo_cm": r_photo[valid],
        "l_photo": l_photo[valid],
        "snec_nu_lnu_g": snec_nu_lnu_g[valid],
        "snec_planck_nu_lnu_g": snec_planck_nu_lnu[valid],
        "l_ns": l_ns[valid],
        "r_ns": r_ns[valid],
        "t_ns_from_snec_radius": t_ns_from_snec_radius[valid],
        "ns_snec_radius_nu_lnu_g": ns_snec_radius_nu_lnu[valid],
        "m_ej_msun": m_ej_g / M_SUN,
        "r_star_rsun": r_star_cm / R_SUN,
        "transition_time_s": t_s,
    }


def make_plot(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 12.0), constrained_layout=True)

    for row, record in enumerate(records):
        color = str(record["color"])
        label = str(record["label"])
        time_h = record["time_h"]
        snec_nu_lnu = record["snec_nu_lnu_g"]
        planck_nu_lnu = record["snec_planck_nu_lnu_g"]
        ns_nu_lnu = record["ns_snec_radius_nu_lnu_g"]
        l_ratio = record["l_ns"] / record["l_photo"]
        r_ratio = record["r_ns"] / record["r_photo_cm"]
        t_ratio = record["t_ns_from_snec_radius"] / record["teff_snec"]
        planck_ratio = planck_nu_lnu / snec_nu_lnu
        ns_ratio = ns_nu_lnu / snec_nu_lnu

        assert isinstance(time_h, np.ndarray)
        assert isinstance(snec_nu_lnu, np.ndarray)
        assert isinstance(planck_nu_lnu, np.ndarray)
        assert isinstance(ns_nu_lnu, np.ndarray)
        assert isinstance(l_ratio, np.ndarray)
        assert isinstance(r_ratio, np.ndarray)
        assert isinstance(t_ratio, np.ndarray)
        assert isinstance(planck_ratio, np.ndarray)
        assert isinstance(ns_ratio, np.ndarray)

        ax_lc, ax_var = axes[row]

        ax_lc.plot(time_h, snec_nu_lnu, color=color, lw=2.2, label="SNEC g band")
        ax_lc.plot(
            time_h,
            planck_nu_lnu,
            color="#111827",
            lw=1.6,
            ls=":",
            label=r"Planck: SNEC $R_{\rm ph},T_{\rm eff}$",
        )
        ax_lc.plot(
            time_h,
            ns_nu_lnu,
            color="#6b7280",
            lw=1.8,
            ls="--",
            label=r"Nakar/Sari $L$ + SNEC $R_{\rm ph}$",
        )
        ax_lc.set_xscale("log")
        ax_lc.set_yscale("log")
        ax_lc.set_title(label)
        ax_lc.set_ylabel(r"$\nu_gL_{\nu,g}$ (erg s$^{-1}$)")
        if row == 0:
            ax_lc.legend(frameon=False, loc="lower left")
        panel_label(ax_lc, f"({chr(97 + 2 * row)})")
        style_axis(ax_lc)

        ax_var.axhline(1.0, color="#111827", lw=0.9)
        ax_var.axhspan(0.5, 2.0, color="#9ca3af", alpha=0.14, lw=0)
        ax_var.plot(time_h, l_ratio, color="#8b1a1a", lw=1.6, label=r"$L_{\rm NS10}/L_{\rm SNEC}$")
        ax_var.plot(time_h, r_ratio, color="#075985", lw=1.5, ls="--", label=r"$R_{\rm NS\,proxy}/R_{\rm ph,SNEC}$")
        ax_var.plot(time_h, t_ratio, color="#166534", lw=1.5, ls="-.", label=r"$T_{\rm NS+R_{ph}}/T_{\rm SNEC}$")
        ax_var.plot(time_h, ns_ratio, color=color, lw=1.8, ls=":", label=r"$g_{\rm NS+R_{ph}}/g_{\rm SNEC}$")
        ax_var.plot(time_h, planck_ratio, color="#111827", lw=1.0, alpha=0.75, label=r"$g_{\rm Planck}/g_{\rm SNEC}$")
        ax_var.set_xscale("log")
        ax_var.set_yscale("log")
        ax_var.set_ylim(0.03, 80.0)
        ax_var.set_title("Variable-ratio diagnosis")
        ax_var.set_ylabel("Ratio to SNEC")
        if row == 0:
            ax_var.legend(frameon=False, loc="upper right", ncols=1)
        panel_label(ax_var, f"({chr(98 + 2 * row)})")
        style_axis(ax_var)

    for ax in axes[-1]:
        ax.set_xlabel("Time since explosion (h)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / "low_ni_paper_variable_treatment_comparison.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def write_summary(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    output = OUT_DIR / "low_ni_paper_variable_treatment_summary.csv"
    fields = [
        "run_id",
        "label",
        "R_star_Rsun",
        "M_ej_Msun",
        "transition_time_s",
        "median_planck_g_to_snec_g",
        "median_ns_g_to_snec_g",
        "median_ns_lbol_to_snec_lphoto",
        "median_ns_radius_proxy_to_snec_rph",
        "median_ns_temperature_to_snec_teff",
        "max_ns_g_to_snec_g",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            snec_nu_lnu = record["snec_nu_lnu_g"]
            planck_nu_lnu = record["snec_planck_nu_lnu_g"]
            ns_nu_lnu = record["ns_snec_radius_nu_lnu_g"]
            l_ratio = record["l_ns"] / record["l_photo"]
            r_ratio = record["r_ns"] / record["r_photo_cm"]
            t_ratio = record["t_ns_from_snec_radius"] / record["teff_snec"]
            planck_ratio = planck_nu_lnu / snec_nu_lnu
            ns_ratio = ns_nu_lnu / snec_nu_lnu
            writer.writerow(
                {
                    "run_id": record["run_id"],
                    "label": record["label"],
                    "R_star_Rsun": f"{float(record['r_star_rsun']):.8g}",
                    "M_ej_Msun": f"{float(record['m_ej_msun']):.8g}",
                    "transition_time_s": f"{float(record['transition_time_s']):.8g}",
                    "median_planck_g_to_snec_g": f"{float(np.nanmedian(planck_ratio)):.6g}",
                    "median_ns_g_to_snec_g": f"{float(np.nanmedian(ns_ratio)):.6g}",
                    "median_ns_lbol_to_snec_lphoto": f"{float(np.nanmedian(l_ratio)):.6g}",
                    "median_ns_radius_proxy_to_snec_rph": f"{float(np.nanmedian(r_ratio)):.6g}",
                    "median_ns_temperature_to_snec_teff": f"{float(np.nanmedian(t_ratio)):.6g}",
                    "max_ns_g_to_snec_g": f"{float(np.nanmax(ns_ratio)):.6g}",
                }
            )
    return output


def main() -> None:
    setup_style()
    records = [load_model(model) for model in MODELS]
    outputs = [make_plot(records), write_summary(records)]
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
