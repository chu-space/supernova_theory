#!/usr/bin/env python3
"""Compare Nakar/Sari-style g-band prediction with the SNEC g-band output."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path("/Users/arifchu/Documents/Codex/2026-06-09/scientific-purpose-looks-like-you-re")
DATA_DIR = (
    Path("/Users/arifchu/Desktop/GitHub/supernova_theory")
    / "snec"
    / "runs"
    / "model_variants"
    / "wn3_sce_mass_m0p01_r5_e1_ni001"
    / "Data"
)
OUT_DIR = WORKSPACE / "output" / "low_ni_diagnostics"

# --- CONSTANTS (cgs units) ---
M_sun = 1.989e33
R_sun = 6.957e10
E_51_scale = 1e51
sigma_SB = 5.67e-5
h = 6.626e-27
k_B = 1.38e-16
c = 3.0e10
pc = 3.086e18

# g-band effective frequency, using lambda_eff ~ 4770 Angstrom.
lam_g = 4770e-8
nu_g = c / lam_g

# --- PROGENITOR PARAMETERS: WN3 + 5 R_sun extended material ---
M_ej = 7.0 * M_sun
R_star = 5.0 * R_sun
E_kin = 1.0 * E_51_scale

M_15 = M_ej / (15.0 * M_sun)
R_5 = R_star / (5.0 * R_sun)
E_51 = E_kin / E_51_scale


def style_axis(ax) -> None:
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=5, width=0.9)
    ax.tick_params(which="minor", length=2.8, width=0.75)


def load_table(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, data.shape[0])
    return data


def ab_mag_to_lnu(m_ab: np.ndarray) -> np.ndarray:
    f_nu_10pc = 10.0 ** (-0.4 * (m_ab + 48.60))
    return 4.0 * np.pi * (10.0 * pc) ** 2 * f_nu_10pc


def lnu_to_ab_mag(l_nu: np.ndarray) -> np.ndarray:
    f_nu_10pc = l_nu / (4.0 * np.pi * (10.0 * pc) ** 2)
    result = np.full_like(f_nu_10pc, np.nan, dtype=float)
    valid = np.isfinite(f_nu_10pc) & (f_nu_10pc > 0.0)
    result[valid] = -2.5 * np.log10(f_nu_10pc[valid]) - 48.60
    return result


def planck_bnu(temperature: np.ndarray) -> np.ndarray:
    exponent = (h * nu_g) / (k_B * temperature)
    exponent = np.clip(exponent, None, 700.0)
    return (2.0 * h * nu_g**3 / c**2) / np.expm1(exponent)


def nakar_sari_bolometric_luminosity(t_sec: np.ndarray) -> tuple[np.ndarray, float]:
    t_s = 90.0 * (M_15**0.41) * (R_5**1.33) * (E_51**-0.58)
    t_min = t_sec / 60.0
    t_hr = t_sec / 3600.0
    planar = 2.0e42 * (M_15**-0.33) * (R_5**2.3) * (E_51**0.34) * (t_min ** (-4.0 / 3.0))
    spherical = 3.5e41 * (M_15**-0.73) * (R_5**1.0) * (E_51**0.91) * (t_hr**-0.35)
    return np.where(t_sec < t_s, planar, spherical), t_s


def analytic_photospheric_radius(t_sec: np.ndarray, t_s: float) -> np.ndarray:
    return np.where(t_sec < t_s, R_star, R_star * (t_sec / t_s) ** 0.725)


def analytic_g_band_curve(t_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    l_bol, t_s = nakar_sari_bolometric_luminosity(t_sec)
    radius = analytic_photospheric_radius(t_sec, t_s)
    t_eff = (l_bol / (4.0 * np.pi * sigma_SB * radius**2)) ** 0.25
    l_nu = 4.0 * np.pi**2 * radius**2 * planck_bnu(t_eff)
    m_g = lnu_to_ab_mag(l_nu)
    return m_g, nu_g * l_nu, radius, t_eff


def load_snec_g_band() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mag = load_table(DATA_DIR / "magnitudes.dat")
    time_h = mag[:, 0] / 3600.0
    m_g = mag[:, 4]
    nu_lnu = nu_g * ab_mag_to_lnu(m_g)
    keep = (
        np.isfinite(time_h)
        & np.isfinite(m_g)
        & np.isfinite(nu_lnu)
        & (time_h > 0.0)
        & (time_h <= 48.0)
        & (nu_lnu > 0.0)
    )
    return time_h[keep], m_g[keep], nu_lnu[keep]


def make_plot() -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.9,
        }
    )

    t_snec_h, m_snec, nu_lnu_snec = load_snec_g_band()
    t_sec = np.linspace(60.0, 48.0 * 3600.0, 2000)
    t_h = t_sec / 3600.0
    m_ns, nu_lnu_ns, radius_ns, teff_ns = analytic_g_band_curve(t_sec)

    m_ns_at_snec = np.interp(t_snec_h, t_h, m_ns)
    nu_lnu_ns_at_snec = np.interp(t_snec_h, t_h, nu_lnu_ns)
    ratio = nu_lnu_ns_at_snec / nu_lnu_snec
    bright = (t_snec_h >= 0.03) & (m_snec <= -5.0) & np.isfinite(ratio) & (ratio > 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    ax_mag, ax_ratio = axes

    ax_mag.plot(t_snec_h[bright], m_snec[bright], color="#2b6cb0", lw=2.1, label="SNEC g band")
    model_keep = (t_h >= 0.03) & (t_h <= 48.0) & np.isfinite(m_ns) & (m_ns <= -5.0)
    ax_mag.plot(t_h[model_keep], m_ns[model_keep], color="#111827", lw=2.0, ls=":", label="Nakar/Sari + Planck g band")
    ax_mag.set_xscale("log")
    ax_mag.invert_yaxis()
    ax_mag.set_xlabel("Time since explosion (h)")
    ax_mag.set_ylabel(r"Absolute $g$ magnitude, $M_g$")
    ax_mag.set_title(r"WN3 $+0.01\,M_\odot$ to $5\,R_\odot$")
    ax_mag.legend(frameon=False, loc="upper right")
    style_axis(ax_mag)

    ax_ratio.axhline(1.0, color="#111827", lw=1.0, alpha=0.8)
    ax_ratio.axhspan(0.5, 2.0, color="#9ca3af", alpha=0.16, lw=0)
    ax_ratio.plot(t_snec_h[bright], ratio[bright], color="#2b6cb0", lw=2.0)
    ax_ratio.set_xscale("log")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlabel("Time since explosion (h)")
    ax_ratio.set_ylabel(r"$(\nu L_{\nu,g})_{\rm N/S}/(\nu L_{\nu,g})_{\rm SNEC}$")
    ax_ratio.set_title("Luminosity ratio")
    ax_ratio.set_ylim(0.08, 12.0)
    style_axis(ax_ratio)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUT_DIR / "low_ni_nakar_sari_vs_snec_gband_ratio.png"
    fig.savefig(plot_path)
    plt.close(fig)

    csv_path = OUT_DIR / "low_ni_nakar_sari_vs_snec_gband_ratio.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time_h",
                "M_g_snec",
                "M_g_nakar_sari",
                "nuLnu_g_snec_erg_s",
                "nuLnu_g_nakar_sari_erg_s",
                "ratio_nakar_sari_to_snec",
            ],
        )
        writer.writeheader()
        for time, snec_mag, ns_mag, snec_lum, ns_lum, value in zip(
            t_snec_h[bright],
            m_snec[bright],
            m_ns_at_snec[bright],
            nu_lnu_snec[bright],
            nu_lnu_ns_at_snec[bright],
            ratio[bright],
        ):
            writer.writerow(
                {
                    "time_h": f"{time:.8e}",
                    "M_g_snec": f"{snec_mag:.8e}",
                    "M_g_nakar_sari": f"{ns_mag:.8e}",
                    "nuLnu_g_snec_erg_s": f"{snec_lum:.8e}",
                    "nuLnu_g_nakar_sari_erg_s": f"{ns_lum:.8e}",
                    "ratio_nakar_sari_to_snec": f"{value:.8e}",
                }
            )

    return plot_path, csv_path


def main() -> None:
    plot_path, csv_path = make_plot()
    print(f"Saved {plot_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
