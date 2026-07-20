#!/usr/bin/env python3
"""Plot the expanding-radius Rayleigh-Jeans correction against SNEC g band."""

from __future__ import annotations

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
    color: str


MODELS = [
    Model("wn3_sce_bare_e1_ni001", "Bare WN3", "#2b6cb0"),
    Model("wn3_sce_mass_m0p01_r5_e1_ni001", r"WN3 + 0.01 $M_\odot$ to 5 $R_\odot$", "#d97706"),
    Model("wn3_sce_radius_m0p01_r50_e1_ni001", r"WN3 + 0.01 $M_\odot$ to 50 $R_\odot$", "#15803d"),
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
    f_nu_10pc = 10.0 ** (-0.4 * (m_ab + 48.60))
    l_nu = 4.0 * np.pi * TEN_PC_CM**2 * f_nu_10pc
    return NU_G_HZ * l_nu


def load_model(model: Model) -> dict[str, np.ndarray | float | str]:
    data_dir = RUN_ROOT / model.run_id / "Data"
    mag = load_table(data_dir / "magnitudes.dat")
    time_s = mag[:, 0]
    time_h = time_s / SEC_PER_HOUR
    teff = mag[:, 1]
    m_g = mag[:, 4]
    nu_lnu = ab_mag_to_nu_lnu(m_g)

    t_r, r_photo_cm = load_two_column(data_dir / "rad_photo.dat")
    r_photo = interp_like(time_s, t_r, r_photo_cm)

    rad_initial = load_table(data_dir / "rad_initial.dat")
    r_star = float(np.nanmax(rad_initial[:, 1] if rad_initial.shape[1] > 1 else rad_initial[:, 0]))

    finite = (
        np.isfinite(time_s)
        & np.isfinite(time_h)
        & np.isfinite(teff)
        & np.isfinite(m_g)
        & np.isfinite(nu_lnu)
        & np.isfinite(r_photo)
        & (time_s > 0.0)
        & (time_h <= 48.0)
        & (teff > 0.0)
        & (nu_lnu > 0.0)
        & (r_photo > 0.0)
    )

    time_s = time_s[finite]
    time_h = time_h[finite]
    teff = teff[finite]
    m_g = m_g[finite]
    nu_lnu = nu_lnu[finite]
    r_photo = r_photo[finite]

    peak_temp_idx = int(np.nanargmax(teff))
    post_breakout = np.arange(len(time_s)) >= peak_temp_idx
    peak_g_local = int(np.nanargmax(nu_lnu[post_breakout]))
    post_indices = np.where(post_breakout)[0]
    peak_g_idx = int(post_indices[peak_g_local])

    # Estimate an expansion speed from the photospheric-radius rise before the g peak.
    fit_mask = post_breakout & (time_s <= time_s[peak_g_idx]) & (r_photo > 0.0)
    fit_t = time_s[fit_mask]
    fit_r = r_photo[fit_mask]
    if len(fit_t) >= 3 and np.nanmax(fit_t) > np.nanmin(fit_t):
        slope, intercept = np.polyfit(fit_t, fit_r, 1)
        v_fit = max(float(slope), 1.0e7)
    else:
        v_fit = max(float(r_photo[peak_g_idx] / time_s[peak_g_idx]), 1.0e7)

    r_model = r_star + v_fit * time_s
    scale = float(nu_lnu[peak_g_idx] / r_model[peak_g_idx] ** 1.5)
    corrected_scaling = scale * r_model**1.5

    t_power = time_s / time_s[peak_g_idx]
    t32_scaling = float(nu_lnu[peak_g_idx]) * t_power**1.5

    return {
        "label": model.label,
        "color": model.color,
        "time_h": time_h,
        "post_breakout": post_breakout,
        "nu_lnu": nu_lnu,
        "corrected_scaling": corrected_scaling,
        "t32_scaling": t32_scaling,
        "peak_time_h": float(time_h[peak_g_idx]),
        "peak_nu_lnu": float(nu_lnu[peak_g_idx]),
        "r_star_rsun": float(r_star / R_SUN),
        "v_fit_cm_s": v_fit,
    }


def make_plot(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), constrained_layout=True)
    ax, ax_ratio = axes

    for record in records:
        mask = record["post_breakout"]
        assert isinstance(mask, np.ndarray)
        time_h = record["time_h"][mask]
        nu_lnu = record["nu_lnu"][mask]
        corrected = record["corrected_scaling"][mask]
        color = str(record["color"])
        label = str(record["label"])

        ax.plot(time_h, nu_lnu, color=color, lw=2.1, label=f"{label} - SNEC g")
        ax.plot(time_h, corrected, color=color, lw=1.8, ls=":", label=f"{label} - $r(t)^{{3/2}}$")

        ratio = corrected / nu_lnu
        ax_ratio.plot(time_h, ratio, color=color, lw=1.9, label=label)
        ax.scatter(float(record["peak_time_h"]), float(record["peak_nu_lnu"]), color=color, edgecolor="black", s=42, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 48.0)
    ax.set_xlabel("Time since explosion (hours)")
    ax.set_ylabel(r"$\nu_g L_{\nu,g}$ from SNEC $M_g$ (erg s$^{-1}$)")
    ax.set_title(r"Corrected Expansion Scaling for $g$ Band")
    style_paper_axis(ax)
    ax.legend(fontsize=8, ncol=1)

    ax_ratio.set_xscale("log")
    ax_ratio.set_yscale("log")
    ax_ratio.axhline(1.0, color="#111827", lw=1.0, alpha=0.7)
    ax_ratio.set_xlim(0.01, 48.0)
    ax_ratio.set_xlabel("Time since explosion (hours)")
    ax_ratio.set_ylabel(r"normalized $r(t)^{3/2}$ / SNEC $g$")
    ax_ratio.set_title("Approximation Residual")
    style_paper_axis(ax_ratio)
    ax_ratio.legend(fontsize=8)

    fig.suptitle(
        r"Board Correction: Use the Expanding Emitting Radius, $r(t)=R_*+vt$",
        fontsize=14,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_gband_time_expanding_radius_correction.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def main() -> None:
    records = [load_model(model) for model in MODELS]
    path = make_plot(records)
    print(f"Saved plot to {path}")
    for record in records:
        print(
            f"{record['label']}: R_star={float(record['r_star_rsun']):.3g} R_sun, "
            f"v_fit={float(record['v_fit_cm_s']):.3e} cm/s, "
            f"g peak={float(record['peak_time_h']):.3g} h"
        )


if __name__ == "__main__":
    main()
