#!/usr/bin/env python3
"""Fit a simple semi-analytic shock-cooling g-band model to SNEC magnitudes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import least_squares


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


def semi_analytic_mg(
    time_h: np.ndarray,
    r_star_cm: float,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fitted semi-analytic M_g, bolometric luminosity, and temperature.

    Parameters are:
    log10_L1h, alpha, log10_v, log10_fT, log10_fR0.
    """
    log10_l1h, alpha, log10_v, log10_ft, log10_fr0 = theta
    t_model_h = np.maximum(time_h, 1.0e-3)
    t_model_s = t_model_h * SEC_PER_HOUR

    l_bol = 10.0**log10_l1h * t_model_h ** (-alpha)
    radius = (10.0**log10_fr0) * r_star_cm + (10.0**log10_v) * t_model_s
    t_eff = (l_bol / (4.0 * np.pi * SIGMA_SB * radius**2)) ** 0.25
    t_color = (10.0**log10_ft) * t_eff
    lnu = 4.0 * np.pi**2 * radius**2 * planck_bnu(NU_G_HZ, t_color)
    return lnu_to_ab_mag(lnu), l_bol, t_color


def load_model(model: Model) -> dict[str, np.ndarray | float | str]:
    data_dir = RUN_ROOT / model.run_id / "Data"
    mag = load_table(data_dir / "magnitudes.dat")
    rad_initial = load_table(data_dir / "rad_initial.dat")

    time_h = mag[:, 0] / SEC_PER_HOUR
    teff = mag[:, 1]
    m_g = mag[:, 4]
    nu_lnu = NU_G_HZ * ab_mag_to_lnu(m_g)

    r_initial = rad_initial[:, 1] if rad_initial.shape[1] > 1 else rad_initial[:, 0]
    r_star_cm = float(np.nanmax(r_initial))

    finite = (
        np.isfinite(time_h)
        & np.isfinite(teff)
        & np.isfinite(m_g)
        & np.isfinite(nu_lnu)
        & (time_h > 0.0)
        & (time_h <= 48.0)
        & (teff > 0.0)
        & (nu_lnu > 0.0)
    )
    time_h = time_h[finite]
    teff = teff[finite]
    m_g = m_g[finite]
    nu_lnu = nu_lnu[finite]

    peak_temp_index = int(np.nanargmax(teff))
    cooling_mask = np.arange(len(time_h)) >= peak_temp_index
    bright_mask = cooling_mask & (nu_lnu >= 0.10 * np.nanmax(nu_lnu[cooling_mask]))

    return {
        "run_id": model.run_id,
        "label": model.label,
        "short_label": model.short_label,
        "color": model.color,
        "time_h": time_h,
        "M_g_snec": m_g,
        "nu_Lnu_snec": nu_lnu,
        "cooling_mask": cooling_mask,
        "bright_mask": bright_mask,
        "R_star_cm": r_star_cm,
        "R_star_Rsun": r_star_cm / R_SUN,
    }


def fit_model(record: dict[str, np.ndarray | float | str]) -> dict[str, np.ndarray | float | str]:
    time_h = record["time_h"]
    m_g = record["M_g_snec"]
    bright_mask = record["bright_mask"]
    r_star_cm = float(record["R_star_cm"])
    assert isinstance(time_h, np.ndarray)
    assert isinstance(m_g, np.ndarray)
    assert isinstance(bright_mask, np.ndarray)

    fit_time = time_h[bright_mask]
    fit_mg = m_g[bright_mask]

    def residual(theta: np.ndarray) -> np.ndarray:
        model_mg, _, _ = semi_analytic_mg(fit_time, r_star_cm, theta)
        return model_mg - fit_mg

    # Initial values: L(1h), slope, v, color factor, radius scale.
    x0 = np.array([41.5, 0.35, 9.0, 0.0, 0.0])
    lower = np.array([38.0, -1.0, 7.0, -0.5, -1.5])
    upper = np.array([45.5, 3.0, 10.5, 0.7, 1.5])
    result = least_squares(residual, x0, bounds=(lower, upper), loss="soft_l1", f_scale=0.05)

    fit_mg_model, fit_lbol, fit_tcolor = semi_analytic_mg(fit_time, r_star_cm, result.x)
    full_mg_model, full_lbol, full_tcolor = semi_analytic_mg(time_h, r_star_cm, result.x)
    full_lnu = ab_mag_to_lnu(full_mg_model)
    ratio = (NU_G_HZ * full_lnu) / record["nu_Lnu_snec"]
    fit_resid = fit_mg_model - fit_mg

    return {
        **record,
        "fit_theta": result.x,
        "fit_success": bool(result.success),
        "fit_cost": float(result.cost),
        "fit_rms_mag": float(np.sqrt(np.nanmean(fit_resid**2))),
        "fit_mad_mag": float(np.nanmedian(np.abs(fit_resid))),
        "M_g_fit": full_mg_model,
        "nu_Lnu_fit": NU_G_HZ * full_lnu,
        "ratio_fit_snec": ratio,
        "L_bol_fit": full_lbol,
        "T_color_fit": full_tcolor,
    }


def model_handles(records: list[dict[str, np.ndarray | float | str]]) -> list[Line2D]:
    return [
        Line2D([0], [0], color=str(record["color"]), lw=2.4, label=str(record["label"]))
        for record in records
    ]


def style_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#111827", lw=2.2, label="SNEC g-band output"),
        Line2D([0], [0], color="#111827", lw=2.0, ls="--", label="fitted semi-analytic model"),
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
        ax_mag.plot(time_h, record["M_g_fit"][mask], color=color, lw=2.0, ls="--")
        ax_ratio.plot(time_h, record["ratio_fit_snec"][mask], color=color, lw=2.0)

    ax_mag.set_xscale("log")
    ax_mag.set_xlim(0.01, 48.0)
    ax_mag.invert_yaxis()
    ax_mag.set_xlabel("Time since explosion (hours)")
    ax_mag.set_ylabel(r"Absolute $g$ magnitude, $M_g$")
    ax_mag.set_title(r"Fitted semi-analytic $g$-band model")
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
    ax_ratio.set_ylabel(r"semi-analytic / SNEC $g$ luminosity")
    ax_ratio.set_title("Fit residual")
    style_paper_axis(ax_ratio)

    fig.suptitle(
        r"Calibrated semi-analytic model: fitting $L_{\rm bol}(t)$, $R(t)$, and color factor",
        fontsize=13,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_gband_fitted_semianalytic_match.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def write_metrics(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    path = OUT_DIR / "low_ni_gband_fitted_semianalytic_metrics.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "label",
        "fit_success",
        "fit_rms_mag",
        "fit_mad_mag",
        "log10_L_1h",
        "alpha_L",
        "v_cm_s",
        "color_factor_fT",
        "initial_radius_scale_fR0",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            theta = record["fit_theta"]
            assert isinstance(theta, np.ndarray)
            writer.writerow(
                {
                    "run_id": record["run_id"],
                    "label": record["short_label"],
                    "fit_success": record["fit_success"],
                    "fit_rms_mag": f"{float(record['fit_rms_mag']):.6g}",
                    "fit_mad_mag": f"{float(record['fit_mad_mag']):.6g}",
                    "log10_L_1h": f"{theta[0]:.6g}",
                    "alpha_L": f"{theta[1]:.6g}",
                    "v_cm_s": f"{10.0**theta[2]:.6e}",
                    "color_factor_fT": f"{10.0**theta[3]:.6g}",
                    "initial_radius_scale_fR0": f"{10.0**theta[4]:.6g}",
                }
            )
    return path


def main() -> None:
    records = [fit_model(load_model(model)) for model in MODELS]
    for output in [make_plot(records), write_metrics(records)]:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
