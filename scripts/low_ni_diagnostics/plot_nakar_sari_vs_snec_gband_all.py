#!/usr/bin/env python3
"""Compare Nakar/Sari-style g-band predictions with all three SNEC WN3 runs."""

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

# --- CONSTANTS (cgs units, matching the working notebook form) ---
M_sun = 1.989e33
R_sun = 6.957e10
E_51_scale = 1e51
sigma_SB = 5.67e-5
h = 6.626e-27
k_B = 1.38e-16
c = 3.0e10
pc = 3.086e18

# g-band effective frequency, lambda_eff ~ 4770 Angstrom.
lam_g = 4770e-8
nu_g = c / lam_g


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
    exponent = np.clip((h * nu_g) / (k_B * temperature), None, 700.0)
    return (2.0 * h * nu_g**3 / c**2) / np.expm1(exponent)


def load_snec_model(model: Model) -> dict[str, np.ndarray | float | str]:
    data_dir = RUN_ROOT / model.run_id / "Data"
    mag = load_table(data_dir / "magnitudes.dat")
    mass_g = load_value_column(data_dir / "mass_initial.dat")
    radius_cm = load_value_column(data_dir / "rad_initial.dat")

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

    return {
        "run_id": model.run_id,
        "label": model.label,
        "color": model.color,
        "time_h": time_h[keep],
        "m_g": m_g[keep],
        "nu_lnu_g": nu_lnu[keep],
        "m_ej_g": float(np.nanmax(mass_g)),
        "r_star_cm": float(np.nanmax(radius_cm)),
    }


def nakar_sari_bolometric_luminosity(
    t_sec: np.ndarray,
    m_ej_g: float,
    r_star_cm: float,
    e_kin_erg: float,
) -> tuple[np.ndarray, float]:
    m_15 = m_ej_g / (15.0 * M_sun)
    r_5 = r_star_cm / (5.0 * R_sun)
    e_51 = e_kin_erg / E_51_scale

    t_s = 90.0 * (m_15**0.41) * (r_5**1.33) * (e_51**-0.58)
    t_min = t_sec / 60.0
    t_hr = t_sec / 3600.0
    planar = 2.0e42 * (m_15**-0.33) * (r_5**2.3) * (e_51**0.34) * (t_min ** (-4.0 / 3.0))
    spherical = 3.5e41 * (m_15**-0.73) * r_5 * (e_51**0.91) * (t_hr**-0.35)
    return np.where(t_sec < t_s, planar, spherical), t_s


def analytic_photospheric_radius(t_sec: np.ndarray, r_star_cm: float, t_s: float) -> np.ndarray:
    return np.where(t_sec < t_s, r_star_cm, r_star_cm * (t_sec / t_s) ** 0.725)


def analytic_g_band_curve(
    t_sec: np.ndarray,
    m_ej_g: float,
    r_star_cm: float,
    e_kin_erg: float = E_51_scale,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    l_bol, t_s = nakar_sari_bolometric_luminosity(t_sec, m_ej_g, r_star_cm, e_kin_erg)
    radius = analytic_photospheric_radius(t_sec, r_star_cm, t_s)
    t_eff = (l_bol / (4.0 * np.pi * sigma_SB * radius**2)) ** 0.25
    l_nu = 4.0 * np.pi**2 * radius**2 * planck_bnu(t_eff)
    m_g = lnu_to_ab_mag(l_nu)
    return m_g, nu_g * l_nu, radius, t_eff, np.full_like(t_sec, t_s)


def comparison_record(record: dict[str, np.ndarray | float | str]) -> dict[str, np.ndarray | float | str]:
    t_sec = np.linspace(60.0, 48.0 * 3600.0, 2400)
    t_h = t_sec / 3600.0
    m_ns, nu_lnu_ns, radius_ns, teff_ns, t_s = analytic_g_band_curve(
        t_sec,
        float(record["m_ej_g"]),
        float(record["r_star_cm"]),
    )

    time_snec = record["time_h"]
    assert isinstance(time_snec, np.ndarray)
    m_snec = record["m_g"]
    nu_lnu_snec = record["nu_lnu_g"]
    assert isinstance(m_snec, np.ndarray)
    assert isinstance(nu_lnu_snec, np.ndarray)

    m_ns_at_snec = np.interp(time_snec, t_h, m_ns)
    nu_lnu_ns_at_snec = np.interp(time_snec, t_h, nu_lnu_ns)
    ratio = nu_lnu_ns_at_snec / nu_lnu_snec
    bright = (
        (time_snec >= 0.03)
        & (m_snec <= -5.0)
        & np.isfinite(ratio)
        & (ratio > 0.0)
        & np.isfinite(m_ns_at_snec)
    )

    return {
        **record,
        "theory_time_h": t_h,
        "theory_m_g": m_ns,
        "theory_nu_lnu_g": nu_lnu_ns,
        "theory_radius_rsun": radius_ns / R_sun,
        "theory_teff": teff_ns,
        "transition_time_s": float(t_s[0]),
        "m_g_theory_at_snec": m_ns_at_snec,
        "nu_lnu_theory_at_snec": nu_lnu_ns_at_snec,
        "ratio": ratio,
        "bright_mask": bright,
    }


def make_plot(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.2,
            "axes.labelsize": 10.8,
            "axes.titlesize": 10.8,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.9,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(12.4, 12.2), constrained_layout=True)

    for row, record in enumerate(records):
        ax_mag, ax_ratio = axes[row]
        color = str(record["color"])
        label = str(record["label"])
        time_h = record["time_h"]
        m_snec = record["m_g"]
        ratio = record["ratio"]
        bright = record["bright_mask"]
        theory_time_h = record["theory_time_h"]
        theory_m_g = record["theory_m_g"]

        assert isinstance(time_h, np.ndarray)
        assert isinstance(m_snec, np.ndarray)
        assert isinstance(ratio, np.ndarray)
        assert isinstance(bright, np.ndarray)
        assert isinstance(theory_time_h, np.ndarray)
        assert isinstance(theory_m_g, np.ndarray)

        model_keep = (theory_time_h >= 0.03) & (theory_time_h <= 48.0) & np.isfinite(theory_m_g) & (theory_m_g <= -5.0)
        ax_mag.plot(time_h[bright], m_snec[bright], color=color, lw=2.0, label="SNEC g band")
        ax_mag.plot(theory_time_h[model_keep], theory_m_g[model_keep], color="#111827", lw=1.9, ls=":", label="Nakar/Sari + Planck")
        ax_mag.set_xscale("log")
        ax_mag.invert_yaxis()
        ax_mag.set_ylabel(r"$M_g$")
        ax_mag.set_title(label)
        if row == 0:
            ax_mag.legend(frameon=False, loc="upper right")
        panel_label(ax_mag, f"({chr(97 + 2 * row)})")
        style_axis(ax_mag)

        ax_ratio.axhline(1.0, color="#111827", lw=1.0, alpha=0.8)
        ax_ratio.axhspan(0.5, 2.0, color="#9ca3af", alpha=0.16, lw=0)
        ax_ratio.plot(time_h[bright], ratio[bright], color=color, lw=2.0)
        ax_ratio.set_xscale("log")
        ax_ratio.set_yscale("log")
        ax_ratio.set_ylim(0.03, 40.0)
        ax_ratio.set_ylabel(r"Theory / SNEC")
        ax_ratio.set_title(r"Ratio in $\nu L_{\nu,g}$")
        panel_label(ax_ratio, f"({chr(98 + 2 * row)})")
        style_axis(ax_ratio)

    for ax in axes[-1]:
        ax.set_xlabel("Time since explosion (h)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / "low_ni_nakar_sari_vs_snec_gband_ratio_all_models.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def write_csv(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    output = OUT_DIR / "low_ni_nakar_sari_vs_snec_gband_ratio_all_models.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "label",
                "R_star_Rsun",
                "M_ej_Msun",
                "transition_time_s",
                "time_h",
                "M_g_snec",
                "M_g_nakar_sari",
                "nuLnu_g_snec_erg_s",
                "nuLnu_g_nakar_sari_erg_s",
                "ratio_nakar_sari_to_snec",
            ],
        )
        writer.writeheader()
        for record in records:
            time_h = record["time_h"]
            bright = record["bright_mask"]
            assert isinstance(time_h, np.ndarray)
            assert isinstance(bright, np.ndarray)
            for values in zip(
                time_h[bright],
                record["m_g"][bright],
                record["m_g_theory_at_snec"][bright],
                record["nu_lnu_g"][bright],
                record["nu_lnu_theory_at_snec"][bright],
                record["ratio"][bright],
            ):
                writer.writerow(
                    {
                        "run_id": record["run_id"],
                        "label": record["label"],
                        "R_star_Rsun": f"{float(record['r_star_cm']) / R_sun:.8e}",
                        "M_ej_Msun": f"{float(record['m_ej_g']) / M_sun:.8e}",
                        "transition_time_s": f"{float(record['transition_time_s']):.8e}",
                        "time_h": f"{values[0]:.8e}",
                        "M_g_snec": f"{values[1]:.8e}",
                        "M_g_nakar_sari": f"{values[2]:.8e}",
                        "nuLnu_g_snec_erg_s": f"{values[3]:.8e}",
                        "nuLnu_g_nakar_sari_erg_s": f"{values[4]:.8e}",
                        "ratio_nakar_sari_to_snec": f"{values[5]:.8e}",
                    }
                )
    return output


def write_summary(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    output = OUT_DIR / "low_ni_nakar_sari_vs_snec_gband_ratio_summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "label",
                "R_star_Rsun",
                "M_ej_Msun",
                "transition_time_s",
                "median_ratio",
                "min_ratio",
                "max_ratio",
            ],
        )
        writer.writeheader()
        for record in records:
            ratio = record["ratio"]
            bright = record["bright_mask"]
            assert isinstance(ratio, np.ndarray)
            assert isinstance(bright, np.ndarray)
            good = ratio[bright & np.isfinite(ratio) & (ratio > 0.0)]
            writer.writerow(
                {
                    "run_id": record["run_id"],
                    "label": record["label"],
                    "R_star_Rsun": f"{float(record['r_star_cm']) / R_sun:.6g}",
                    "M_ej_Msun": f"{float(record['m_ej_g']) / M_sun:.6g}",
                    "transition_time_s": f"{float(record['transition_time_s']):.6g}",
                    "median_ratio": f"{float(np.nanmedian(good)):.6g}",
                    "min_ratio": f"{float(np.nanmin(good)):.6g}",
                    "max_ratio": f"{float(np.nanmax(good)):.6g}",
                }
            )
    return output


def main() -> None:
    records = [comparison_record(load_snec_model(model)) for model in MODELS]
    for output in [make_plot(records), write_csv(records), write_summary(records)]:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
