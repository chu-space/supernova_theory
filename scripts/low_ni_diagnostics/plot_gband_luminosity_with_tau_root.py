#!/usr/bin/env python3
"""Plot SNEC g-band luminosity against Nakar/Sari + tau-root R_ph conversion."""

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
TAU_PHOTOSPHERE = 2.0 / 3.0

LAMBDA_G_CM = 4770e-8
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


def load_value_column(path: Path) -> np.ndarray:
    data = load_table(path)
    return data[:, 1] if data.shape[1] > 1 else data[:, 0]


def parse_xg_blocks(path: Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    blocks: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    current_time: float | None = None
    rows: list[tuple[float, float]] = []

    def flush() -> None:
        nonlocal rows, current_time
        if current_time is not None and rows:
            data = np.asarray(rows, dtype=float)
            blocks[current_time] = (data[:, 0], data[:, 1])
        rows = []

    with path.open() as handle:
        for raw in handle:
            line = raw.strip().strip('"').strip()
            if not line:
                continue
            if line.startswith("Time ="):
                flush()
                current_time = float(line.split("=", 1)[1])
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))
    flush()
    return blocks


def root_radius_for_tau(radius_cm: np.ndarray, tau: np.ndarray, target: float = TAU_PHOTOSPHERE) -> float:
    keep = np.isfinite(radius_cm) & np.isfinite(tau) & (radius_cm > 0.0) & (tau >= 0.0)
    radius = radius_cm[keep]
    tau_values = tau[keep]
    if len(radius) < 2:
        return float("nan")

    order = np.argsort(radius)
    radius = radius[order]
    tau_values = tau_values[order]
    diff = tau_values - target
    crossings = np.where(diff[:-1] * diff[1:] <= 0.0)[0]
    if len(crossings) == 0:
        return float("nan")

    idx = int(crossings[-1])
    r0, r1 = radius[idx], radius[idx + 1]
    tau0, tau1 = tau_values[idx], tau_values[idx + 1]
    if tau0 == tau1:
        return float(0.5 * (r0 + r1))
    if tau0 > 0.0 and tau1 > 0.0:
        frac = (np.log(target) - np.log(tau0)) / (np.log(tau1) - np.log(tau0))
    else:
        frac = (target - tau0) / (tau1 - tau0)
    return float(r0 + np.clip(frac, 0.0, 1.0) * (r1 - r0))


def ab_mag_to_nu_lnu(m_ab: np.ndarray) -> np.ndarray:
    f_nu_10pc = 10.0 ** (-0.4 * (m_ab + 48.60))
    l_nu = 4.0 * np.pi * (10.0 * PC_CM) ** 2 * f_nu_10pc
    return NU_G * l_nu


def planck_bnu(temperature: np.ndarray) -> np.ndarray:
    exponent = np.clip((H_PLANCK * NU_G) / (K_B * temperature), None, 700.0)
    return (2.0 * H_PLANCK * NU_G**3 / C_LIGHT**2) / np.expm1(exponent)


def nakar_sari_lbol(time_s: np.ndarray, m_ej_g: float, r_star_cm: float, e_kin_erg: float = E_51_SCALE) -> np.ndarray:
    m_15 = m_ej_g / (15.0 * M_SUN)
    r_5 = r_star_cm / (5.0 * R_SUN)
    e_51 = e_kin_erg / E_51_SCALE
    t_s = 90.0 * (m_15**0.41) * (r_5**1.33) * (e_51**-0.58)
    t_min = time_s / 60.0
    t_hr = time_s / SEC_PER_HOUR
    planar = 2.0e42 * (m_15**-0.33) * (r_5**2.3) * (e_51**0.34) * (t_min ** (-4.0 / 3.0))
    spherical = 3.5e41 * (m_15**-0.73) * r_5 * (e_51**0.91) * (t_hr**-0.35)
    return np.where(time_s < t_s, planar, spherical)


def gband_from_lbol_and_radius(l_bol: np.ndarray, radius_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_eff = (l_bol / (4.0 * np.pi * SIGMA_SB * radius_cm**2)) ** 0.25
    l_nu = 4.0 * np.pi**2 * radius_cm**2 * planck_bnu(t_eff)
    return NU_G * l_nu, t_eff


def load_model(model: Model) -> dict[str, object]:
    data_dir = RUN_ROOT / model.run_id / "Data"
    mag = load_table(data_dir / "magnitudes.dat")
    snec_time_h = mag[:, 0] / SEC_PER_HOUR
    snec_nu_lnu = ab_mag_to_nu_lnu(mag[:, 4])
    keep = (snec_time_h > 0.0) & (snec_time_h <= 48.0) & np.isfinite(snec_nu_lnu) & (snec_nu_lnu > 0.0)

    tau_blocks = parse_xg_blocks(data_dir / "tau.xg")
    radius_blocks = parse_xg_blocks(data_dir / "radius.xg")
    root_time_s: list[float] = []
    root_radius_cm: list[float] = []
    for time_s in sorted(set(tau_blocks).intersection(radius_blocks)):
        if time_s <= 0.0 or time_s > 48.0 * SEC_PER_HOUR:
            continue
        _, tau = tau_blocks[time_s]
        _, radius_cm = radius_blocks[time_s]
        r_root = root_radius_for_tau(radius_cm, tau)
        if np.isfinite(r_root) and r_root > 0.0:
            root_time_s.append(time_s)
            root_radius_cm.append(r_root)

    root_time_s_arr = np.asarray(root_time_s)
    root_radius_cm_arr = np.asarray(root_radius_cm)
    m_ej_g = float(np.nanmax(load_value_column(data_dir / "mass_initial.dat")))
    r_star_cm = float(np.nanmax(load_value_column(data_dir / "rad_initial.dat")))
    l_bol_root = nakar_sari_lbol(root_time_s_arr, m_ej_g, r_star_cm)
    root_nu_lnu, root_teff = gband_from_lbol_and_radius(l_bol_root, root_radius_cm_arr)
    snec_at_root = np.interp(root_time_s_arr / SEC_PER_HOUR, snec_time_h[keep], snec_nu_lnu[keep])

    return {
        "model": model,
        "snec_time_h": snec_time_h[keep],
        "snec_nu_lnu": snec_nu_lnu[keep],
        "root_time_h": root_time_s_arr / SEC_PER_HOUR,
        "root_radius_rsun": root_radius_cm_arr / R_SUN,
        "root_nu_lnu": root_nu_lnu,
        "root_teff": root_teff,
        "ratio": root_nu_lnu / snec_at_root,
        "snec_at_root": snec_at_root,
    }


def make_plot(records: list[dict[str, object]]) -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 10.8,
            "axes.titlesize": 10.8,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.9,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 12.0), constrained_layout=True)

    for row, record in enumerate(records):
        model = record["model"]
        assert isinstance(model, Model)
        color = model.color
        snec_time_h = record["snec_time_h"]
        snec_nu_lnu = record["snec_nu_lnu"]
        root_time_h = record["root_time_h"]
        root_nu_lnu = record["root_nu_lnu"]
        ratio = record["ratio"]
        assert isinstance(snec_time_h, np.ndarray)
        assert isinstance(snec_nu_lnu, np.ndarray)
        assert isinstance(root_time_h, np.ndarray)
        assert isinstance(root_nu_lnu, np.ndarray)
        assert isinstance(ratio, np.ndarray)

        ax_lum, ax_ratio = axes[row]
        bright = (snec_time_h >= 0.03) & (snec_nu_lnu > 0.0)
        ax_lum.plot(snec_time_h[bright], snec_nu_lnu[bright], color=color, lw=2.0, label="SNEC g band")
        ax_lum.plot(
            root_time_h,
            root_nu_lnu,
            color="#111827",
            lw=1.4,
            ls=":",
            marker="o",
            ms=4.7,
            mfc="white",
            mec="#111827",
            label=r"Nakar/Sari + root $R_{\rm ph}$",
        )
        ax_lum.set_xscale("log")
        ax_lum.set_yscale("log")
        ax_lum.set_ylabel(r"$\nu_gL_{\nu,g}$ (erg s$^{-1}$)")
        ax_lum.set_title(model.label)
        if row == 0:
            ax_lum.legend(frameon=False, loc="lower right")
        style_axis(ax_lum)

        ax_ratio.axhline(1.0, color="#111827", lw=0.9)
        ax_ratio.axhspan(0.5, 2.0, color="#9ca3af", alpha=0.16, lw=0)
        ax_ratio.plot(root_time_h, ratio, color=color, lw=1.5, marker="o", ms=4.6)
        ax_ratio.set_xscale("log")
        ax_ratio.set_yscale("log")
        ax_ratio.set_ylim(0.03, 40.0)
        ax_ratio.set_ylabel("Theory / SNEC")
        ax_ratio.set_title(r"Ratio at $\tau=2/3$ roots")
        style_axis(ax_ratio)

    for ax in axes[-1]:
        ax.set_xlabel("Time since explosion (h)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / "low_ni_gband_luminosity_tau_root_comparison.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def write_csv(records: list[dict[str, object]]) -> tuple[Path, Path]:
    points_path = OUT_DIR / "low_ni_gband_luminosity_tau_root_points.csv"
    summary_path = OUT_DIR / "low_ni_gband_luminosity_tau_root_summary.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with points_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "label",
                "time_h",
                "R_tau_root_Rsun",
                "Teff_theory_K",
                "nuLnu_g_theory_erg_s",
                "nuLnu_g_snec_erg_s",
                "ratio_theory_to_snec",
            ],
        )
        writer.writeheader()
        for record in records:
            model = record["model"]
            assert isinstance(model, Model)
            for values in zip(
                record["root_time_h"],
                record["root_radius_rsun"],
                record["root_teff"],
                record["root_nu_lnu"],
                record["snec_at_root"],
                record["ratio"],
            ):
                writer.writerow(
                    {
                        "run_id": model.run_id,
                        "label": model.label,
                        "time_h": f"{values[0]:.8e}",
                        "R_tau_root_Rsun": f"{values[1]:.8e}",
                        "Teff_theory_K": f"{values[2]:.8e}",
                        "nuLnu_g_theory_erg_s": f"{values[3]:.8e}",
                        "nuLnu_g_snec_erg_s": f"{values[4]:.8e}",
                        "ratio_theory_to_snec": f"{values[5]:.8e}",
                    }
                )

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "label", "median_ratio", "min_ratio", "max_ratio"])
        writer.writeheader()
        for record in records:
            model = record["model"]
            assert isinstance(model, Model)
            ratio = record["ratio"]
            assert isinstance(ratio, np.ndarray)
            good = ratio[np.isfinite(ratio) & (ratio > 0.0)]
            writer.writerow(
                {
                    "run_id": model.run_id,
                    "label": model.label,
                    "median_ratio": f"{float(np.nanmedian(good)):.6g}",
                    "min_ratio": f"{float(np.nanmin(good)):.6g}",
                    "max_ratio": f"{float(np.nanmax(good)):.6g}",
                }
            )
    return points_path, summary_path


def main() -> None:
    records = [load_model(model) for model in MODELS]
    outputs = [make_plot(records), *write_csv(records)]
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
