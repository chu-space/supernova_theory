#!/usr/bin/env python3
"""Paper-style statistics plots for the low-Ni WN3 semi-analytic model inputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


WORKSPACE = Path("/Users/arifchu/Documents/Codex/2026-06-09/scientific-purpose-looks-like-you-re")
SNEC_ROOT = Path("/Users/arifchu/Desktop/GitHub/supernova_theory")
RUN_ROOT = SNEC_ROOT / "snec" / "runs" / "model_variants"
OUT_DIR = WORKSPACE / "output" / "low_ni_diagnostics"

SEC_PER_HOUR = 3600.0
R_SUN = 6.957e10
M_SUN = 1.98847e33
PC_CM = 3.0856775814913673e18
TEN_PC_CM = 10.0 * PC_CM
C_LIGHT = 2.99792458e10
H_PLANCK = 6.62607015e-27
K_BOLTZ = 1.380649e-16
LAMBDA_G_CM = 4770.0e-8
NU_G_HZ = C_LIGHT / LAMBDA_G_CM
KAPPA_HE_PROXY = 0.20


@dataclass(frozen=True)
class Model:
    run_id: str
    label: str
    short_label: str
    color: str


MODELS = [
    Model("wn3_sce_bare_e1_ni001", "Bare WN3", "bare", "#2b6cb0"),
    Model(
        "wn3_sce_mass_m0p01_r5_e1_ni001",
        r"$+0.01\,M_\odot$ to $5\,R_\odot$",
        "m0p01_r5",
        "#d97706",
    ),
    Model(
        "wn3_sce_radius_m0p01_r50_e1_ni001",
        r"$+0.01\,M_\odot$ to $50\,R_\odot$",
        "m0p01_r50",
        "#15803d",
    ),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
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
        0.94,
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


def ab_mag_to_nu_lnu(m_ab: np.ndarray) -> np.ndarray:
    f_nu_10pc = 10.0 ** (-0.4 * (m_ab + 48.60))
    l_nu = 4.0 * np.pi * TEN_PC_CM**2 * f_nu_10pc
    return NU_G_HZ * l_nu


def interp_at(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    order = np.argsort(x)
    return float(np.interp(x0, x[order], y[order]))


def optical_depth_proxy(radius_cm: np.ndarray, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(radius_cm)
    radius = radius_cm[order]
    rho = density[order]
    dr = np.diff(radius)
    segment_tau = KAPPA_HE_PROXY * 0.5 * (rho[:-1] + rho[1:]) * np.maximum(dr, 0.0)
    tau = np.zeros_like(radius)
    tau[:-1] = np.cumsum(segment_tau[::-1])[::-1]
    path_cm = np.maximum(radius[-1] - radius, 0.0)
    tdiff_h = tau * path_cm / C_LIGHT / SEC_PER_HOUR
    inverse = np.argsort(order)
    return tau[inverse], tdiff_h[inverse]


def outer_power_law(radius_rsun: np.ndarray, density: np.ndarray, exterior_mass_msun: np.ndarray) -> tuple[float, float, float]:
    mask = (
        (exterior_mass_msun <= 0.01)
        & np.isfinite(radius_rsun)
        & np.isfinite(density)
        & (radius_rsun > 0.0)
        & (density > 0.0)
    )
    if int(np.sum(mask)) < 4:
        return float("nan"), float("nan"), float("nan")
    x = np.log10(radius_rsun[mask])
    y = np.log10(density[mask])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), float(np.nanstd(y - (slope * x + intercept)))


def load_profile(model: Model) -> dict[str, np.ndarray | float | str]:
    data_dir = RUN_ROOT / model.run_id / "Data"
    mass_msun = load_value_column(data_dir / "mass_initial.dat") / M_SUN
    radius_rsun = load_value_column(data_dir / "rad_initial.dat") / R_SUN
    density = load_value_column(data_dir / "rho_initial.dat")
    total_mass_msun = float(np.nanmax(mass_msun))
    exterior_mass_msun = np.clip(total_mass_msun - mass_msun, 1.0e-12, None)
    tau, tdiff_h = optical_depth_proxy(radius_rsun * R_SUN, density)
    slope, intercept, scatter = outer_power_law(radius_rsun, density, exterior_mass_msun)
    idx_base = int(np.nanargmin(np.abs(exterior_mass_msun - 0.01)))

    return {
        "run_id": model.run_id,
        "label": model.label,
        "short_label": model.short_label,
        "color": model.color,
        "mass_msun": mass_msun,
        "radius_rsun": radius_rsun,
        "density": density,
        "total_mass_msun": total_mass_msun,
        "exterior_mass_msun": exterior_mass_msun,
        "radius_star_rsun": float(np.nanmax(radius_rsun)),
        "tau_proxy": tau,
        "tdiff_proxy_h": tdiff_h,
        "outer_slope": slope,
        "outer_intercept": intercept,
        "outer_fit_scatter_dex": scatter,
        "base_outer_radius_rsun": float(radius_rsun[idx_base]),
        "base_outer_tau_proxy": float(tau[idx_base]),
        "base_outer_tdiff_proxy_h": float(tdiff_h[idx_base]),
    }


def load_light_stats(profile: dict[str, np.ndarray | float | str]) -> dict[str, float | str]:
    data_dir = RUN_ROOT / str(profile["run_id"]) / "Data"
    mag = load_table(data_dir / "magnitudes.dat")
    rad_photo = load_table(data_dir / "rad_photo.dat")

    time_h = mag[:, 0] / SEC_PER_HOUR
    teff = mag[:, 1]
    m_g = mag[:, 4]
    nu_lnu_g = ab_mag_to_nu_lnu(m_g)

    keep = (
        np.isfinite(time_h)
        & np.isfinite(teff)
        & np.isfinite(m_g)
        & np.isfinite(nu_lnu_g)
        & (time_h > 0.0)
        & (time_h <= 48.0)
        & (nu_lnu_g > 0.0)
        & (teff > 0.0)
    )
    time_h = time_h[keep]
    teff = teff[keep]
    m_g = m_g[keep]
    nu_lnu_g = nu_lnu_g[keep]
    peak_index = int(np.nanargmax(nu_lnu_g))
    t_peak_h = float(time_h[peak_index])
    peak_nu_lnu = float(nu_lnu_g[peak_index])
    peak_m_g = float(m_g[peak_index])
    peak_teff = float(teff[peak_index])
    r_photo_rsun = interp_at(rad_photo[:, 0] / SEC_PER_HOUR, rad_photo[:, 1] / R_SUN, t_peak_h)

    cooling = (time_h > t_peak_h) & (time_h <= 48.0) & (nu_lnu_g >= 0.1 * peak_nu_lnu)
    if int(np.sum(cooling)) >= 4:
        slope, _ = np.polyfit(np.log10(time_h[cooling]), np.log10(nu_lnu_g[cooling]), 1)
        cooling_slope = float(slope)
    else:
        cooling_slope = float("nan")

    return {
        "run_id": str(profile["run_id"]),
        "label": str(profile["label"]),
        "radius_star_rsun": float(profile["radius_star_rsun"]),
        "time_h": time_h,
        "teff": teff,
        "m_g": m_g,
        "nu_lnu_g": nu_lnu_g,
        "peak_t_h": t_peak_h,
        "peak_m_g": peak_m_g,
        "peak_nu_lnu_g": peak_nu_lnu,
        "peak_teff_k": peak_teff,
        "peak_r_photo_rsun": r_photo_rsun,
        "cooling_slope_log_lg_log_t": cooling_slope,
    }


def plot_structure_statistics(profiles: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
    ax_density, ax_mass, ax_tau, ax_depth = axes.ravel()

    for profile in profiles:
        color = str(profile["color"])
        label = str(profile["label"])
        radius = profile["radius_rsun"]
        density = profile["density"]
        exterior_mass = profile["exterior_mass_msun"]
        tau = profile["tau_proxy"]
        tdiff = profile["tdiff_proxy_h"]

        keep = np.isfinite(radius) & np.isfinite(density) & (radius > 0.0) & (density > 0.0)
        ax_density.plot(radius[keep], density[keep], color=color, lw=2.0, label=label)

        fit_mask = keep & (exterior_mass <= 0.01)
        if int(np.sum(fit_mask)) >= 4 and np.isfinite(float(profile["outer_slope"])):
            x_fit = np.array([np.nanmin(radius[fit_mask]), np.nanmax(radius[fit_mask])])
            y_fit = 10.0 ** (float(profile["outer_slope"]) * np.log10(x_fit) + float(profile["outer_intercept"]))
            ax_density.plot(x_fit, y_fit, color=color, lw=1.35, ls=":")

        mass_keep = np.isfinite(radius) & np.isfinite(exterior_mass) & (radius > 0.0) & (exterior_mass > 1.0e-10)
        ax_mass.plot(radius[mass_keep], exterior_mass[mass_keep], color=color, lw=2.0)

        tau_keep = np.isfinite(radius) & np.isfinite(tau) & (radius > 0.0) & (tau > 0.0)
        ax_tau.plot(radius[tau_keep], tau[tau_keep], color=color, lw=2.0)

        depth_keep = np.isfinite(exterior_mass) & np.isfinite(tdiff) & (exterior_mass > 1.0e-10) & (tdiff > 0.0)
        ax_depth.plot(exterior_mass[depth_keep], tdiff[depth_keep], color=color, lw=2.0)

    ax_density.set_xscale("log")
    ax_density.set_yscale("log")
    ax_density.set_xlabel(r"Radius, $r$ ($R_\odot$)")
    ax_density.set_ylabel(r"Density, $\rho$ (g cm$^{-3}$)")
    model_legend = ax_density.legend(loc="lower left", frameon=False)
    ax_density.add_artist(model_legend)
    panel_label(ax_density, "(a)")

    slope_lines = [
        Line2D(
            [0],
            [0],
            color=str(profile["color"]),
            lw=1.7,
            ls=":",
            label=rf"$n={-float(profile['outer_slope']):.1f}$",
        )
        for profile in profiles
    ]
    ax_density.legend(handles=slope_lines, title=r"outer $0.01\,M_\odot$", loc="upper right", frameon=False)

    ax_mass.set_xscale("log")
    ax_mass.set_yscale("log")
    ax_mass.set_xlabel(r"Radius, $r$ ($R_\odot$)")
    ax_mass.set_ylabel(r"Exterior mass, $M(>r)$ ($M_\odot$)")
    panel_label(ax_mass, "(b)")

    ax_tau.set_xscale("log")
    ax_tau.set_yscale("log")
    ax_tau.set_xlabel(r"Radius, $r$ ($R_\odot$)")
    ax_tau.set_ylabel(r"Optical depth proxy, $\tau(>r)$")
    ax_tau.text(
        0.05,
        0.08,
        rf"$\kappa={KAPPA_HE_PROXY:.2f}$ cm$^2$ g$^{{-1}}$",
        transform=ax_tau.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )
    panel_label(ax_tau, "(c)")

    ax_depth.set_xscale("log")
    ax_depth.set_yscale("log")
    ax_depth.set_xlabel(r"Exterior mass, $M(>r)$ ($M_\odot$)")
    ax_depth.set_ylabel(r"$t_{\rm diff,proxy}=\tau(R_\star-r)/c$ (h)")
    panel_label(ax_depth, "(d)")

    for ax in axes.ravel():
        style_axis(ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / "low_ni_semianalytic_structure_statistics.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_emission_depth_statistics(profiles: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)
    ax_photo, ax_lumshell = axes

    for profile in profiles:
        data_dir = RUN_ROOT / str(profile["run_id"]) / "Data"
        total_mass_g = float(profile["total_mass_msun"]) * M_SUN
        color = str(profile["color"])
        label = str(profile["label"])

        for ax, filename in [(ax_photo, "mass_photo.dat"), (ax_lumshell, "mass_lumshell.dat")]:
            data = load_table(data_dir / filename)
            time_h = data[:, 0] / SEC_PER_HOUR
            exterior_mass = np.clip((total_mass_g - data[:, 1]) / M_SUN, 1.0e-12, None)
            keep = np.isfinite(time_h) & np.isfinite(exterior_mass) & (time_h > 0.0) & (time_h <= 48.0)
            ax.plot(time_h[keep], exterior_mass[keep], color=color, lw=2.0, label=label)

    ax_photo.set_xscale("log")
    ax_photo.set_yscale("log")
    ax_photo.set_xlabel("Time since explosion (h)")
    ax_photo.set_ylabel(r"Mass above photosphere ($M_\odot$)")
    ax_photo.legend(frameon=False, loc="lower right")
    panel_label(ax_photo, "(a)")

    ax_lumshell.set_xscale("log")
    ax_lumshell.set_yscale("log")
    ax_lumshell.set_xlabel("Time since explosion (h)")
    ax_lumshell.set_ylabel(r"Mass above luminosity shell ($M_\odot$)")
    panel_label(ax_lumshell, "(b)")

    for ax in axes:
        ax.set_xlim(0.02, 48.0)
        style_axis(ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / "low_ni_semianalytic_emission_depth_statistics.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_observable_statistics(stats: list[dict[str, float | str | np.ndarray]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0), constrained_layout=True)
    ax_mag, ax_peak, ax_tpeak, ax_temp = axes.ravel()

    radii = np.array([float(item["radius_star_rsun"]) for item in stats])
    peaks = np.array([float(item["peak_nu_lnu_g"]) for item in stats])
    t_peaks = np.array([float(item["peak_t_h"]) for item in stats])
    peak_teffs = np.array([float(item["peak_teff_k"]) for item in stats])
    peak_rphotos = np.array([float(item["peak_r_photo_rsun"]) for item in stats])

    for item, model in zip(stats, MODELS):
        color = model.color
        time_h = item["time_h"]
        m_g = item["m_g"]
        keep = (time_h >= 0.03) & (time_h <= 48.0) & (m_g <= -5.0)
        ax_mag.plot(time_h[keep], m_g[keep], color=color, lw=2.0, label=model.label)
        ax_peak.scatter(float(item["radius_star_rsun"]), float(item["peak_nu_lnu_g"]), s=54, color=color, zorder=3)
        ax_tpeak.scatter(float(item["radius_star_rsun"]), float(item["peak_t_h"]), s=54, color=color, zorder=3)
        ax_temp.scatter(float(item["peak_r_photo_rsun"]), float(item["peak_teff_k"]), s=54, color=color, zorder=3)

    for y, ax, label in [
        (peaks, ax_peak, r"$\nu L_{\nu,g}^{\rm pk}\propto R_\star^{%.2f}$"),
        (t_peaks, ax_tpeak, r"$t_{\rm pk}\propto R_\star^{%.2f}$"),
    ]:
        slope, intercept = np.polyfit(np.log10(radii), np.log10(y), 1)
        x_fit = np.geomspace(np.nanmin(radii), np.nanmax(radii), 100)
        ax.plot(x_fit, 10.0 ** (intercept + slope * np.log10(x_fit)), color="#111827", lw=1.1, ls=":")
        ax.text(0.05, 0.90, label % slope, transform=ax.transAxes, ha="left", va="top", fontsize=9)

    ax_mag.set_xscale("log")
    ax_mag.invert_yaxis()
    ax_mag.set_xlim(0.03, 48.0)
    ax_mag.set_xlabel("Time since explosion (h)")
    ax_mag.set_ylabel(r"Absolute $g$ magnitude, $M_g$")
    ax_mag.legend(frameon=False, loc="upper right")
    panel_label(ax_mag, "(a)")

    ax_peak.set_xscale("log")
    ax_peak.set_yscale("log")
    ax_peak.set_xlabel(r"Initial outer radius, $R_\star$ ($R_\odot$)")
    ax_peak.set_ylabel(r"Peak $\nu L_{\nu,g}$ (erg s$^{-1}$)")
    panel_label(ax_peak, "(b)")

    ax_tpeak.set_xscale("log")
    ax_tpeak.set_yscale("log")
    ax_tpeak.set_xlabel(r"Initial outer radius, $R_\star$ ($R_\odot$)")
    ax_tpeak.set_ylabel(r"$g$-band peak time (h)")
    panel_label(ax_tpeak, "(c)")

    ax_temp.set_xlabel(r"$R_{\rm ph}$ at $g$ peak ($R_\odot$)")
    ax_temp.set_ylabel(r"$T_{\rm eff}$ at $g$ peak (K)")
    ax_temp.set_xlim(0.90 * np.nanmin(peak_rphotos), 1.08 * np.nanmax(peak_rphotos))
    ax_temp.set_ylim(0.95 * np.nanmin(peak_teffs), 1.08 * np.nanmax(peak_teffs))
    for radius, teff, rstar, model in zip(peak_rphotos, peak_teffs, radii, MODELS):
        ax_temp.annotate(
            rf"$R_\star={rstar:.1f}$",
            (radius, teff),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color="#111827",
        )
    panel_label(ax_temp, "(d)")

    for ax in axes.ravel():
        style_axis(ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / "low_ni_semianalytic_observable_statistics.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def write_summary(
    profiles: list[dict[str, np.ndarray | float | str]],
    light_stats: list[dict[str, float | str | np.ndarray]],
) -> Path:
    by_run = {str(item["run_id"]): item for item in light_stats}
    output = OUT_DIR / "low_ni_semianalytic_paper_stats.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "label",
        "R_star_Rsun",
        "M_total_Msun",
        "outer_density_n_fit",
        "outer_fit_scatter_dex",
        "R_at_Mext_0p01_Rsun",
        "tau_proxy_at_Mext_0p01",
        "tdiff_proxy_at_Mext_0p01_h",
        "peak_Mg",
        "peak_time_h",
        "peak_nuLnu_g_erg_s",
        "Teff_at_g_peak_K",
        "Rph_at_g_peak_Rsun",
        "cooling_slope_log_nuLnu_vs_log_t",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            stat = by_run[str(profile["run_id"])]
            writer.writerow(
                {
                    "run_id": profile["run_id"],
                    "label": profile["label"],
                    "R_star_Rsun": f"{float(profile['radius_star_rsun']):.6g}",
                    "M_total_Msun": f"{float(profile['total_mass_msun']):.8g}",
                    "outer_density_n_fit": f"{-float(profile['outer_slope']):.6g}",
                    "outer_fit_scatter_dex": f"{float(profile['outer_fit_scatter_dex']):.6g}",
                    "R_at_Mext_0p01_Rsun": f"{float(profile['base_outer_radius_rsun']):.6g}",
                    "tau_proxy_at_Mext_0p01": f"{float(profile['base_outer_tau_proxy']):.6e}",
                    "tdiff_proxy_at_Mext_0p01_h": f"{float(profile['base_outer_tdiff_proxy_h']):.6e}",
                    "peak_Mg": f"{float(stat['peak_m_g']):.6g}",
                    "peak_time_h": f"{float(stat['peak_t_h']):.6g}",
                    "peak_nuLnu_g_erg_s": f"{float(stat['peak_nu_lnu_g']):.6e}",
                    "Teff_at_g_peak_K": f"{float(stat['peak_teff_k']):.6g}",
                    "Rph_at_g_peak_Rsun": f"{float(stat['peak_r_photo_rsun']):.6g}",
                    "cooling_slope_log_nuLnu_vs_log_t": f"{float(stat['cooling_slope_log_lg_log_t']):.6g}",
                }
            )
    return output


def main() -> None:
    setup_style()
    profiles = [load_profile(model) for model in MODELS]
    light_stats = [load_light_stats(profile) for profile in profiles]
    outputs = [
        plot_structure_statistics(profiles),
        plot_emission_depth_statistics(profiles),
        plot_observable_statistics(light_stats),
        write_summary(profiles, light_stats),
    ]
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
