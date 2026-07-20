#!/usr/bin/env python3
"""Root-find the tau=2/3 photosphere and check for radius turnover."""

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

R_SUN = 6.957e10
SEC_PER_HOUR = 3600.0
TAU_PHOTOSPHERE = 2.0 / 3.0
E_51_SCALE = 1.0e51
M_SUN = 1.989e33


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
    """Parse SNEC .xg blocks into {time_s: (mass_coordinate, value)}."""
    blocks: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    current_time: float | None = None
    rows: list[tuple[float, float]] = []

    def flush() -> None:
        nonlocal rows, current_time
        if current_time is None or not rows:
            rows = []
            return
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
    """Find R where tau(R)=target using a bracketed log-linear interpolation."""
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

    # Choose the outermost crossing, which is the photospheric surface.
    idx = int(crossings[-1])
    r0, r1 = radius[idx], radius[idx + 1]
    tau0, tau1 = tau_values[idx], tau_values[idx + 1]
    if tau0 == tau1:
        return float(0.5 * (r0 + r1))
    if tau0 > 0.0 and tau1 > 0.0 and target > 0.0:
        x0, x1 = np.log(tau0), np.log(tau1)
        frac = (np.log(target) - x0) / (x1 - x0)
    else:
        frac = (target - tau0) / (tau1 - tau0)
    return float(r0 + np.clip(frac, 0.0, 1.0) * (r1 - r0))


def analytic_power_law_radius(time_s: np.ndarray, r_star_cm: float, m_ej_g: float) -> tuple[np.ndarray, float]:
    m_15 = m_ej_g / (15.0 * M_SUN)
    r_5 = r_star_cm / (5.0 * R_SUN)
    t_s = 90.0 * (m_15**0.41) * (r_5**1.33)
    radius = np.where(time_s < t_s, r_star_cm, r_star_cm * (time_s / t_s) ** 0.725)
    return radius, float(t_s)


def derivative_roots(time_h: np.ndarray, radius_rsun: np.ndarray) -> list[float]:
    keep = np.isfinite(time_h) & np.isfinite(radius_rsun) & (time_h > 0.0) & (radius_rsun > 0.0)
    t = time_h[keep]
    r = radius_rsun[keep]
    if len(t) < 4:
        return []
    drdt = np.gradient(r, t)
    roots: list[float] = []
    for idx in np.where(drdt[:-1] * drdt[1:] <= 0.0)[0]:
        if drdt[idx] > 0.0 and drdt[idx + 1] <= 0.0:
            denom = drdt[idx + 1] - drdt[idx]
            frac = 0.0 if denom == 0.0 else -drdt[idx] / denom
            roots.append(float(t[idx] + np.clip(frac, 0.0, 1.0) * (t[idx + 1] - t[idx])))
    return roots


def load_model(model: Model) -> dict[str, object]:
    data_dir = RUN_ROOT / model.run_id / "Data"
    tau_blocks = parse_xg_blocks(data_dir / "tau.xg")
    radius_blocks = parse_xg_blocks(data_dir / "radius.xg")
    common_times = sorted(set(tau_blocks).intersection(radius_blocks))

    root_times_h: list[float] = []
    root_radius_rsun: list[float] = []
    snec_radius_at_root_rsun: list[float] = []
    rad_photo = load_table(data_dir / "rad_photo.dat")
    photo_time_h = rad_photo[:, 0] / SEC_PER_HOUR
    photo_radius_rsun = rad_photo[:, 1] / R_SUN
    valid_photo = (photo_time_h > 0.0) & (photo_radius_rsun > 0.0) & np.isfinite(photo_radius_rsun)

    for time_s in common_times:
        _, tau = tau_blocks[time_s]
        _, radius_cm = radius_blocks[time_s]
        root_radius_cm = root_radius_for_tau(radius_cm, tau)
        if not np.isfinite(root_radius_cm):
            continue
        root_times_h.append(time_s / SEC_PER_HOUR)
        root_radius_rsun.append(root_radius_cm / R_SUN)
        snec_radius_at_root_rsun.append(float(np.interp(time_s / SEC_PER_HOUR, photo_time_h[valid_photo], photo_radius_rsun[valid_photo])))

    r_star_cm = float(np.nanmax(load_value_column(data_dir / "rad_initial.dat")))
    m_ej_g = float(np.nanmax(load_value_column(data_dir / "mass_initial.dat")))
    analytic_radius_cm, transition_s = analytic_power_law_radius(rad_photo[:, 0], r_star_cm, m_ej_g)
    local_derivative_roots_h = derivative_roots(photo_time_h[valid_photo], photo_radius_rsun[valid_photo])

    return {
        "model": model,
        "photo_time_h": photo_time_h[valid_photo],
        "photo_radius_rsun": photo_radius_rsun[valid_photo],
        "analytic_radius_rsun": analytic_radius_cm[valid_photo] / R_SUN,
        "root_times_h": np.asarray(root_times_h),
        "root_radius_rsun": np.asarray(root_radius_rsun),
        "snec_radius_at_root_rsun": np.asarray(snec_radius_at_root_rsun),
        "local_derivative_roots_h": local_derivative_roots_h,
        "transition_s": transition_s,
        "r_star_rsun": r_star_cm / R_SUN,
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
        time_h = record["photo_time_h"]
        radius = record["photo_radius_rsun"]
        analytic_radius = record["analytic_radius_rsun"]
        root_time = record["root_times_h"]
        root_radius = record["root_radius_rsun"]
        derivative_roots_h = record["local_derivative_roots_h"]
        assert isinstance(time_h, np.ndarray)
        assert isinstance(radius, np.ndarray)
        assert isinstance(analytic_radius, np.ndarray)
        assert isinstance(root_time, np.ndarray)
        assert isinstance(root_radius, np.ndarray)
        assert isinstance(derivative_roots_h, list)

        ax_r, ax_d = axes[row]
        ax_r.plot(time_h, radius, color=color, lw=2.0, label="SNEC rad_photo.dat")
        ax_r.scatter(root_time, root_radius, s=26, facecolor="white", edgecolor="#111827", linewidth=0.9, zorder=4, label=r"root: $\tau=2/3$")
        ax_r.plot(time_h, analytic_radius, color="#111827", lw=1.7, ls=":", label=r"old $R_\star(t/t_s)^{0.725}$")
        ax_r.set_xscale("log")
        ax_r.set_yscale("log")
        ax_r.set_ylabel(r"$R_{\rm ph}$ ($R_\odot$)")
        ax_r.set_title(model.label)
        if row == 0:
            ax_r.legend(frameon=False, loc="upper left")
        style_axis(ax_r)

        drdt = np.gradient(radius, time_h)
        ax_d.axhline(0.0, color="#111827", lw=0.9)
        ax_d.plot(time_h, drdt, color=color, lw=1.9)
        for root_h in derivative_roots_h:
            ax_d.axvline(root_h, color="#111827", lw=0.9, ls=":", alpha=0.85)
        if derivative_roots_h:
            ax_d.text(
                0.05,
                0.86,
                r"local $dR_{\rm ph}/dt=0$ crossings",
                transform=ax_d.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
            )
        else:
            ax_d.text(
                0.05,
                0.86,
                r"no local $dR_{\rm ph}/dt=0$ crossing",
                transform=ax_d.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
            )
        ax_d.set_xscale("log")
        ax_d.set_ylabel(r"$dR_{\rm ph}/dt$ ($R_\odot$ h$^{-1}$)")
        ax_d.set_title(r"$dR_{\rm ph}/dt$ crossing check")
        style_axis(ax_d)

    for ax in axes[-1]:
        ax.set_xlabel("Time since explosion (h)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / "low_ni_tau_root_photosphere_turnover.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def write_summary(records: list[dict[str, object]]) -> tuple[Path, Path]:
    root_output = OUT_DIR / "low_ni_tau_root_photosphere_points.csv"
    summary_output = OUT_DIR / "low_ni_tau_root_photosphere_summary.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with root_output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_id", "label", "time_h", "R_root_Rsun", "R_snec_interp_Rsun", "frac_difference"],
        )
        writer.writeheader()
        for record in records:
            model = record["model"]
            assert isinstance(model, Model)
            root_time = record["root_times_h"]
            root_radius = record["root_radius_rsun"]
            snec_radius = record["snec_radius_at_root_rsun"]
            assert isinstance(root_time, np.ndarray)
            assert isinstance(root_radius, np.ndarray)
            assert isinstance(snec_radius, np.ndarray)
            for time_h, r_root, r_snec in zip(root_time, root_radius, snec_radius):
                writer.writerow(
                    {
                        "run_id": model.run_id,
                        "label": model.label,
                        "time_h": f"{time_h:.8e}",
                        "R_root_Rsun": f"{r_root:.8e}",
                        "R_snec_interp_Rsun": f"{r_snec:.8e}",
                        "frac_difference": f"{((r_root - r_snec) / r_snec):.8e}" if r_snec != 0.0 else "nan",
                    }
                )

    with summary_output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "label",
                "R_star_Rsun",
                "N_tau_roots",
                "local_dRdt_root_times_h",
                "sustained_recession_detected",
                "max_Rph_Rsun",
                "max_Rph_time_h",
            ],
        )
        writer.writeheader()
        for record in records:
            model = record["model"]
            assert isinstance(model, Model)
            time_h = record["photo_time_h"]
            radius = record["photo_radius_rsun"]
            root_time = record["root_times_h"]
            derivative_roots_h = record["local_derivative_roots_h"]
            assert isinstance(time_h, np.ndarray)
            assert isinstance(radius, np.ndarray)
            assert isinstance(root_time, np.ndarray)
            assert isinstance(derivative_roots_h, list)
            max_index = int(np.nanargmax(radius))
            sustained_recession = bool(max_index < len(radius) - 1 and np.nanmedian(np.gradient(radius[max_index:], time_h[max_index:])) < 0.0)
            writer.writerow(
                {
                    "run_id": model.run_id,
                    "label": model.label,
                    "R_star_Rsun": f"{float(record['r_star_rsun']):.6g}",
                    "N_tau_roots": str(len(root_time)),
                    "local_dRdt_root_times_h": ";".join(f"{value:.6g}" for value in derivative_roots_h) if derivative_roots_h else "none in 0-48h",
                    "sustained_recession_detected": "yes" if sustained_recession else "no",
                    "max_Rph_Rsun": f"{radius[max_index]:.6g}",
                    "max_Rph_time_h": f"{time_h[max_index]:.6g}",
                }
            )
    return root_output, summary_output


def main() -> None:
    records = [load_model(model) for model in MODELS]
    outputs = [make_plot(records), *write_summary(records)]
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
