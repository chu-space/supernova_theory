#!/usr/bin/env python3
"""Make log density-radius and X_i-versus-element plots for low-Ni WN3 models."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SNEC_ROOT = Path("/Users/arifchu/Desktop/GitHub/supernova_theory")
RUN_ROOT = SNEC_ROOT / "snec" / "runs" / "model_variants"
OUT_DIR = Path("output/low_ni_diagnostics")

R_SUN = 6.957e10
M_SUN = 1.98847e33


@dataclass(frozen=True)
class Model:
    run_id: str
    label: str
    short_label: str
    color: str


MODELS = [
    Model("wn3_sce_bare_e1_ni001", "Bare WN3", "bare_wn3", "#2b6cb0"),
    Model("wn3_sce_mass_m0p01_r5_e1_ni001", r"+0.01 $M_\odot$ to 5 $R_\odot$", "m0p01_r5", "#d97706"),
    Model("wn3_sce_radius_m0p01_r50_e1_ni001", r"+0.01 $M_\odot$ to 50 $R_\odot$", "m0p01_r50", "#15803d"),
]

ELEMENTS = ["H", "He", "C", "O", "Other metals", "Ni"]
ELEMENT_COLORS = {
    "H": "#7dd3fc",
    "He": "#f59e0b",
    "C": "#111827",
    "O": "#2563eb",
    "Other metals": "#a855f7",
    "Ni": "#dc2626",
}


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


def load_value_column(path: Path) -> np.ndarray:
    data = load_table(path)
    return data[:, 1] if data.shape[1] > 1 else data[:, 0]


def shell_weights(enclosed_mass_msun: np.ndarray) -> np.ndarray:
    """Approximate mass weights for profile zones."""
    edges = np.empty(len(enclosed_mass_msun) + 1)
    edges[1:-1] = 0.5 * (enclosed_mass_msun[:-1] + enclosed_mass_msun[1:])
    edges[0] = enclosed_mass_msun[0]
    edges[-1] = enclosed_mass_msun[-1]
    return np.clip(np.diff(edges), 0.0, None)


def load_model(model: Model) -> dict[str, np.ndarray | float | str]:
    data_dir = RUN_ROOT / model.run_id / "Data"
    mass = load_value_column(data_dir / "mass_initial.dat") / M_SUN
    radius = load_value_column(data_dir / "rad_initial.dat") / R_SUN
    density = load_value_column(data_dir / "rho_initial.dat")
    h = load_value_column(data_dir / "H_init_frac.dat")
    he = load_value_column(data_dir / "He_init_frac.dat")
    c = load_value_column(data_dir / "C_init_frac.dat")
    o = load_value_column(data_dir / "O_init_frac.dat")
    ni = load_value_column(data_dir / "Ni_init_frac.dat")
    other = np.clip(1.0 - (h + he + c + o + ni), 0.0, 1.0)
    total_mass = float(np.nanmax(mass))
    exterior_mass = np.clip(total_mass - mass, 1.0e-10, None)

    return {
        "run_id": model.run_id,
        "label": model.label,
        "short_label": model.short_label,
        "color": model.color,
        "mass": mass,
        "radius": radius,
        "density": density,
        "M_ext": exterior_mass,
        "H": h,
        "He": he,
        "C": c,
        "O": o,
        "Ni": ni,
        "Other metals": other,
        "M_total_Msun": total_mass,
        "R_star_Rsun": float(np.nanmax(radius)),
    }


def outer_average(record: dict[str, np.ndarray | float | str], element: str, outer_mass_msun: float = 0.01) -> float:
    mask = record["M_ext"] <= outer_mass_msun
    weights = shell_weights(record["mass"])[mask]
    values = record[element][mask]
    if np.nansum(weights) <= 0.0:
        return float(values[-1])
    return float(np.nansum(values * weights) / np.nansum(weights))


def plot_density(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, ax = plt.subplots(figsize=(8.3, 6.2), constrained_layout=True)

    for record in records:
        radius = record["radius"]
        density = record["density"]
        keep = np.isfinite(radius) & np.isfinite(density) & (radius > 0.0) & (density > 0.0)
        ax.plot(radius[keep], density[keep], lw=2.1, color=str(record["color"]), label=str(record["label"]))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Radius, $r$ ($R_\odot$)")
    ax.set_ylabel(r"Density, $\rho$ (g cm$^{-3}$)")
    ax.set_title(r"Initial Profile: $\log \rho$ vs $\log r$")
    style_paper_axis(ax)
    ax.legend(fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_log_density_vs_log_radius.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def plot_xi(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 6.0), constrained_layout=True)

    x = np.arange(len(ELEMENTS))
    width = 0.22
    offsets = np.linspace(-width, width, len(records))

    for offset, record in zip(offsets, records):
        xi = np.array([outer_average(record, element, outer_mass_msun=0.01) for element in ELEMENTS])
        ax.bar(
            x + offset,
            np.clip(xi, 1.0e-9, None),
            width=width * 0.92,
            color=str(record["color"]),
            label=str(record["label"]),
            alpha=0.9,
        )

    ax.set_yscale("log")
    ax.set_ylim(1.0e-8, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(ELEMENTS)
    ax.set_xlabel("Element type")
    ax.set_ylabel(r"Mass fraction, $X_i$")
    ax.set_title(r"Outer $0.01\,M_\odot$ Composition: $X_i$ by Element")
    style_paper_axis(ax)
    ax.legend(fontsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_xi_vs_element_type.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def write_summary(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    path = OUT_DIR / "low_ni_xi_outer_0p01_summary.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "label", "R_star_Rsun"] + [f"X_{element.replace(' ', '_')}" for element in ELEMENTS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "run_id": record["run_id"],
                "label": record["short_label"],
                "R_star_Rsun": f"{float(record['R_star_Rsun']):.6g}",
            }
            for element in ELEMENTS:
                row[f"X_{element.replace(' ', '_')}"] = f"{outer_average(record, element):.6e}"
            writer.writerow(row)
    return path


def main() -> None:
    records = [load_model(model) for model in MODELS]
    for output in [plot_density(records), plot_xi(records), write_summary(records)]:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
