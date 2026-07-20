#!/usr/bin/env python3
"""Plot the initial material/composition structure of the low-Ni WN3 models."""

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

SPECIES_COLORS = {
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
    exterior_mass = np.clip(total_mass - mass, 1.0e-8, None)

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
        "R_star_Rsun": float(np.nanmax(radius)),
        "M_total_Msun": total_mass,
    }


def make_plot(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    bare, r5, r50 = records
    reference = r50

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), constrained_layout=True)
    ax_full, ax_outer, ax_radius, ax_density = axes.ravel()

    species_stack = ["He", "C", "O", "Other metals", "H", "Ni"]
    ax_full.stackplot(
        reference["mass"],
        [reference[name] for name in species_stack],
        labels=species_stack,
        colors=[SPECIES_COLORS[name] for name in species_stack],
        alpha=0.92,
        linewidth=0.0,
    )
    ax_full.set_xlim(0.0, float(reference["M_total_Msun"]))
    ax_full.set_ylim(0.0, 1.0)
    ax_full.set_xlabel(r"Enclosed mass ($M_\odot$)")
    ax_full.set_ylabel("Mass fraction")
    ax_full.set_title("Composition Through the WN3 Model")
    style_paper_axis(ax_full)
    ax_full.legend(fontsize=8, ncol=3, loc="lower center")

    outer_mask = reference["M_ext"] <= 3.0e-2
    for name in ["He", "O", "C", "Other metals", "H", "Ni"]:
        ax_outer.plot(
            reference["M_ext"][outer_mask],
            np.clip(reference[name][outer_mask], 1.0e-9, None),
            color=SPECIES_COLORS[name],
            lw=2.0,
            label=name,
        )
    ax_outer.set_xscale("log")
    ax_outer.set_yscale("log")
    ax_outer.invert_xaxis()
    ax_outer.set_xlim(3.0e-2, 1.0e-5)
    ax_outer.set_ylim(1.0e-7, 1.0)
    ax_outer.set_xlabel(r"Exterior mass, $M_{\rm ext}=M_{\rm tot}-m$ ($M_\odot$)")
    ax_outer.set_ylabel("Mass fraction")
    ax_outer.set_title("Outer Material Composition")
    style_paper_axis(ax_outer)
    ax_outer.legend(fontsize=8, ncol=2, loc="lower left")

    for record in records:
        mask = record["M_ext"] <= 3.0e-2
        ax_radius.plot(
            record["M_ext"][mask],
            record["radius"][mask],
            color=str(record["color"]),
            lw=2.2,
            label=str(record["label"]),
        )
    ax_radius.set_xscale("log")
    ax_radius.set_yscale("log")
    ax_radius.invert_xaxis()
    ax_radius.set_xlim(3.0e-2, 1.0e-5)
    ax_radius.set_xlabel(r"Exterior mass ($M_\odot$)")
    ax_radius.set_ylabel(r"Radius ($R_\odot$)")
    ax_radius.set_title("Same Surface Material Spread to Different Radii")
    style_paper_axis(ax_radius)
    ax_radius.legend(fontsize=8)

    for record in records:
        mask = record["M_ext"] <= 3.0e-2
        ax_density.plot(
            record["radius"][mask],
            np.clip(record["density"][mask], 1.0e-20, None),
            color=str(record["color"]),
            lw=2.2,
            label=str(record["label"]),
        )
    ax_density.set_xscale("log")
    ax_density.set_yscale("log")
    ax_density.set_xlabel(r"Radius ($R_\odot$)")
    ax_density.set_ylabel(r"Density (g cm$^{-3}$)")
    ax_density.set_title("Density Structure of the Outer Material")
    style_paper_axis(ax_density)

    fig.suptitle(
        r"Low-Ni WN3 Material Anatomy: He-rich, H-poor Surface Material",
        fontsize=14,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_wn3_material_anatomy.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def write_summary(records: list[dict[str, np.ndarray | float | str]]) -> Path:
    path = OUT_DIR / "low_ni_wn3_material_anatomy_summary.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "label",
        "M_total_Msun",
        "R_star_Rsun",
        "surface_H",
        "surface_He",
        "surface_C",
        "surface_O",
        "surface_other_metals",
        "surface_Ni",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "run_id": record["run_id"],
                    "label": record["short_label"],
                    "M_total_Msun": f"{float(record['M_total_Msun']):.6g}",
                    "R_star_Rsun": f"{float(record['R_star_Rsun']):.6g}",
                    "surface_H": f"{float(record['H'][-1]):.6e}",
                    "surface_He": f"{float(record['He'][-1]):.6e}",
                    "surface_C": f"{float(record['C'][-1]):.6e}",
                    "surface_O": f"{float(record['O'][-1]):.6e}",
                    "surface_other_metals": f"{float(record['Other metals'][-1]):.6e}",
                    "surface_Ni": f"{float(record['Ni'][-1]):.6e}",
                }
            )
    return path


def main() -> None:
    records = [load_model(model) for model in MODELS]
    for output in [make_plot(records), write_summary(records)]:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
