#!/usr/bin/env python3
"""Plot WN3 initial composition profiles for the low-Ni model set."""

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
M_SUN = 1.98847e33
R_SUN = 6.957e10


@dataclass(frozen=True)
class Model:
    run_id: str
    label: str
    color: str


MODELS = [
    Model("wn3_sce_bare_e1_ni001", "Bare WN3", "#2b6cb0"),
    Model("wn3_sce_mass_m0p01_r5_e1_ni001", r"+0.01 $M_\odot$ to 5 $R_\odot$", "#d97706"),
    Model("wn3_sce_radius_m0p01_r50_e1_ni001", r"+0.01 $M_\odot$ to 50 $R_\odot$", "#15803d"),
]

SPECIES_FILES = {
    "H": "H_init_frac.dat",
    "He": "He_init_frac.dat",
    "C": "C_init_frac.dat",
    "O": "O_init_frac.dat",
    "Ni": "Ni_init_frac.dat",
    "Z": "metallicity_init.dat",
}

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


def load_two_column(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, 2)
    return data[:, 0], data[:, 1]


def load_model(model: Model) -> dict[str, np.ndarray | str | float]:
    data_dir = RUN_ROOT / model.run_id / "Data"
    _, mass_g = load_two_column(data_dir / "mass_initial.dat")
    _, radius_cm = load_two_column(data_dir / "rad_initial.dat")

    mass_msun = mass_g / M_SUN
    radius_rsun = radius_cm / R_SUN
    exterior_mass = np.nanmax(mass_msun) - mass_msun
    exterior_mass = np.maximum(exterior_mass, 1.0e-8)

    species: dict[str, np.ndarray] = {}
    for name, filename in SPECIES_FILES.items():
        _, frac = load_two_column(data_dir / filename)
        species[name] = frac

    other_metals = species["Z"] - species["C"] - species["O"] - species["Ni"]
    species["Other metals"] = np.maximum(other_metals, 0.0)

    return {
        "run_id": model.run_id,
        "label": model.label,
        "color": model.color,
        "mass_msun": mass_msun,
        "exterior_mass_msun": exterior_mass,
        "radius_rsun": radius_rsun,
        "species": species,
        "total_mass_msun": float(np.nanmax(mass_msun)),
        "surface_radius_rsun": float(np.nanmax(radius_rsun)),
    }


def surface_values(record: dict[str, np.ndarray | str | float]) -> dict[str, float]:
    species = record["species"]
    assert isinstance(species, dict)
    return {
        name: float(species[name][-1])
        for name in ["H", "He", "C", "O", "Other metals", "Ni"]
    }


def average_outer_values(
    record: dict[str, np.ndarray | str | float],
    outer_mass_msun: float = 0.01,
) -> dict[str, float]:
    species = record["species"]
    exterior_mass = record["exterior_mass_msun"]
    assert isinstance(species, dict)
    assert isinstance(exterior_mass, np.ndarray)
    mask = exterior_mass <= outer_mass_msun
    return {
        name: float(np.nanmean(species[name][mask]))
        for name in ["H", "He", "C", "O", "Other metals", "Ni"]
    }


def make_plot(records: list[dict[str, np.ndarray | str | float]]) -> Path:
    fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax_comp = fig.add_subplot(gs[0, :])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_radius = fig.add_subplot(gs[1, 1])

    # Composition in the outer layers. Use the 50 R_sun model as the representative
    # because the 5 R_sun extension has the same composition versus mass coordinate.
    representative = records[2]
    species = representative["species"]
    exterior_mass = representative["exterior_mass_msun"]
    assert isinstance(species, dict)
    assert isinstance(exterior_mass, np.ndarray)
    mask = exterior_mass <= 0.08
    for name in ["He", "C", "O", "Other metals", "H", "Ni"]:
        y = species[name][mask]
        ax_comp.plot(
            exterior_mass[mask],
            np.maximum(y, 1.0e-8),
            lw=2.0 if name in {"He", "C", "O", "Other metals"} else 1.4,
            color=SPECIES_COLORS[name],
            label=name,
        )
    ax_comp.axvspan(1.0e-8, 0.01, color="#e5e7eb", alpha=0.55, zorder=-1)
    ax_comp.text(
        0.001,
        0.035,
        r"outer 0.01 $M_\odot$ added region",
        fontsize=9,
        color="#374151",
    )
    ax_comp.set_xscale("log")
    ax_comp.set_yscale("log")
    ax_comp.set_xlim(1.0e-5, 0.08)
    ax_comp.set_ylim(1.0e-7, 1.2)
    ax_comp.invert_xaxis()
    ax_comp.set_xlabel(r"Exterior mass coordinate, $M_{\rm ext}=M_{\rm tot}-m$ ($M_\odot$)")
    ax_comp.set_ylabel("Mass fraction")
    ax_comp.set_title(r"Outer Composition of the WN3 Extension")
    ax_comp.legend(ncol=6, fontsize=8.5, loc="lower left")
    style_paper_axis(ax_comp)

    # Surface stacked bars.
    x = np.arange(len(records))
    bottoms = np.zeros(len(records))
    species_order = ["H", "He", "C", "O", "Other metals", "Ni"]
    for name in species_order:
        vals = np.array([surface_values(record)[name] for record in records])
        ax_bar.bar(
            x,
            vals,
            bottom=bottoms,
            color=SPECIES_COLORS[name],
            edgecolor="white",
            linewidth=0.4,
            label=name,
        )
        bottoms += vals
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(record["label"]) for record in records], rotation=20, ha="right")
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_ylabel("Surface mass fraction")
    ax_bar.set_title("Surface Composition")
    ax_bar.legend(fontsize=7.7, loc="upper right")
    style_paper_axis(ax_bar)

    # Radius placement: same chemistry, different outer radius.
    for record in records:
        exterior = record["exterior_mass_msun"]
        radius = record["radius_rsun"]
        assert isinstance(exterior, np.ndarray)
        assert isinstance(radius, np.ndarray)
        mask_r = exterior <= 0.08
        ax_radius.plot(
            exterior[mask_r],
            radius[mask_r],
            lw=2.0,
            color=str(record["color"]),
            label=str(record["label"]),
        )
    ax_radius.axvspan(1.0e-8, 0.01, color="#e5e7eb", alpha=0.55, zorder=-1)
    ax_radius.set_xscale("log")
    ax_radius.set_yscale("log")
    ax_radius.set_xlim(1.0e-5, 0.08)
    ax_radius.invert_xaxis()
    ax_radius.set_xlabel(r"Exterior mass coordinate ($M_\odot$)")
    ax_radius.set_ylabel(r"Radius ($R_\odot$)")
    ax_radius.set_title("Same Added Composition, Different Radius")
    ax_radius.legend(fontsize=8.0, loc="upper right")
    style_paper_axis(ax_radius)

    fig.suptitle("Low-Ni WN3 Initial Composition: Core Surface and Added Material", fontsize=15)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "low_ni_wn3_composition_materials.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def write_summary(records: list[dict[str, np.ndarray | str | float]]) -> Path:
    path = OUT_DIR / "low_ni_wn3_composition_summary.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "label",
        "surface_radius_Rsun",
        "surface_H",
        "surface_He",
        "surface_C",
        "surface_O",
        "surface_other_metals",
        "surface_Ni",
        "outer_0p01_H",
        "outer_0p01_He",
        "outer_0p01_C",
        "outer_0p01_O",
        "outer_0p01_other_metals",
        "outer_0p01_Ni",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            surf = surface_values(record)
            outer = average_outer_values(record, 0.01)
            writer.writerow(
                {
                    "run_id": record["run_id"],
                    "label": record["label"],
                    "surface_radius_Rsun": f"{float(record['surface_radius_rsun']):.8g}",
                    "surface_H": f"{surf['H']:.8e}",
                    "surface_He": f"{surf['He']:.8e}",
                    "surface_C": f"{surf['C']:.8e}",
                    "surface_O": f"{surf['O']:.8e}",
                    "surface_other_metals": f"{surf['Other metals']:.8e}",
                    "surface_Ni": f"{surf['Ni']:.8e}",
                    "outer_0p01_H": f"{outer['H']:.8e}",
                    "outer_0p01_He": f"{outer['He']:.8e}",
                    "outer_0p01_C": f"{outer['C']:.8e}",
                    "outer_0p01_O": f"{outer['O']:.8e}",
                    "outer_0p01_other_metals": f"{outer['Other metals']:.8e}",
                    "outer_0p01_Ni": f"{outer['Ni']:.8e}",
                }
            )
    return path


def main() -> None:
    records = [load_model(model) for model in MODELS]
    plot_path = make_plot(records)
    summary_path = write_summary(records)
    print(f"Saved plot to {plot_path}")
    print(f"Saved summary to {summary_path}")
    for record in records:
        surf = surface_values(record)
        print(
            f"{record['label']}: "
            f"H={surf['H']:.3e}, He={surf['He']:.6f}, C={surf['C']:.6f}, "
            f"O={surf['O']:.6f}, other metals={surf['Other metals']:.6f}, Ni={surf['Ni']:.3e}"
        )


if __name__ == "__main__":
    main()
