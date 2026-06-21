#!/usr/bin/env python3
"""Plot SNEC progenitor structure and composition profiles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

M_SUN = 1.98847e33
R_SUN = 6.957e10

PROFILE_PRESETS = {
    "rsg": ("15 Msol RSG", "snec/profiles/15Msol_RSG.short", "snec/profiles/15Msol_RSG.iso.dat"),
    "stripped": (
        "Practice stripped model",
        "snec/profiles/stripped_star.short",
        "snec/profiles/stripped_star.iso.dat",
    ),
    "avishai_bsg2": (
        "Avishai BSG2",
        "avishai_models/MW-M15M8.25P367-primary-BSG2.short",
        "avishai_models/MW-M15M8.25P367-primary-BSG2.iso.dat",
    ),
    "avishai_bsg3": (
        "Avishai BSG3",
        "avishai_models/MW-M22M5.5P367-primary-BSG3.short",
        "avishai_models/MW-M22M5.5P367-primary-BSG3.iso.dat",
    ),
    "avishai_wn3": (
        "Avishai WN3",
        "avishai_models/MW-M25M13.75P4-primary-WN3.short",
        "avishai_models/MW-M25M13.75P4-primary-WN3.iso.dat",
    ),
}

SELECTED_ISOTOPES = [
    (1, 1, "H-1"),
    (4, 2, "He-4"),
    (12, 6, "C-12"),
    (16, 8, "O-16"),
    (28, 14, "Si-28"),
    (52, 26, "Fe-52"),
    (56, 28, "Ni-56"),
]


@dataclass
class StellarProfile:
    name: str
    mass_msun: np.ndarray
    radius_rsun: np.ndarray
    temperature_k: np.ndarray
    density_cgs: np.ndarray
    comp_mass_msun: np.ndarray
    isotope_a: np.ndarray
    isotope_z: np.ndarray
    composition: np.ndarray


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_fortran_floats(line: str) -> list[float]:
    return [float(value.replace("D", "E").replace("d", "e")) for value in line.split()]


def load_profile(name: str, short_path: Path, iso_path: Path) -> StellarProfile:
    short = np.loadtxt(short_path, skiprows=1)
    if short.ndim == 1:
        short = short.reshape(1, -1)

    # .short columns: zone, mass(g), radius(cm), temperature(K), density(g/cm^3), ...
    mass_msun = short[:, 1] / M_SUN
    radius_rsun = short[:, 2] / R_SUN
    temperature_k = short[:, 3]
    density_cgs = short[:, 4]

    with iso_path.open() as f:
        header = f.readline().split()
        n_zones = int(header[0])
        n_comps = int(header[1])
        isotope_a = np.array(parse_fortran_floats(f.readline()))
        isotope_z = np.array(parse_fortran_floats(f.readline()))

    iso = np.loadtxt(iso_path, skiprows=3)
    if iso.ndim == 1:
        iso = iso.reshape(1, -1)

    if iso.shape[0] != n_zones or iso.shape[1] != n_comps + 2:
        raise ValueError(f"Unexpected composition shape in {iso_path}: {iso.shape}")

    return StellarProfile(
        name=name,
        mass_msun=mass_msun,
        radius_rsun=radius_rsun,
        temperature_k=temperature_k,
        density_cgs=density_cgs,
        comp_mass_msun=iso[:, 0] / M_SUN,
        isotope_a=isotope_a,
        isotope_z=isotope_z,
        composition=iso[:, 2:],
    )


def isotope_index(profile: StellarProfile, mass_number: int, charge: int) -> int | None:
    match = np.where(
        (np.rint(profile.isotope_a).astype(int) == mass_number)
        & (np.rint(profile.isotope_z).astype(int) == charge)
    )[0]
    if len(match) == 0:
        return None
    return int(match[0])


def setup_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def plot_structure(profiles: list[StellarProfile], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True, sharex=True)
    colors = plt.cm.Set2(np.linspace(0.1, 0.9, len(profiles)))

    for profile, color in zip(profiles, colors):
        axes[0].plot(profile.mass_msun, profile.density_cgs, label=profile.name, color=color)
        axes[1].plot(profile.mass_msun, profile.temperature_k, label=profile.name, color=color)
        axes[2].plot(profile.mass_msun, profile.radius_rsun, label=profile.name, color=color)

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Density (g cm$^{-3}$)")
    axes[0].set_title("Pre-Supernova Density Profile")

    axes[1].set_yscale("log")
    axes[1].set_ylabel("Temperature (K)")
    axes[1].set_title("Pre-Supernova Temperature Profile")

    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"Enclosed mass ($M_\odot$)")
    axes[2].set_ylabel(r"Radius ($R_\odot$)")
    axes[2].set_title("Pre-Supernova Radius Profile")

    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_composition(profiles: list[StellarProfile], output_path: Path) -> None:
    fig, axes = plt.subplots(
        len(profiles),
        1,
        figsize=(10, 4.5 * len(profiles)),
        constrained_layout=True,
        sharex=False,
    )
    if len(profiles) == 1:
        axes = np.array([axes])

    colors = plt.cm.tab10(np.linspace(0, 1, len(SELECTED_ISOTOPES)))

    for ax, profile in zip(axes, profiles):
        for (mass_number, charge, label), color in zip(SELECTED_ISOTOPES, colors):
            idx = isotope_index(profile, mass_number, charge)
            if idx is None:
                continue
            y = profile.composition[:, idx]
            if np.nanmax(y) < 1.0e-6:
                continue
            ax.plot(
                profile.comp_mass_msun,
                y,
                color=color,
                linewidth=1.5,
                label=label,
            )

        ax.set_title(f"{profile.name}: Composition vs Enclosed Mass")
        ax.set_xlabel(r"Enclosed mass ($M_\odot$)")
        ax.set_ylabel("Mass fraction")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(PROFILE_PRESETS),
        default=["rsg", "stripped"],
        help="Profile presets to compare",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "results" / "stellar_profiles",
        help="Directory for output PNGs",
    )
    args = parser.parse_args()

    setup_plot_style()
    root = repo_root()
    profiles = []

    for preset in args.profiles:
        label, short_rel, iso_rel = PROFILE_PRESETS[preset]
        profiles.append(
            load_profile(
                label,
                root / short_rel,
                root / iso_rel,
            )
        )

    plot_structure(profiles, args.output_dir / "stellar_structure_compare.png")
    plot_composition(profiles, args.output_dir / "stellar_composition_profiles.png")

    print(f"Saved stellar structure plot to {args.output_dir / 'stellar_structure_compare.png'}")
    print(f"Saved composition plot to {args.output_dir / 'stellar_composition_profiles.png'}")
    for profile in profiles:
        print(
            f"{profile.name}: final mass={profile.mass_msun[-1]:.2f} Msol, "
            f"surface radius={profile.radius_rsun[-1]:.1f} Rsun"
        )


if __name__ == "__main__":
    main()
