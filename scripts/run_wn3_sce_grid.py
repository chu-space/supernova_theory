#!/usr/bin/env python3
"""Generate and run a high-cadence WN3 shock-cooling grid.

The added-material profiles are controlled toy extensions of the bare Avishai
WN3 surface. They are meant to isolate how a small nearby wind/envelope changes
the first 0-48 hours of SNEC emission, not to replace a self-consistent stellar
evolution calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

M_SUN = 1.98847e33
R_SUN = 6.957e10

NI_MASS_MSUN = 0.05
TEND_DAYS = 2.0
DTOUT_SECONDS = 21600.0
DTOUT_SCALAR_SECONDS = 60.0
DTOUT_CHECK_SECONDS = 21600.0
DTMAX_SECONDS = 10.0
SHELL_ZONES = 180
TEMPERATURE_FLOOR_K = 1.0e4

BASE_SHORT = "avishai_models/MW-M25M13.75P4-primary-WN3.short"
BASE_ISO = "avishai_models/MW-M25M13.75P4-primary-WN3.iso.dat"


@dataclass(frozen=True)
class StructureSpec:
    key: str
    label: str
    profile: str
    role: str
    added_mass_msun: float
    outer_radius_rsun: float | None


STRUCTURES = [
    StructureSpec(
        key="bare",
        label="Bare Avishai WN3",
        profile="avishai_wn3",
        role="baseline",
        added_mass_msun=0.0,
        outer_radius_rsun=None,
    ),
    StructureSpec(
        key="mass_m0p01_r5",
        label="WN3 + 0.01 Msol to 5 Rsun",
        profile="avishai_wn3_ext_m0p01_r5",
        role="mass_variant",
        added_mass_msun=0.01,
        outer_radius_rsun=5.0,
    ),
    StructureSpec(
        key="radius_m0p01_r50",
        label="WN3 + 0.01 Msol to 50 Rsun",
        profile="avishai_wn3_ext_m0p01_r50",
        role="radius_variant",
        added_mass_msun=0.01,
        outer_radius_rsun=50.0,
    ),
]

ENERGIES_FOE = [0.5, 1.0, 2.0]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def energy_tag(energy_foe: float) -> str:
    return f"{energy_foe:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def ni_tag(ni_mass_msun: float) -> str:
    return f"{int(round(ni_mass_msun * 1000)):03d}"


def run_id(structure: StructureSpec, energy_foe: float) -> str:
    return f"wn3_sce_{structure.key}_e{energy_tag(energy_foe)}_ni{ni_tag(NI_MASS_MSUN)}"


def parse_iso_header(iso_path: Path) -> tuple[str, str, np.ndarray]:
    with iso_path.open() as f:
        header = f.readline().strip()
        a_line = f.readline().strip()
        z_line = f.readline().strip()
    return a_line, z_line, np.loadtxt(iso_path, skiprows=3)


def write_short(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"{len(rows)}\n")
        for row in rows:
            f.write(
                f"{int(row[0]):6d} "
                f"{row[1]:16.8e} {row[2]:16.8e} {row[3]:16.8e} "
                f"{row[4]:16.8e} {row[5]:16.8e} {row[6]:16.8e} {row[7]:16.8e}\n"
            )


def write_iso(path: Path, a_line: str, z_line: str, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_comps = rows.shape[1] - 2
    with path.open("w") as f:
        f.write(f"{len(rows)}   {n_comps}\n")
        f.write(f"{a_line}\n")
        f.write(f"{z_line}\n")
        for row in rows:
            values = " ".join(f"{value:16.8e}" for value in row)
            f.write(f"{values}\n")


def make_extended_profile(
    short: np.ndarray,
    iso: np.ndarray,
    added_mass_msun: float,
    outer_radius_rsun: float,
    shell_zones: int = SHELL_ZONES,
) -> tuple[np.ndarray, np.ndarray]:
    base = short[-1]
    base_mass = base[1]
    base_radius = base[2]
    base_temp = base[3]
    base_velocity = base[5]
    base_extra = base[6:]

    added_mass = added_mass_msun * M_SUN
    outer_radius = outer_radius_rsun * R_SUN
    if outer_radius <= base_radius:
        raise ValueError(
            f"Requested outer radius {outer_radius_rsun:g} Rsun is inside "
            f"the WN3 surface radius {base_radius / R_SUN:g} Rsun."
        )

    radius = np.linspace(base_radius, outer_radius, shell_zones + 1)[1:]
    mass = base_mass + added_mass * (radius - base_radius) / (outer_radius - base_radius)

    prev_radius = np.concatenate(([base_radius], radius[:-1]))
    prev_mass = np.concatenate(([base_mass], mass[:-1]))
    volume = (4.0 / 3.0) * np.pi * (radius**3 - prev_radius**3)
    density = (mass - prev_mass) / volume
    temperature = np.maximum(
        base_temp * (radius / base_radius) ** -0.5,
        TEMPERATURE_FLOOR_K,
    )

    shell = np.zeros((shell_zones, short.shape[1]))
    shell[:, 0] = np.arange(int(short[-1, 0]) + 1, int(short[-1, 0]) + shell_zones + 1)
    shell[:, 1] = mass
    shell[:, 2] = radius
    shell[:, 3] = temperature
    shell[:, 4] = density
    shell[:, 5] = base_velocity
    shell[:, 6:] = base_extra

    iso_shell = np.zeros((shell_zones, iso.shape[1]))
    iso_shell[:, 0] = mass
    iso_shell[:, 1] = radius
    iso_shell[:, 2:] = iso[-1, 2:]

    return np.vstack([short, shell]), np.vstack([iso, iso_shell])


def generate_profiles(root: Path) -> list[dict[str, float | str]]:
    short = np.loadtxt(root / BASE_SHORT, skiprows=1)
    iso_a_line, iso_z_line, iso = parse_iso_header(root / BASE_ISO)
    profile_dir = root / "snec" / "profiles" / "generated"
    summaries: list[dict[str, float | str]] = []

    for structure in STRUCTURES:
        if structure.role == "baseline":
            final_mass = short[-1, 1] / M_SUN
            final_radius = short[-1, 2] / R_SUN
        else:
            assert structure.outer_radius_rsun is not None
            extended_short, extended_iso = make_extended_profile(
                short,
                iso,
                added_mass_msun=structure.added_mass_msun,
                outer_radius_rsun=structure.outer_radius_rsun,
            )
            stem = structure.profile
            write_short(profile_dir / f"{stem}.short", extended_short)
            write_iso(profile_dir / f"{stem}.iso.dat", iso_a_line, iso_z_line, extended_iso)
            final_mass = extended_short[-1, 1] / M_SUN
            final_radius = extended_short[-1, 2] / R_SUN

        summaries.append(
            {
                "key": structure.key,
                "label": structure.label,
                "profile": structure.profile,
                "role": structure.role,
                "added_mass_msun": structure.added_mass_msun,
                "outer_radius_rsun": "" if structure.outer_radius_rsun is None else structure.outer_radius_rsun,
                "final_mass_msun": final_mass,
                "final_radius_rsun": final_radius,
            }
        )

    return summaries


def manifest_rows(root: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for structure in STRUCTURES:
        for energy_foe in ENERGIES_FOE:
            rid = run_id(structure, energy_foe)
            rows.append(
                {
                    "run_id": rid,
                    "label": structure.label,
                    "structure_key": structure.key,
                    "role": structure.role,
                    "profile": structure.profile,
                    "energy_foe": energy_foe,
                    "ni_mass_msun": NI_MASS_MSUN,
                    "tend_days": TEND_DAYS,
                    "dtout_s": DTOUT_SECONDS,
                    "dtout_scalar_s": DTOUT_SCALAR_SECONDS,
                    "dtout_check_s": DTOUT_CHECK_SECONDS,
                    "dtmax_s": DTMAX_SECONDS,
                    "data_dir": str(root / "snec" / "runs" / "model_variants" / rid / "Data"),
                }
            )
    return rows


def write_table(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rows, f, indent=2)


def run_grid(root: Path, dry_run: bool, skip_existing: bool) -> None:
    for row in manifest_rows(root):
        cmd = [
            sys.executable,
            str(root / "scripts" / "run_model_variant.py"),
            "--profile",
            str(row["profile"]),
            "--energy-foe",
            str(row["energy_foe"]),
            "--ni-mass",
            str(row["ni_mass_msun"]),
            "--run-id",
            str(row["run_id"]),
            "--early-sce",
            "--tend-days",
            str(row["tend_days"]),
            "--dtout-s",
            str(row["dtout_s"]),
            "--dtout-scalar-s",
            str(row["dtout_scalar_s"]),
            "--dtout-check-s",
            str(row["dtout_check_s"]),
            "--dtmax-s",
            str(row["dtmax_s"]),
        ]
        if dry_run:
            cmd.append("--dry-run")
        if skip_existing:
            cmd.append("--skip-existing")

        subprocess.run(cmd, cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write profile variants, manifest, and SNEC parameter files only.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a real SNEC run if its lum_observed.dat already exists.",
    )
    args = parser.parse_args()

    root = repo_root()
    output_dir = root / "results" / "wn3_sce_grid"

    profile_summary = generate_profiles(root)
    manifest = manifest_rows(root)

    write_table(output_dir / "wn3_profile_variants.csv", profile_summary)
    write_json(output_dir / "wn3_profile_variants.json", profile_summary)
    write_table(output_dir / "wn3_sce_grid_manifest.csv", manifest)
    write_json(output_dir / "wn3_sce_grid_manifest.json", manifest)

    run_grid(root, dry_run=args.dry_run, skip_existing=args.skip_existing)

    mode = "staged" if args.dry_run else "ran"
    print(f"WN3 SCE grid {mode}: {len(manifest)} runs", flush=True)
    print(f"Manifest: {output_dir / 'wn3_sce_grid_manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
