#!/usr/bin/env python3
"""Run one SNEC model variant with chosen progenitor, energy, and Ni mass."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

PROFILE_PRESETS = {
    "rsg": ("profiles/15Msol_RSG.short", "profiles/15Msol_RSG.iso.dat"),
    "stripped": ("profiles/stripped_star.short", "profiles/stripped_star.iso.dat"),
    "avishai_bsg2": (
        "avishai_models/MW-M15M8.25P367-primary-BSG2.short",
        "avishai_models/MW-M15M8.25P367-primary-BSG2.iso.dat",
    ),
    "avishai_bsg3": (
        "avishai_models/MW-M22M5.5P367-primary-BSG3.short",
        "avishai_models/MW-M22M5.5P367-primary-BSG3.iso.dat",
    ),
    "avishai_wn3": (
        "avishai_models/MW-M25M13.75P4-primary-WN3.short",
        "avishai_models/MW-M25M13.75P4-primary-WN3.iso.dat",
    ),
}

SCE_TEND_DAYS = 10.0
SCE_DTOUT_SECONDS = 300.0
SCE_DTOUT_SCALAR_SECONDS = 60.0
SCE_DTOUT_CHECK_SECONDS = 300.0
SCE_DTMAX_SECONDS = 10.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def format_float_for_snec(value: float) -> str:
    text = f"{value:.6g}"
    if "e" in text or "E" in text:
        mantissa, exponent = text.lower().split("e")
        if "." not in mantissa:
            mantissa += ".0"
        return f"{mantissa}d{int(exponent):+03d}"
    if "." not in text:
        text += ".0"
    return text


def default_run_id(profile: str, energy_foe: float, ni_mass: float) -> str:
    energy_tag = f"{energy_foe:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    ni_tag = f"{int(round(ni_mass * 1000)):03d}"
    return f"{profile}_e{energy_tag}_ni{ni_tag}"


def patch_line(text: str, key: str, value: str) -> str:
    if not re.search(rf"^{re.escape(key)}\s*=", text, re.MULTILINE):
        raise ValueError(f"{key} not found in template parameters")
    return re.sub(
        rf"^{re.escape(key)}\s*=.*$",
        f"{key} = {value}",
        text,
        count=1,
        flags=re.MULTILINE,
    )


def patch_parameters(
    template: str,
    profile: str,
    energy_foe: float,
    ni_mass: float,
    outdir: str,
    tend_seconds: float | None = None,
    dtout_seconds: float | None = None,
    dtout_scalar_seconds: float | None = None,
    dtout_check_seconds: float | None = None,
    dtmax_seconds: float | None = None,
) -> str:
    profile_name, comp_profile_name = PROFILE_PRESETS[profile]
    text = template

    text = patch_line(text, "outdir", outdir)
    text = patch_line(text, "profile_name", f'"{profile_name}"')
    text = patch_line(text, "comp_profile_name", f'"{comp_profile_name}"')
    text = patch_line(text, "final_energy", f"{format_float_for_snec(energy_foe)}d51")

    ni_switch = 0 if ni_mass == 0.0 else 1
    text = patch_line(text, "Ni_switch", str(ni_switch))
    text = patch_line(text, "Ni_mass", f"{format_float_for_snec(ni_mass)} #(in solar mass)")

    if tend_seconds is not None:
        text = patch_line(text, "tend", format_float_for_snec(tend_seconds))
    if dtout_seconds is not None:
        text = patch_line(text, "dtout", format_float_for_snec(dtout_seconds))
    if dtout_scalar_seconds is not None:
        text = patch_line(text, "dtout_scalar", format_float_for_snec(dtout_scalar_seconds))
    if dtout_check_seconds is not None:
        text = patch_line(text, "dtout_check", format_float_for_snec(dtout_check_seconds))
    if dtmax_seconds is not None:
        text = patch_line(text, "dtmax", format_float_for_snec(dtmax_seconds))

    return text


def setup_run_directory(
    snec_dir: Path,
    runs_root: Path,
    run_id: str,
    params_content: str,
) -> Path:
    run_dir = runs_root / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    for name in ("snec", "profiles", "tables", "avishai_models"):
        target = snec_dir / name
        if name == "avishai_models":
            target = snec_dir.parent / name
        if target.exists():
            (run_dir / name).symlink_to(target.resolve())

    (run_dir / "parameters").write_text(params_content)
    (run_dir / "Data").mkdir()
    return run_dir


def run_snec(run_dir: Path, log_path: Path) -> tuple[bool, float]:
    start = datetime.now()
    with log_path.open("w") as log:
        log.write(f"Started: {start.isoformat()}\n")
        log.write(f"Command: ./snec (cwd={run_dir})\n\n")
        log.flush()
        result = subprocess.run(
            ["./snec"],
            cwd=run_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = (datetime.now() - start).total_seconds()
        log.write(f"\nFinished: {datetime.now().isoformat()}\n")
        log.write(f"Exit code: {result.returncode}\n")
        log.write(f"Elapsed: {elapsed:.1f} s\n")
    return result.returncode == 0, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_PRESETS),
        default="rsg",
        help="Progenitor profile preset",
    )
    parser.add_argument(
        "--energy-foe",
        type=float,
        default=1.0,
        help="Explosion final_energy in foe units, where 1 foe = 1e51 erg",
    )
    parser.add_argument(
        "--ni-mass",
        type=float,
        default=0.05,
        help="Ni_mass in solar masses",
    )
    parser.add_argument(
        "--run-id",
        help="Run directory name. Defaults to profile/energy/Ni tag.",
    )
    parser.add_argument(
        "--early-sce",
        action="store_true",
        help=(
            "Use high-cadence early-time output for shock-cooling emission: "
            f"tend={SCE_TEND_DAYS:g} d, dtout_scalar={SCE_DTOUT_SCALAR_SECONDS:g} s, "
            f"dtmax={SCE_DTMAX_SECONDS:g} s."
        ),
    )
    parser.add_argument(
        "--tend-days",
        type=float,
        help="Override total run duration in days",
    )
    parser.add_argument(
        "--dtout-s",
        type=float,
        help="Override full profile output cadence in seconds",
    )
    parser.add_argument(
        "--dtout-scalar-s",
        type=float,
        help="Override scalar output cadence in seconds",
    )
    parser.add_argument(
        "--dtout-check-s",
        type=float,
        help="Override check-output cadence in seconds",
    )
    parser.add_argument(
        "--dtmax-s",
        type=float,
        help="Override maximum internal hydrodynamic timestep in seconds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write parameters only; do not execute SNEC",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip when lum_observed.dat already exists",
    )
    args = parser.parse_args()

    root = repo_root()
    snec_dir = root / "snec"
    exe = snec_dir / "snec"
    if not exe.exists():
        raise FileNotFoundError(f"SNEC executable not found: {exe}. Run 'make' in snec/.")

    run_id = args.run_id or default_run_id(args.profile, args.energy_foe, args.ni_mass)
    runs_root = snec_dir / "runs" / "model_variants"
    logs_root = root / "results" / "model_variants" / "logs"
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    template = (snec_dir / "parameters").read_text()
    tend_days = args.tend_days
    dtout_s = args.dtout_s
    dtout_scalar_s = args.dtout_scalar_s
    dtout_check_s = args.dtout_check_s
    dtmax_s = args.dtmax_s

    if args.early_sce:
        tend_days = SCE_TEND_DAYS if tend_days is None else tend_days
        dtout_s = SCE_DTOUT_SECONDS if dtout_s is None else dtout_s
        dtout_scalar_s = (
            SCE_DTOUT_SCALAR_SECONDS if dtout_scalar_s is None else dtout_scalar_s
        )
        dtout_check_s = SCE_DTOUT_CHECK_SECONDS if dtout_check_s is None else dtout_check_s
        dtmax_s = SCE_DTMAX_SECONDS if dtmax_s is None else dtmax_s

    params_content = patch_parameters(
        template,
        profile=args.profile,
        energy_foe=args.energy_foe,
        ni_mass=args.ni_mass,
        outdir="Data",
        tend_seconds=None if tend_days is None else tend_days * 86400.0,
        dtout_seconds=dtout_s,
        dtout_scalar_seconds=dtout_scalar_s,
        dtout_check_seconds=dtout_check_s,
        dtmax_seconds=dtmax_s,
    )

    lc_path = runs_root / run_id / "Data" / "lum_observed.dat"
    if args.skip_existing and lc_path.exists():
        print(f"[SKIP] {run_id}: output exists at {lc_path}")
        return

    if args.dry_run:
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "parameters").write_text(params_content)
        print(f"[DRY] wrote parameters to {run_dir / 'parameters'}")
        return

    run_dir = setup_run_directory(snec_dir, runs_root, run_id, params_content)
    log_path = logs_root / f"{run_id}.log"

    print(
        f"Running {run_id}: profile={args.profile}, "
        f"energy={args.energy_foe:.2f} foe, Ni_mass={args.ni_mass:.3f} Msol"
    )
    ok, elapsed = run_snec(run_dir, log_path)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {run_id} - {elapsed:.0f} s - log: {log_path}")


if __name__ == "__main__":
    main()
