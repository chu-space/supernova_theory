#!/usr/bin/env python3
"""Generate Ni_mass parameter variants and launch SNEC runs for Week 1 study."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Low/mid/high set: enough to show the nickel-powered tail changing
# without spending a full day on a dense grid.
NI_MASSES = [0.00, 0.05, 0.15]
RUN_PREFIX = "ni"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ni_run_id(ni_mass: float) -> str:
    return f"{RUN_PREFIX}_{int(round(ni_mass * 1000)):03d}"


def patch_parameters(template: str, ni_mass: float, outdir: str) -> str:
    """Return parameters file content with updated Ni_mass and outdir."""
    text = template

    # outdir
    if re.search(r"^outdir\s*=", text, re.MULTILINE):
        text = re.sub(
            r'^outdir\s*=.*$',
            f'outdir              = {outdir}',
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        raise ValueError("outdir not found in template parameters")

    # Ni_mass — must include decimal point for SNEC parser
    ni_str = f"{ni_mass:.2f}".rstrip("0").rstrip(".")
    if "." not in ni_str:
        ni_str += ".0"
    # SNEC's Fortran parser does not reliably strip literal tab characters
    # before fixed-format floating point reads.
    ni_line = f"Ni_mass = {ni_str} #(in solar mass)"

    if re.search(r"^Ni_mass\s*=", text, re.MULTILINE):
        text = re.sub(r"^Ni_mass\s*=.*$", ni_line, text, count=1, flags=re.MULTILINE)
    else:
        raise ValueError("Ni_mass not found in template parameters")

    # Ni_switch: off when Ni_mass is zero
    ni_switch = 0 if ni_mass == 0.0 else 1
    if re.search(r"^Ni_switch\s*=", text, re.MULTILINE):
        text = re.sub(
            r"^Ni_switch\s*=.*$",
            f"Ni_switch = {ni_switch}",
            text,
            count=1,
            flags=re.MULTILINE,
        )

    return text


def run_snec(snec_dir: Path, run_dir: Path, log_path: Path) -> tuple[bool, float]:
    """Execute SNEC from run_dir. Returns (success, elapsed_seconds)."""
    exe = snec_dir / "snec"
    if not exe.exists():
        raise FileNotFoundError(f"SNEC executable not found: {exe}. Run 'make' in snec/.")

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


def setup_run_directory(
    snec_dir: Path,
    runs_root: Path,
    run_id: str,
    params_content: str,
) -> Path:
    """Create an isolated run directory with symlinks to shared SNEC assets."""
    run_dir = runs_root / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    # Symlink static assets
    for name in ("snec", "profiles", "tables"):
        target = snec_dir / name
        if target.exists():
            (run_dir / name).symlink_to(target.resolve())

    # Write parameters and create output dir
    (run_dir / "parameters").write_text(params_content)
    outdir = run_dir / "Data"
    outdir.mkdir()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--masses",
        nargs="+",
        type=float,
        default=NI_MASSES,
        help="Ni_mass values in solar masses",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate parameter files only; do not execute SNEC",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose lum_observed.dat already exists",
    )
    args = parser.parse_args()

    root = repo_root()
    snec_dir = root / "snec"
    template_path = snec_dir / "parameters"
    runs_root = snec_dir / "runs" / "week1_nickel_study"
    logs_root = root / "results" / "week1_nickel_study" / "logs"
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    template = template_path.read_text()
    summary_lines = [
        f"Week 1 Nickel Study — {datetime.now().isoformat()}",
        f"Template: {template_path}",
        "",
    ]

    for ni_mass in args.masses:
        run_id = ni_run_id(ni_mass)
        outdir_rel = "Data"
        params_content = patch_parameters(template, ni_mass, outdir_rel)

        lc_path = runs_root / run_id / "Data" / "lum_observed.dat"
        if args.skip_existing and lc_path.exists():
            summary_lines.append(f"[SKIP] {run_id} Ni_mass={ni_mass} — output exists")
            continue

        if args.dry_run:
            archive = runs_root / run_id / "parameters"
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_text(params_content)
            summary_lines.append(f"[DRY] {run_id} Ni_mass={ni_mass} — parameters written")
            continue

        # Full run directory with symlinks
        run_dir = setup_run_directory(snec_dir, runs_root, run_id, params_content)
        log_path = logs_root / f"{run_id}.log"

        print(f"Running {run_id} (Ni_mass={ni_mass} M☉)...")
        ok, elapsed = run_snec(snec_dir, run_dir, log_path)
        status = "OK" if ok else "FAIL"
        summary_lines.append(
            f"[{status}] {run_id} Ni_mass={ni_mass} — {elapsed:.0f} s — log: {log_path}"
        )
        if not ok:
            print(f"  WARNING: {run_id} failed (exit != 0). See {log_path}")

    summary_path = logs_root / "run_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
