# SNEC Setup Guide

This document records the verified installation of **SNEC v1.01** (2016-10-07) in this repository.

## Source and Data Downloads

| Package | URL | Size |
|---------|-----|------|
| SNEC source | `https://stellarcollapse.org/codes/SNEC-1.01-20161007.tar.gz` | ~1.5 MB |
| Example models | `https://stellarcollapse.org/simdata/models_and_parameters.tar.gz` | ~6.5 MB |

Commands used:

```bash
cd /Users/arifchu/Desktop/GitHub/supernova_theory
mkdir -p snec downloads

curl -L -o downloads/SNEC-1.01-20161007.tar.gz \
  "https://stellarcollapse.org/codes/SNEC-1.01-20161007.tar.gz"

curl -L -o downloads/models_and_parameters.tar.gz \
  "https://stellarcollapse.org/simdata/models_and_parameters.tar.gz"

tar -xzf downloads/SNEC-1.01-20161007.tar.gz -C snec --strip-components=1
tar -xzf downloads/models_and_parameters.tar.gz -C snec/models
```

**Note:** The models link on `Morozova2015.html` (`Morozova2015/simdata/...`) returns 404. The working path is `stellarcollapse.org/simdata/models_and_parameters.tar.gz`.

## Version

- **SNEC-1.01-20161007** (October 7, 2016)
- Physics identical to v1.00; ~2× faster due to opacity caching and routine optimizations
- Changelog: `snec/changes_in_SNEC-1.01.pdf`

## Compiler Requirements

| Tool | Required | This system |
|------|----------|-------------|
| `gfortran` | Yes | GNU Fortran 14.2.0 (Homebrew) |
| `make` | Yes | GNU Make 3.81 |
| LAPACK | Yes | macOS Accelerate framework |

No external libraries beyond the Fortran compiler and system LAPACK/Accelerate are needed. The bundled `libs.sh` references a legacy path and is **not** required on macOS.

## Build Procedure

```bash
cd snec
make
```

Build invokes `make -C src`, producing the executable at:

```
snec/snec
```

### Build configuration (`snec/make.inc`)

```makefile
F90=gfortran
F90FLAGS=-g -O3 -Warray-bounds -fbounds-check
LAPACKLIBS=-framework Accelerate   # macOS
# LAPACKLIBS=-llapack              # Linux
```

### Clean rebuild

```bash
cd snec
make clean && make
```

## Runtime Procedure

SNEC must be launched from the `snec/` directory (where `parameters` and `profiles/` live):

```bash
cd snec
mkdir -p Data          # output directory must exist before launch
./snec
```

On startup SNEC will:

1. Parse `parameters` in the current working directory
2. Verify `outdir` exists (stops with error if missing)
3. Wipe contents of `outdir` (default `wipe_outdir = .true.`)
4. Copy `parameters` into `outdir` for archival

### Required files for a simulation

| File | Purpose |
|------|---------|
| `parameters` | Runtime configuration |
| `profiles/<name>.short` | Stellar structure (mass, radius, density, temperature, velocity) |
| `profiles/<name>.iso.dat` | Composition profile (isotope mass fractions) |
| `tables/` | OPAL opacity tables (bundled) |
| `Data/` (or custom `outdir`) | Output directory (must pre-exist) |

### Default baseline model (15 M☉ RSG)

From `snec/parameters`:

```
outdir              = Data
profile_name        = profiles/15Msol_RSG.short
comp_profile_name   = profiles/15Msol_RSG.iso.dat
initial_data        = Thermal_Bomb
```

## Output Files

Scalar light-curve data (bolometric):

| File | Content |
|------|---------|
| `lum_observed.dat` | Bolometric luminosity (erg/s) vs time (s) — **primary light curve** |
| `lum_photo.dat` | Photospheric luminosity |
| `T_eff.dat` | Effective temperature |
| `Ni_total_luminosity.dat` | Nickel heating luminosity |
| `magnitudes.dat` | Band magnitudes (blackbody + bolometric corrections) |

Grid snapshots (`.xg` files) are written at intervals set by `dtout`.

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Output directory does not exist` | `outdir` not created | `mkdir -p Data` (or your chosen path) |
| `Parameter X could not be read` | Missing required entry in `parameters` | Add the parameter; doubles must contain a decimal point (e.g. `1.0d0`) |
| `No quotes in my strings, please!` | Quoted string in parameters | Remove quotes from path values |
| `mass fraction of Ni is lower than NIMIN` | `Ni_switch=1` but no Ni seeded | Set `Ni_mass > 0` or `Ni_switch = 0` |
| Opacity table build slow | First-run OPAL table construction | Normal; takes a few minutes at startup |
| `GridPattern.dat` line mismatch | `imax` ≠ grid file lines | Match `imax` to profile zones or use `gridding = from_file_by_mass` |
| Linux LAPACK link error | Wrong `LAPACKLIBS` | Switch to `-llapack` in `make.inc` |

## Verified Build (this machine)

```bash
cd snec && make
# Exit code: 0
# Executable: snec/snec (Mach-O 64-bit arm64, ~4.5 MB)
```

## References

- Morozova et al. 2015, ApJ 814, 63 — [SNEC I paper](http://adsabs.harvard.edu/abs/2015ApJ...814...63M)
- SNEC homepage: https://stellarcollapse.org/SNEC.html
- Code notes: https://stellarcollapse.org/codes/snec_notes-1.00.pdf
