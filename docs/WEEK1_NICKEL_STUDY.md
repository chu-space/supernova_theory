# Week 1: Nickel Mass Study

## Objective

Investigate how varying ⁵⁶Ni mass affects the bolometric light curve of the default 15 M☉ red supergiant model (`15Msol_RSG`).

## Scientific Motivation

Radioactive ⁵⁶Ni is the primary power source for the late-time emission of Type II supernovae. Synthesized in the inner ejecta during explosion, it decays through ⁵⁶Co to ⁵⁶Fe, releasing γ-rays that thermalize and re-emit as optical radiation. The nickel mass is therefore a key observable-linked parameter:

- It sets the slope and normalization of the **radioactive tail** (t ≳ 100 days)
- It influences the **plateau decline rate** once recombination ends
- It is correlated with explosion energy and progenitor structure in nature

SNEC allows direct control of `Ni_mass` without rerunning stellar evolution, making it ideal for isolating nickel's effect on the light curve.

## Experimental Design

### Baseline model

| Setting | Value |
|---------|-------|
| Profile | `profiles/15Msol_RSG.short` |
| Composition | `profiles/15Msol_RSG.iso.dat` |
| Explosion | `Thermal_Bomb`, `final_energy = 1.0d51` |
| Ni distribution | `Ni_boundary_mass = 3.0 M☉`, `Ni_by_hand = 1` |

### Controlled variable

**`Ni_mass`** (M☉):

| Run ID | Ni_mass |
|--------|---------|
| ni_000 | 0.00 |
| ni_001 | 0.01 |
| ni_003 | 0.03 |
| ni_005 | 0.05 (default) |
| ni_010 | 0.10 |
| ni_015 | 0.15 |

All other parameters match the default `snec/parameters` file.

### Assumptions

1. Nickel is uniformly distributed from the inner boundary to `Ni_boundary_mass = 3 M☉`
2. Boxcar smoothing (`boxcar_smoothing = 1`) is applied after Ni seeding
3. `Ni_switch = 1` for all runs except `Ni_mass = 0` (where Ni heating is off by definition)
4. Bolometric luminosity from `lum_observed.dat` is the primary observable
5. Explosion energy and progenitor structure are identical across runs

## Quantities To Measure

For each run:

| Quantity | Definition |
|----------|------------|
| Peak luminosity | Maximum `lum_observed` |
| Time of peak | Time of maximum luminosity (days) |
| Plateau duration | Days where L > 0.85 × L(50 d), between 30 and 150 d |
| L at 100 days | Interpolated bolometric luminosity |
| L at 150 days | Interpolated bolometric luminosity |
| L at 200 days | Interpolated bolometric luminosity |

## Plots To Produce

1. **Overlaid light curves** — all Ni masses on one axes
2. **Peak luminosity vs Ni mass**
3. **Plateau duration vs Ni mass**
4. **Luminosity at 150 days vs Ni mass**

Saved under `results/week1_nickel_study/`.

## Expected Physical Trends

| Observable | Trend with increasing Ni_mass |
|------------|-------------------------------|
| Peak luminosity | Weak increase or nearly unchanged (plateau dominated by recombination) |
| Time of peak | Approximately unchanged |
| Plateau duration | Slight increase (Ni heating slows decline) |
| L at 100 days | Increase (Ni contribution growing) |
| L at 150 days | Clear increase |
| L at 200 days | Strong increase (tail dominated by Ni) |
| Tail decline rate | Slower (more heating) |

For `Ni_mass = 0`, the tail should drop sharply after the plateau with no radioactive powering.

## Running the Study

```bash
# Generate parameter files and run SNEC (long — ~hours for full grid)
python3 scripts/run_nickel_study.py

# Analyze completed runs and produce plots
python3 scripts/analyze_nickel_study.py
```

Individual runs are stored in `snec/runs/week1_nickel_study/ni_XXX/Data/`.

## References

- Morozova et al. 2015, ApJ 814, 63
- Swartz et al. 1995, ApJ 446, 766 (Ni γ-ray deposition)
- Arnett, W. D. 1982, ApJ 253, 785 (radioactive decay model)
