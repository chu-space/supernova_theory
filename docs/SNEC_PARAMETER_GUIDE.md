# SNEC Parameter Guide

Complete reference for every parameter in the default `snec/parameters` file (15 M☉ red supergiant baseline). Parameters are read by `src/input_parser.F90`.

**Format rules:** Double-precision values must contain a decimal point (e.g. `1.0d0`, not `1`). Logical flags use `0` (false) or `1` (true). String paths must not use quotes.

---

## Running SNEC

See [SNEC_SETUP.md](SNEC_SETUP.md) for installation. Summary:

```bash
cd snec && make          # build → snec/snec
mkdir -p Data            # required output directory
./snec                   # reads ./parameters
```

**Typical workflow:**

1. Choose or create stellar profiles in `profiles/`
2. Edit `parameters` (or copy to a run-specific file and symlink)
3. Create `outdir` if it does not exist
4. Run `./snec` from the `snec/` directory
5. Analyze scalar outputs (`lum_observed.dat`, `magnitudes.dat`, etc.)

SNEC copies `parameters` into `outdir` and wipes previous `outdir` contents on each run.

---

## Parameter Reference

### Launch

#### `outdir`

| Field | Value |
|-------|-------|
| **Physical meaning** | Directory for all simulation output |
| **Units** | Path string |
| **Default** | `"Data"` |
| **Effect on light curve** | None (storage only) |
| **Effect on explosion physics** | None |
| **Beginner-safe** | Yes — change to organize runs (e.g. `results/baseline`) |

---

### Stellar Profile

#### `profile_name`

| Field | Value |
|-------|-------|
| **Physical meaning** | Path to the stellar structure profile (`.short` format, GR1D-compatible) |
| **Units** | Path string |
| **Default** | `profiles/15Msol_RSG.short` |
| **Typical values** | `profiles/15Msol_RSG.short`, `profiles/stripped_star.short`, `profiles/sedov.short` |
| **Effect on light curve** | Sets progenitor radius, envelope mass, and density structure — controls plateau length, peak brightness, and rise time |
| **Effect on explosion physics** | Defines initial mass, radius, temperature, and velocity grid |
| **Beginner-safe** | Yes — swap between bundled profiles; do not edit `.short` files without understanding the format |

#### `comp_profile_name`

| Field | Value |
|-------|-------|
| **Physical meaning** | Path to composition profile (isotope mass fractions per zone) |
| **Units** | Path string |
| **Default** | `profiles/15Msol_RSG.iso.dat` |
| **Effect on light curve** | Hydrogen/helium content affects opacity, recombination, and recombination-powered plateau; must pair with matching `.short` file |
| **Effect on explosion physics** | Sets composition, electron fraction, mean atomic weight; required for Ni heating and OPAL opacities |
| **Beginner-safe** | Yes — use matching pairs from `profiles/` |

---

### Explosion Mechanism

#### `initial_data`

| Field | Value |
|-------|-------|
| **Physical meaning** | Method for initiating the explosion |
| **Units** | String keyword |
| **Default** | `Thermal_Bomb` |
| **Options** | `Thermal_Bomb`, `Piston_Explosion` |
| **Effect on light curve** | Bomb gives smoother energy deposition; piston can produce sharper shock breakout |
| **Effect on explosion physics** | Selects which energy-injection subroutine is active |
| **Beginner-safe** | Yes — keep `Thermal_Bomb` for RSG models |

#### `piston_vel`

| Field | Value |
|-------|-------|
| **Physical meaning** | Inward piston velocity at the inner boundary |
| **Units** | cm/s |
| **Default** | `5.0d9` (used only if `initial_data = Piston_Explosion`) |
| **Typical values** | ~10⁹ cm/s |
| **Effect on light curve** | Higher velocity → earlier, brighter shock breakout |
| **Effect on explosion physics** | Drives shock via inner boundary motion |
| **Beginner-safe** | N/A for Thermal Bomb runs |

#### `piston_tstart`

| Field | Value |
|-------|-------|
| **Physical meaning** | Time when piston motion begins |
| **Units** | s |
| **Default** | `0.0d0` |
| **Beginner-safe** | N/A for Thermal Bomb |

#### `piston_tend`

| Field | Value |
|-------|-------|
| **Physical meaning** | Time when piston motion ends |
| **Units** | s |
| **Default** | `1.0d-2` |
| **Beginner-safe** | N/A for Thermal Bomb |

#### `final_energy`

| Field | Value |
|-------|-------|
| **Physical meaning** | Total explosion energy. With `bomb_mode=1` (default): asymptotic total energy of star + bomb; bomb energy = `final_energy − initial_stellar_energy` |
| **Units** | erg |
| **Default** | `1.0d51` (~0.1 foe) |
| **Typical values** | `0.5d51` – `2.0d51` for core-collapse SNe |
| **Effect on light curve** | Higher energy → brighter, broader light curve; shorter plateau for fixed envelope |
| **Effect on explosion physics** | Sets amount of thermal energy deposited by the bomb |
| **Beginner-safe** | Moderate — small changes (±20%) are informative; large changes alter progenitor matching |

#### `bomb_tstart`

| Field | Value |
|-------|-------|
| **Physical meaning** | Start time of thermal energy deposition |
| **Units** | s |
| **Default** | `0.0d0` |
| **Effect on light curve** | Delays explosion onset |
| **Beginner-safe** | Yes |

#### `bomb_tend`

| Field | Value |
|-------|-------|
| **Physical meaning** | End time of thermal energy deposition |
| **Units** | s |
| **Default** | `0.1d0` |
| **Typical values** | 0.01 – 1 s |
| **Effect on light curve** | Longer deposition → smoother shock, slightly delayed peak |
| **Effect on explosion physics** | Controls exponential time profile of bomb heating |
| **Beginner-safe** | Yes — keep near default |

#### `bomb_mass_spread`

| Field | Value |
|-------|-------|
| **Physical meaning** | Mass extent over which bomb energy is deposited (starting at `bomb_start_point`) |
| **Units** | M☉ |
| **Default** | `0.1d0` |
| **Effect on light curve** | Broader deposition can smooth early-time behavior |
| **Effect on explosion physics** | Sets number of Lagrangian zones receiving bomb heating |
| **Beginner-safe** | Moderate |

#### `bomb_start_point`

| Field | Value |
|-------|-------|
| **Physical meaning** | Grid index where bomb energy deposition begins (innermost heated zone) |
| **Units** | Zone index (integer) |
| **Default** | `1` |
| **Effect on light curve** | Shifts where explosion initiates in mass coordinate |
| **Beginner-safe** | Caution — index depends on grid; keep at 1 for beginners |

#### `bomb_mode` (optional, SNEC 1.01+)

| Field | Value |
|-------|-------|
| **Physical meaning** | Interpretation of `final_energy` |
| **Units** | Integer |
| **Default** | `1` (if omitted) |
| **Options** | `1` = asymptotic total energy; `2` = bomb energy directly |
| **Beginner-safe** | Yes — omit or set to `1` |

---

### Grid

#### `imax`

| Field | Value |
|-------|-------|
| **Physical meaning** | Number of Lagrangian mass zones |
| **Units** | Integer |
| **Default** | `1000` |
| **Typical values** | 500 – 2000; must be consistent with `gridding` mode |
| **Effect on light curve** | Finer grid → more accurate diffusion; negligible if converged |
| **Effect on explosion physics** | Resolution of hydrodynamics and radiation |
| **Beginner-safe** | Caution — increasing imax with `uniform_by_mass` changes resolution; use `from_file_by_mass` with bundled profiles |

#### `gridding`

| Field | Value |
|-------|-------|
| **Physical meaning** | How mass zones are distributed |
| **Units** | String keyword |
| **Default** | `from_file_by_mass` |
| **Options** | `from_file_by_mass`, `uniform_by_mass` |
| **Effect on light curve** | Indirect via resolution near photosphere |
| **Beginner-safe** | Yes — keep `from_file_by_mass` for bundled profiles |

#### `mass_excision`

| Field | Value |
|-------|-------|
| **Physical meaning** | Whether to excise innermost mass (replace with boundary condition) |
| **Units** | Logical (0/1) |
| **Default** | `1` |
| **Effect on light curve** | Excision avoids proto-neutron star region not modeled in SNEC |
| **Beginner-safe** | Yes — keep enabled for stellar profiles |

#### `mass_excised`

| Field | Value |
|-------|-------|
| **Physical meaning** | Mass coordinate of inner boundary when excision is enabled |
| **Units** | M☉ |
| **Default** | `1.4` |
| **Typical values** | 1.2 – 1.5 M☉ |
| **Effect on light curve** | Affects inner boundary location; must be less than `Ni_boundary_mass` |
| **Beginner-safe** | Moderate — do not set below physical core mass |

---

### Evolution

#### `radiation`

| Field | Value |
|-------|-------|
| **Physical meaning** | Enable flux-limited diffusion radiation transport |
| **Units** | Logical (0/1) |
| **Default** | `1` |
| **Effect on light curve** | Required for realistic light curves; `0` gives pure hydrodynamics |
| **Beginner-safe** | Yes — always `1` for light-curve studies |

#### `eoskey`

| Field | Value |
|-------|-------|
| **Physical meaning** | Equation of state selector |
| **Units** | Integer |
| **Default** | `2` |
| **Options** | `1` = ideal gas; `2` = Paczyński EOS |
| **Effect on light curve** | Paczyński EOS is standard for stellar envelopes |
| **Beginner-safe** | Yes — keep `2` |

#### `Ni_switch`

| Field | Value |
|-------|-------|
| **Physical meaning** | Enable radioactive ⁵⁶Ni heating |
| **Units** | Integer (0/1) |
| **Default** | `1` |
| **Effect on light curve** | Powers radioactive tail; affects late-time plateau decline and peak if Ni is near surface |
| **Beginner-safe** | Yes — primary knob for Week 1 study is `Ni_mass`, not this switch |

#### `Ni_mass`

| Field | Value |
|-------|-------|
| **Physical meaning** | Total mass of ⁵⁶Ni to distribute in the ejecta (when `Ni_by_hand=1`) |
| **Units** | M☉ |
| **Default** | `0.05` |
| **Typical values** | 0.001 – 0.2 M☉ (observed SNe: ~0.01 – 0.1 M☉) |
| **Effect on light curve** | More Ni → brighter tail, slower plateau decline, slightly higher late-time luminosity |
| **Effect on explosion physics** | Adds internal energy via γ-ray deposition (Swartz et al. 1995 scheme) |
| **Beginner-safe** | **Yes — ideal parameter for first experiments** |

#### `Ni_boundary_mass`

| Field | Value |
|-------|-------|
| **Physical meaning** | Outer mass coordinate to which ⁵⁶Ni is uniformly distributed (from inner boundary) |
| **Units** | M☉ |
| **Default** | `3.0d0` |
| **Typical values** | 2 – 5 M☉; must exceed `mass_excised` |
| **Effect on light curve** | Concentrated Ni (low boundary) vs extended Ni (high boundary) changes heating distribution |
| **Beginner-safe** | Moderate — keep fixed when studying `Ni_mass` |

#### `Ni_period`

| Field | Value |
|-------|-------|
| **Physical meaning** | Time interval between nickel heating updates |
| **Units** | s |
| **Default** | `1.0d4` (~2.8 hours) |
| **Effect on light curve** | Smaller values → more accurate Ni heating; negligible if much less than dynamical times |
| **Beginner-safe** | Yes — keep default |

#### `Ni_by_hand` (optional, SNEC 1.01+)

| Field | Value |
|-------|-------|
| **Physical meaning** | How ⁵⁶Ni mass fractions are set |
| **Units** | Integer |
| **Default** | `1` (if omitted) |
| **Options** | `1` = use `Ni_mass` and `Ni_boundary_mass`; `0` = read from composition profile |
| **Beginner-safe** | Yes — keep `1` for parametric Ni studies |

#### `saha_ncomps`

| Field | Value |
|-------|-------|
| **Physical meaning** | Number of species treated in Saha ionization equilibrium |
| **Units** | Integer |
| **Default** | `3` |
| **Effect on light curve** | Affects recombination and opacity in H/He envelope |
| **Beginner-safe** | Yes — keep default |

#### `boxcar_smoothing`

| Field | Value |
|-------|-------|
| **Physical meaning** | Apply boxcar average to composition after Ni seeding |
| **Units** | Logical (0/1) |
| **Default** | `1` |
| **Effect on light curve** | Smooths sharp composition edges; affects Ni distribution boundary |
| **Beginner-safe** | Yes |

#### `opacity_floor_envelope`

| Field | Value |
|-------|-------|
| **Physical meaning** | Minimum opacity (cm²/g) in the hydrogen envelope |
| **Units** | cm²/g |
| **Default** | `0.01d0` |
| **Effect on light curve** | Higher floor → more trapped radiation, affects diffusion timescale |
| **Beginner-safe** | Caution — advanced tuning |

#### `opacity_floor_core`

| Field | Value |
|-------|-------|
| **Physical meaning** | Minimum opacity in the core/heavy-element regions |
| **Units** | cm²/g |
| **Default** | `0.24d0` |
| **Beginner-safe** | Caution — advanced tuning |

---

### Timing and Output

#### `ntmax`

| Field | Value |
|-------|-------|
| **Physical meaning** | Maximum number of time steps |
| **Units** | Integer |
| **Default** | `10000000000000` (effectively unlimited) |
| **Beginner-safe** | Yes |

#### `tend`

| Field | Value |
|-------|-------|
| **Physical meaning** | Simulation end time |
| **Units** | s |
| **Default** | `17000000.0d0` (~197 days) |
| **Effect on light curve** | Must exceed desired observation epoch (e.g. 200 days = 1.73×10⁷ s) |
| **Beginner-safe** | Yes |

#### `dtout`

| Field | Value |
|-------|-------|
| **Physical meaning** | Interval for full grid dumps (`.xg` files) |
| **Units** | s |
| **Default** | `1.7d4` (~0.2 days) |
| **Beginner-safe** | Yes — larger values reduce disk use |

#### `dtout_scalar`

| Field | Value |
|-------|-------|
| **Physical meaning** | Interval for scalar outputs (light curve, photosphere) |
| **Units** | s |
| **Default** | `1.7d3` (~0.02 days) |
| **Effect on light curve** | Sets light-curve sampling cadence |
| **Beginner-safe** | Yes |

#### `dtout_check`

| Field | Value |
|-------|-------|
| **Physical meaning** | Interval for checkpoint-style output |
| **Units** | s |
| **Default** | `1.7d3` |
| **Beginner-safe** | Yes |

#### `ntout`

| Field | Value |
|-------|-------|
| **Physical meaning** | Grid dump every N steps (−1 = use `dtout` only) |
| **Units** | Integer |
| **Default** | `-1` |
| **Beginner-safe** | Yes |

#### `ntout_scalar`

| Field | Value |
|-------|-------|
| **Physical meaning** | Scalar dump every N steps (−1 = use `dtout_scalar`) |
| **Units** | Integer |
| **Default** | `-1` |
| **Beginner-safe** | Yes |

#### `ntout_check`

| Field | Value |
|-------|-------|
| **Physical meaning** | Checkpoint every N steps |
| **Units** | Integer |
| **Default** | `-1` |
| **Beginner-safe** | Yes |

#### `ntinfo`

| Field | Value |
|-------|-------|
| **Physical meaning** | Print progress to stdout every N steps |
| **Units** | Integer |
| **Default** | `1000` |
| **Beginner-safe** | Yes |

#### `dtmin`

| Field | Value |
|-------|-------|
| **Physical meaning** | Minimum allowed timestep |
| **Units** | s |
| **Default** | `1.0d-10` |
| **Beginner-safe** | Yes — do not change |

#### `dtmax`

| Field | Value |
|-------|-------|
| **Physical meaning** | Maximum allowed timestep |
| **Units** | s |
| **Default** | `1.0d2` |
| **Beginner-safe** | Caution |

---

### Test Mode

#### `sedov`

| Field | Value |
|-------|-------|
| **Physical meaning** | Enable Sedov blast-wave test (disables gravity) |
| **Units** | Logical (0/1) |
| **Default** | `0` |
| **Beginner-safe** | Yes — keep `0` for astrophysical runs |

---

## Physics Overview

### Core-collapse supernovae

Massive stars (>8 M☉) develop iron cores that collapse when electron capture and photodisintegration remove pressure support. The collapse rebounds, launching a shock that propagates through the star. SNEC does not simulate collapse itself; it begins from a pre-explosion stellar profile and injects explosion energy via a piston or thermal bomb.

### Type II-P supernovae

Type II supernovae show hydrogen in their spectra. The **II-P** subtype exhibits a long **plateau** in the optical/bolometric light curve (~80–100 days), powered by the recombination of hydrogen in the extended envelope as the photosphere recedes through constant-ionization mass shells. SNEC models this via Lagrangian hydrodynamics, flux-limited diffusion, and Saha ionization.

### Red supergiant progenitors

The default `15Msol_RSG` profile is a 15 M☉ (ZAMS) star evolved with MESA to a red supergiant stage: large radius (~10³ R☉), extended hydrogen envelope, and low effective temperature. These properties produce the characteristic Type II-P plateau in SNEC.

### Thermal bomb explosions

With `initial_data = Thermal_Bomb`, SNEC deposits energy exponentially in time and mass (`bomb_profile.F90`). The parameter `final_energy` (with `bomb_mode=1`) sets the asymptotic total energy; the code computes the required bomb energy above the initial stellar binding energy. This mimics the deposition of explosion energy without resolving the central engine.

### Radioactive nickel heating

SNEC does not run a full nuclear network. Instead, ⁵⁶Ni is seeded by hand (`Ni_mass`, `Ni_boundary_mass`) or read from the composition file. The `nickel.F90` routine solves γ-ray transport in gray approximation (Swartz et al. 1995) and deposits energy locally.

### Ni-56 → Co-56 → Fe-56 decay

⁵⁶Ni decays to ⁵⁶Co (τ ≈ 8.8 days) then ⁵⁶Fe (τ ≈ 111 days), releasing γ-rays that thermalize in the ejecta. The resulting luminosity L ∝ M_Ni (after ~60 days, when positrons also contribute). This powers the **radioactive tail** at t ≳ 100 days.

### Why Ni-56 affects light curves

During the plateau, luminosity is dominated by envelope recombination. At late times, when the photosphere becomes optically thin to the Ni-powered layers, radioactive heating sustains emission. More ⁵⁶Ni → higher tail luminosity, slower decline from the plateau, and (for some progenitors) a brighter overall peak.

### Plateau phase

The plateau occurs when the photosphere recedes through hydrogen layers with nearly constant recombination temperature (~5000–6000 K). Duration depends on hydrogen envelope mass and explosion energy. For 15 M☉ RSG models with substantial H envelopes, SNEC produces plateaus of tens to ~100 days (Morozova et al. 2015).

### Radioactive tail phase

After the plateau, the light curve declines until Ni heating dominates, producing a linear decline in magnitude (slope ≈ 0.01 mag/day for typical Ni masses). The tail shape is sensitive to `Ni_mass` and Ni distribution.

### How SNEC models these processes

| Process | SNEC treatment |
|---------|----------------|
| Hydrodynamics | Lagrangian, artificial viscosity, optional gravity |
| Radiation | Flux-limited diffusion (`radiation=1`) |
| Opacities | OPAL tables + floors |
| Recombination | Saha ionization (H, He) |
| Explosion | Thermal bomb or piston |
| Ni heating | Gray γ-ray deposition, updated every `Ni_period` |
| Light curve | `lum_observed = L_photosphere + L_Ni_above_photosphere` |

---

## Quick Reference: Beginner-Safe Parameters

| Safe to vary | Keep fixed initially |
|--------------|---------------------|
| `Ni_mass` | `profile_name`, `comp_profile_name` |
| `outdir` | `final_energy` |
| `tend` | `imax`, `gridding` |
| `Ni_switch` (on/off) | `mass_excised`, opacity floors |
| `dtout_scalar` (sampling) | `bomb_start_point` |
