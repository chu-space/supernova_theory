# Supernova Type II Theory

## One-Sentence Project Core

This note develops supernova light-curve physics through 1D progenitor/explosion models, then connects the numerical outputs back to analytic scalings for diffusion, shock cooling, recombination, radioactive 56Ni/56Co heating, and progenitor structure.

## The Scientific Idea

The theory builds from the observational distinction between Type I and Type II supernovae toward a practical numerical project:

- Type I supernovae lack hydrogen in their spectra. Type Ia events are thermonuclear explosions of white dwarfs and are largely powered around peak by radioactive 56Ni -> 56Co -> 56Fe decay.
- Type II supernovae show hydrogen and usually come from core collapse of massive stars that retain some hydrogen-rich envelope.
- Type II-P events have a plateau: as the ejecta expands and cools, the hydrogen recombination front moves inward in mass coordinate and regulates the optical luminosity.
- Type II-L/IIb/Ib/Ic behavior reflects different amounts of remaining H/He envelope material, stripping, progenitor radius, and radioactive heating.
- The early light curve, especially the first hours to days, is shock-cooling emission from shock-heated outer layers. This phase is a direct probe of progenitor radius, envelope mass, envelope composition, opacity, and E/M.

In plain language: the project asks how the observed light curve encodes the exploding star's pre-supernova structure.

## Physical Backbone

### Diffusion versus expansion

Photons do not stream straight out of the ejecta. They random-walk through optically thick material:

- Optical depth: tau ~ kappa * rho * length.
- Diffusion time: t_diff ~ tau * r / c.
- In an expanding ejecta, r ~ v * t.
- A useful peak-time estimate comes from t_diff ~ t_exp, giving approximately:

```text
t_peak ~ sqrt(kappa * M / (v * c))
```

where kappa is opacity, M is ejecta mass, v is characteristic expansion velocity, and c is the speed of light.

This is the conceptual reason broader light curves often mean larger ejecta masses, higher opacities, or lower expansion velocities.

### Opacity and recombination

Opacity is a central control knob. A useful electron-scattering scale is:

```text
kappa_es ~ sigma_T / m_p ~ 0.4 cm^2/g
```

for ionized hydrogen-rich gas. Once hydrogen recombines, opacity drops sharply, photons escape more easily, and the observed light curve changes. That recombination physics is the heart of Type II-P plateaus.

### Shock-cooling emission

After shock breakout, the outer ejecta expands and cools. The early luminosity is powered by internal energy deposited by the shock, not mainly by radioactive decay. A useful scaling is:

```text
L_shock-cooling scales with R0 * (E_SN / M) / kappa
```

More explicitly, this resembles the common analytic scaling:

```text
L ~ (4*pi*c*R0/kappa) * (E_SN/M)
```

where R0 is progenitor radius and E_SN/M is explosion energy per unit ejecta mass. This makes early light curves especially valuable: a red supergiant radius can be orders of magnitude larger than a white dwarf or compact stripped star, so the first few days can strongly separate progenitor scenarios.

### Radioactive heating

For Type Ia and for the late-time tails of core-collapse events, radioactive decay matters:

- 56Ni decays to 56Co, then 56Co decays to 56Fe.
- A typical core-collapse 56Ni mass scale is ~0.05 M_sun.
- Type Ia light curves rise and decline largely through diffusion of radioactive energy, while Type II-P light curves have an additional recombination-regulated plateau.

## Suggested Modeling Workflow

The practical modeling workflow is:

1. Start from a progenitor model.
   - Use a stellar-evolution profile, likely from MESA or a downloaded model grid.
   - Inspect mass coordinate, radius, density, temperature, velocity, composition, and isotope profiles.
   - Pay special attention to the outermost layers, since early shock-cooling emission is sensitive to them.

2. Choose explosion parameters.
   - Explosion energy E_SN, often near 1e51 erg as a baseline.
   - Ejecta mass and remnant mass cut.
   - 56Ni mass and mixing/distribution.
   - Degree of envelope stripping: red supergiant, partially stripped Type IIb, helium star, Wolf-Rayet-like compact progenitor.

3. Run a 1D radiation-hydrodynamics / light-curve code.
   - SNEC-style models are a natural fit, and the same questions also overlap strongly with MESA+STELLA modeling literature.
   - SNEC is written in Fortran and produces bolometric and band light curves from progenitor, explosion energy, and 56Ni input.
   - Practical run issues include compiler setup, lowering the time step if negative pressures occur, choosing output cadence, and running the simulation out to roughly 200 days.

4. Analyze outputs.
   - Bolometric luminosity versus time.
   - Band light curves and photometric magnitudes.
   - Effective/color temperature.
   - Photospheric radius/velocity if available.
   - Internal profiles: mass, radius, density, temperature, velocity, composition.
   - Scalar summaries: peak time, early decline, plateau duration, day-50 luminosity, radioactive tail luminosity, and inferred 56Ni mass.

5. Connect back to physics.
   - Compare simulated peak times with t_peak ~ sqrt(kappa M / v c).
   - Compare early luminosity and temperature to shock-cooling scalings.
   - Compare plateau duration/luminosity with Type II-P scaling laws.
   - Vary one physical parameter at a time: R0, M_ej, E_SN, kappa/composition, H-envelope mass, 56Ni mass, 56Ni mixing.

## Why This Matters

The significance is that supernova light curves are one of the few ways to infer what the progenitor star looked like immediately before explosion. Direct progenitor detections are rare; early high-cadence light curves are becoming common. The project sits exactly in that observational window.

Scientifically, it helps answer:

- What progenitor radius and envelope structure produced the observed early emission?
- How much hydrogen/helium was left at explosion?
- Was the star stripped by winds, binary interaction, or pre-supernova mass loss?
- How much 56Ni was synthesized and how deeply was it mixed?
- Which observables actually break degeneracies between M_ej, R0, and E_SN?
- When can simple analytic scaling laws be trusted, and when do numerical models disagree?

This is also pedagogically strong: an observer can run a model, plot its light curve, inspect the ejecta profiles, and then explain the result from diffusion, recombination, and shock-cooling physics.

## Literature This Builds On

### Core classification and context

- [Gal-Yam 2017, "Observational and Physical Classification of Supernovae"](https://arxiv.org/abs/1611.09353)  
  Best orientation for Type I/II/Ia/Ib/Ic/IIb/IIn terminology and the link between observational classes and progenitors.

- [Janka 2012, "Explosion Mechanisms of Core-Collapse Supernovae"](https://arxiv.org/abs/1206.2503)  
  Background on the core-collapse engine. Not the light-curve project itself, but useful context.

- [Smartt 2009, "Progenitors of Core-Collapse Supernovae"](https://arxiv.org/abs/0908.0700)  
  Establishes why progenitor inference is hard and important.

### Analytic light-curve physics

- [Arnett 1982, "Type I supernovae. I - Analytic solutions for the early part of the light curve"](https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract)  
  Classic radioactive-diffusion foundation; source of the intuition behind Arnett-style peak arguments.

- [Arnett, Fryer, and Matheson 2016, "Pre-nebular light curves of type I supernovae"](https://arxiv.org/abs/1611.08746)  
  Modern discussion of pre-nebular Type I light-curve modeling and Arnett's rule.

- [Popov 1993, "Analytical modeling of plateau supernova light curves"](https://ui.adsabs.harvard.edu/abs/1993ApJ...414..712P/abstract)  
  Classic analytic Type II-P plateau model.

- [Kasen and Woosley 2009, "Type II Supernovae: Model Light Curves and Standard Candle Relationships"](https://arxiv.org/abs/0910.1590)  
  Numerical models and scaling relations for Type II plateau luminosity/duration.

### Shock breakout and shock-cooling emission

- [Nakar and Sari 2010, "Early supernovae light-curves following the shock-breakout"](https://arxiv.org/abs/1004.2496)  
  Foundational analytic treatment of post-breakout cooling emission.

- [Rabinak and Waxman 2011, "The early UV/Optical emission from core-collapse supernovae"](https://arxiv.org/abs/1002.3414)  
  Directly relevant to using early UV/optical light curves to infer progenitor radius and envelope composition.

- [Nakar and Piro 2014, "Supernovae with Two Peaks in the Optical Light Curve and the Signature of Progenitors with Low-mass Extended Envelopes"](https://arxiv.org/abs/1401.7013)  
  Key for double-peaked light curves and extended low-mass envelopes.

- [Piro 2015, "Using Double-peaked Supernova Light Curves to Study Extended Material"](https://arxiv.org/abs/1505.07103)  
  Useful for analytic first-peak fitting and extended material.

- [Piro, Haynie, and Yao 2020, "Shock Cooling Emission from Extended Material Revisited"](https://arxiv.org/abs/2007.08543)  
  Updated analytic model for shock-cooling emission from extended material.

- [Morag, Sapir, and Waxman 2022, "Shock cooling emission from explosions of red super-giants: I"](https://arxiv.org/abs/2207.06179)  
  Numerically calibrated analytic model for red-supergiant shock cooling; good for modern comparisons.

### Numerical modeling and codes

- [Paxton et al. 2011, "Modules for Experiments in Stellar Astrophysics (MESA)"](https://arxiv.org/abs/1009.1622)  
  Core MESA instrument paper for progenitor modeling.

- [Paxton et al. 2015, "MESA: Binaries, Pulsations, and Explosions"](https://arxiv.org/abs/1506.03146)  
  Useful for stripping, binaries, and explosion-oriented modeling.

- [Morozova et al. 2015, "Light Curves of Core-Collapse Supernovae with Substantial Mass Loss using SNEC"](https://arxiv.org/abs/1505.06746)  
  The key SNEC reference: open-source 1D Lagrangian radiation-hydro, bolometric and band light curves, mass stripping, plateau behavior, double-peaked IIb-like curves.

- [Goldberg, Bildsten, and Paxton 2019, "Inferring Explosion Properties from Type II-Plateau Supernova Light Curves"](https://arxiv.org/abs/1903.09114)  
  Very close to this project: MESA to shock breakout plus STELLA light curves, mapping observables to degenerate families of E, M, and R.

- [Goldberg and Bildsten 2020, "The Value of Progenitor Radius Measurements for Explosion Modeling of Type II-Plateau Supernovae"](https://arxiv.org/abs/2005.07290)  
  Excellent for understanding why radius/early-time constraints matter.

- [Fremling and Hinds 2026, "SuperSNEC"](https://arxiv.org/abs/2603.05680)  
  Recent, optional if the project becomes a large grid: accelerated SNEC-style light-curve production.

### Stripped-envelope and binary-channel context

- [Lyman et al. 2014, "Bolometric light curves and explosion parameters of 38 stripped-envelope core-collapse supernovae"](https://arxiv.org/abs/1406.3667)  
  Observational baseline for IIb/Ib/Ic bolometric properties and binary-stripping implications.

- [Woosley, Sukhbold, and Kasen 2020, "Model Light Curves for Type Ib and Ic Supernovae"](https://arxiv.org/abs/2009.06868)  
  Good comparison set if the model runs focus on stripped helium-star progenitors.

## Suggested Reading Order

1. Gal-Yam 2017 for classification vocabulary.
2. Morozova et al. 2015 for SNEC and this style of numerical experiment.
3. Rabinak and Waxman 2011 plus Nakar and Sari 2010 for shock-cooling intuition.
4. Nakar and Piro 2014, then Piro 2015 or Piro et al. 2020 for double peaks and extended material.
5. Goldberg et al. 2019 and Goldberg and Bildsten 2020 for MESA/STELLA degeneracies and progenitor radius constraints.
6. Kasen and Woosley 2009 for Type II plateau scaling relations.
7. Lyman et al. 2014 or Woosley et al. 2020 if the project pivots toward stripped-envelope IIb/Ib/Ic events.

## A Good First Mini-Project

Run a small parameter study with one progenitor model:

1. Fix a baseline progenitor and explosion energy.
2. Vary progenitor radius or envelope stripping.
3. Vary 56Ni mass and mixing separately.
4. Plot bolometric luminosity, band magnitudes, effective temperature, and photospheric radius/velocity.
5. Measure t_peak, early decline slope, plateau duration, and radioactive-tail luminosity.
6. Compare each trend to the analytic scalings above.

The clean scientific deliverable would be a short note or notebook titled something like:

"How progenitor radius, envelope mass, and 56Ni control early and plateau supernova light curves in 1D models."
